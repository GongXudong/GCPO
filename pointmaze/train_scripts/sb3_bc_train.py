import sys
from pathlib import Path
from copy import deepcopy
import argparse
from typing import List

import numpy as np
import gymnasium as gym
import minari
import torch as th

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure, Logger
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.ppo import MlpPolicy, MultiInputPolicy

from imitation.algorithms import bc
from imitation.util.logger import HierarchicalLogger
from imitation.util import util
from imitation.data import rollout, types
from imitation.data.types import TransitionsMinimal, Transitions, Trajectory

PROJECT_ROOT_DIR = Path(__file__).parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from models.sb3_model import PPOWithBCLoss
from utils.sb3_env_wrappers import ScaledObservationWrapper
from configs.load_config import load_config
from utils.sb3_schedule import linear_schedule
from utils.load_data import load_data, scale_obs, split_data
# from rollout.load_data import load_data as load_data_from_hd5



def get_ppo_algo(env):
    policy_kwargs = dict(
        full_std=True,  # 使用state dependant exploration
        # squash_output=True,  # 使用state dependant exploration
        net_arch=dict(
            pi=NET_ARCH,
            vf=deepcopy(NET_ARCH)
        ),
        activation_fn=th.nn.Tanh,
        ortho_init=True,
        optimizer_class=th.optim.Adam,
        optimizer_kwargs={
            "eps": 1e-5
        }
    )

    return PPOWithBCLoss(
        policy=MultiInputPolicy, 
        env=env, 
        bc_trained_algo=None,
        kl_coef_with_bc=KL_WITH_BC_MODEL_COEF,
        seed=RL_SEED,
        batch_size=PPO_BATCH_SIZE,
        gamma=GAMMA,
        n_steps=128,  # 采样时每个环境采样的step数
        n_epochs=5,  # 采样的数据在训练中重复使用的次数
        policy_kwargs=policy_kwargs,
        use_sde=True,  # 使用state dependant exploration
        normalize_advantage=True,
        learning_rate=linear_schedule(3e-4),
        device="cuda"
    )


# strategy for save policy，根据最小的loss保存。
# 因为bc.train()方法的on_batch_end是没有参数的回调函数，所以这里使用闭包，通过一个外部变量记录最小的loss
def on_best_loss_save(algo: PPO, validation_transitions: TransitionsMinimal, loss_calculator: bc.BehaviorCloningLossCalculator, sb3_logger: Logger):
    min_loss = LOSS_THRESHOLD  # 预估一个初始值，不然训练开始阶段会浪费大量的时间在存储模型上！！！
    def calc_func():
        algo.policy.set_training_mode(mode=False)
        
        nonlocal min_loss
        
        obs_tensor = types.map_maybe_dict(
            lambda x: util.safe_to_tensor(x, device="cuda"),
            types.maybe_unwrap_dictobs(validation_transitions.obs),
        )
        acts_tensor = util.safe_to_tensor(validation_transitions.acts).to("cuda")
        
        metrics: bc.BCTrainingMetrics = loss_calculator(policy=algo.policy, obs=obs_tensor, acts=acts_tensor)
        cur_loss = metrics.loss
        if cur_loss < min_loss:
            sb3_logger.info(f"update loss from {min_loss} to {cur_loss}!")
            min_loss = cur_loss

            # save policy
            checkpoint_save_dir = PROJECT_ROOT_DIR / "checkpoints" / "bc" / EXPERIMENT_NAME
            if not checkpoint_save_dir.exists():
                checkpoint_save_dir.mkdir()

            algo.save(str(checkpoint_save_dir / POLICY_FILE_SAVE_NAME))

        algo.policy.set_training_mode(mode=True)
    return calc_func


def train():

    sb3_logger: Logger = configure(folder=str((PROJECT_ROOT_DIR / "logs" / "bc" / EXPERIMENT_NAME).absolute()), format_strings=['stdout', 'log', 'csv', 'tensorboard'])

    origin_env = gym.make(ENV_NAME, continuing_task=False)
    
    if EXPERT_DATA_CACHE_DIR == "cache":
        sb3_logger.info("load data from Minari cache.")
        trajs: List[Trajectory] = load_data(DATASET_MINARI_ID)
        transitions = rollout.flatten_trajectories(trajs)

    else:
        raise NotImplementedError()

    scaled_obs, scaler = scale_obs(transitions.obs)
    
    scaled_transitions = Transitions(
        obs=scaled_obs,
        acts=transitions.acts,
        infos=transitions.infos,
        next_obs=transitions.next_obs,
        dones=transitions.dones
    )

    env = ScaledObservationWrapper(env=origin_env, scaler=scaler)

    algo_ppo = get_ppo_algo(env)
    sb3_logger.info(str(algo_ppo.policy))
    # print(algo_ppo.policy)

    rng = np.random.default_rng(SEED)
    
    
    train_transitions, validation_transitions, test_transitions = split_data(
        transitions=scaled_transitions,
        train_size=DATASET_SPLIT[0],
        validation_size=DATASET_SPLIT[1],
        test_size=DATASET_SPLIT[2],
        shuffle=True,
    )

    sb3_logger.info(f"train_set: obs size, {train_transitions.obs.shape}, act size, {train_transitions.acts.shape}")
    sb3_logger.info(f"validation_set: obs size, {validation_transitions.obs.shape}, act size, {validation_transitions.acts.shape}")
    sb3_logger.info(f"test_set: obs size, {test_transitions.obs.shape}, act size, {test_transitions.acts.shape}")

    # evaluate with environment
    reward, _ = evaluate_policy(algo_ppo.policy, env, n_eval_episodes=10)
    sb3_logger.info(f"Reward before BC: {reward}")

    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        policy=algo_ppo.policy,
        batch_size=BC_BATCH_SIZE,
        ent_weight=BC_ENT_WEIGHT,
        l2_weight=BC_L2_WEIGHT,
        demonstrations=train_transitions,
        rng=rng,
        device="cuda",
        custom_logger=HierarchicalLogger(sb3_logger)
    )

    # train
    bc_trainer.train(
        n_epochs=TRAIN_EPOCHS,
        # on_batch_end=on_best_act_prob_save(algo_ppo, validation_transitions, sb3_logger),
        on_batch_end=on_best_loss_save(algo_ppo, validation_transitions, bc_trainer.loss_calculator, sb3_logger),
    )

    # evaluate with environment
    reward, _ = evaluate_policy(algo_ppo.policy, env, n_eval_episodes=10)
    sb3_logger.info(f"Reward after BC: {reward}")
    # sb3_logger.info(f"Reward after BC: {reward}, normalized: {origin_env.get_normalized_score(reward)}")

    # 最终的policy在测试集上的prob_true_act / loss
    test_on_loss(algo_ppo.policy, test_transitions, bc_trainer.loss_calculator, sb3_logger, "训练结束时的策略", "测试集")

    return sb3_logger, validation_transitions, test_transitions, bc_trainer, origin_env, env


def test_on_loss(
        policy: MlpPolicy, 
        test_transitions: TransitionsMinimal, 
        loss_calculator: bc.BehaviorCloningLossCalculator, 
        sb3_logger: Logger, 
        policy_descreption: str, dataset_descreption: str
    ):
    policy.set_training_mode(mode=False)

    # obs = util.safe_to_tensor(test_transitions.obs).to("cuda")
    # print("here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ", test_transitions.obs["observation"].shape)
    obs_tensor = types.map_maybe_dict(
        lambda x: util.safe_to_tensor(x, device="cuda"),
        types.maybe_unwrap_dictobs(test_transitions.obs),
    )
    acts_tensor = util.safe_to_tensor(test_transitions.acts).to("cuda")
    
    metrics: bc.BCTrainingMetrics = loss_calculator(policy=policy, obs=obs_tensor, acts=acts_tensor)
    sb3_logger.info(f"{policy_descreption}在{dataset_descreption}上的loss: {metrics.loss}.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="传入配置文件")
    parser.add_argument("--config-file-name", type=str, help="配置文件名", default="configs/pointmaze-dense/seed1/256_256.json")
    args = parser.parse_args()

    custom_config = load_config(args.config_file_name)

    ENV_NAME = custom_config["env"]["name"]

    EXPERIMENT_NAME = custom_config["bc"]["experiment_name"]
    SEED = custom_config["bc"]["seed"]
    POLICY_FILE_SAVE_NAME = custom_config["bc"]["policy_file_save_name"]
    TRAIN_EPOCHS = custom_config["bc"]["train_epochs"]
    BC_BATCH_SIZE = custom_config["bc"]["batch_size"]
    BC_L2_WEIGHT = custom_config["bc"].get("l2_weight", 0.0)
    BC_ENT_WEIGHT = custom_config["bc"].get("ent_weight", 1e-3)
    EXPERT_DATA_CACHE_DIR = custom_config["bc"]["data_cache_dir"]
    DATASET_MINARI_ID = custom_config["bc"]["dataset_minari_id"]
    LOSS_THRESHOLD = custom_config["bc"]["loss_threshold"]
    DATASET_SPLIT = custom_config["bc"].get("dataset_split", [0.96, 0.02, 0.02])

    RL_SEED = custom_config["rl"]["seed"]
    NET_ARCH = custom_config["rl_bc"]["net_arch"]
    PPO_BATCH_SIZE = custom_config["rl_bc"]["batch_size"]
    GAMMA = custom_config["rl_bc"]["gamma"]
    KL_WITH_BC_MODEL_COEF = custom_config["rl_bc"]["kl_with_bc_model_coef"]


    sb3_logger, validation_transitions, test_transitions, bc_trainer, origin_env, env = train()

    policy_save_dir = PROJECT_ROOT_DIR / "checkpoints" / "bc" / EXPERIMENT_NAME
    algo_ppo = PPOWithBCLoss.load(str((policy_save_dir / POLICY_FILE_SAVE_NAME).absolute()))

    test_on_loss(algo_ppo.policy, validation_transitions, bc_trainer.loss_calculator, sb3_logger, "最优策略", "验证集")
    test_on_loss(algo_ppo.policy, test_transitions, bc_trainer.loss_calculator, sb3_logger, "最优策略", "测试集")
    
    reward, _ = evaluate_policy(algo_ppo.policy, env, n_eval_episodes=10)
    sb3_logger.info(f"最优策略得分: {reward}")
    # sb3_logger.info(f"最优策略得分: {reward}, 标准化后的得分: {origin_env.get_normalized_score(reward)}")
