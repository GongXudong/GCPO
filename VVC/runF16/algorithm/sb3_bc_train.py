import gymnasium as gym
import numpy as np
from pathlib import Path
import logging
import torch as th
import argparse
from copy import deepcopy
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure, Logger
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.ppo import MlpPolicy
from imitation.algorithms import bc
from imitation.util.logger import HierarchicalLogger
from imitation.util import util
from imitation.data.types import TransitionsMinimal

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from runF16.algorithm.models.sb3_model import PPOWithBCLoss
from runF16.env.utils.sb3_utils import get_vec_env_dense_reward
from runF16.algorithm.utils.sb3_evaluate_policy import evaluate_policy_with_success_rate
from runF16.algorithm.data_sb3.load_data import load_data_from_cache
from runF16.algorithm.data_sb3.load_data_by_difficulty import load_data_by_difficulty
from runF16.algorithm.data_sb3.load_data_by_chi import load_data_by_chi
from runF16.utils_my.my_log import get_logger
from runF16.configs.load_config import load_config
from runF16.algorithm.utils.sb3_schedule import linear_schedule

parser = argparse.ArgumentParser(description="传入配置文件")
parser.add_argument("--config_file_name", type=str, help="配置文件名", default="config_10hz_loss_128_128.json")
parser.add_argument("--train_on_difficulty_task_set", action="store_true", help="是否在指定难度的小任务集上训练bc")
parser.add_argument("--train_on_chi_set", action="store_true", help="是否在指定chi的小任务集上训练bc")
parser.add_argument("--difficulty", type=int, default=0, help="训练要使用的任务难度")
parser.add_argument("--task_set_capacity", type=int, default=100, help="训练要使用的任务数")
parser.add_argument("--target_chi", type=int, default=0, help="训练要使用的chi")
args = parser.parse_args()

custom_config = load_config(args.config_file_name)

EXPERIMENT_NAME = custom_config["bc"]["experiment_name"]
SEED = custom_config["bc"]["seed"]
POLICY_FILE_SAVE_NAME = custom_config["bc"]["policy_file_save_name"]
TRAIN_EPOCHS = custom_config["bc"]["train_epochs"]
BC_BATCH_SIZE = custom_config["bc"]["batch_size"]
BC_L2_WEIGHT = custom_config["bc"].get("l2_weight", 0.0)
BC_ENT_WEIGHT = custom_config["bc"].get("ent_weight", 1e-3)
EXPERT_DATA_CACHE_DIR = custom_config["bc"]["data_cache_dir"]
STEP_FREQUENCE = custom_config["step_frequence"]  # 用在测试环境中
EXPERT_DATA_DIR = custom_config["rollout"]["expert_data_dir"]
USE_LOSS_CALLBACK = custom_config["bc"]["use_loss_callback"]
PROB_TRUE_ACT_THRESHOLD = custom_config["bc"]["prob_true_act_threshold"]  # validate的时候，当prob_true_act大于这个值的时候，开始保存prob_true_act最优的policy
LOSS_THRESHOLD = custom_config["bc"]["loss_threshold"]

MAX_SIMULATE_TIME = custom_config["env"]['max_simulate_time']

RL_SEED = custom_config["rl"]["seed"]
NET_ARCH = custom_config["rl_bc"]["net_arch"]
PPO_BATCH_SIZE = custom_config["rl_bc"]["batch_size"]
GAMMA = custom_config["rl_bc"]["gamma"]
KL_WITH_BC_MODEL_COEF = custom_config["rl_bc"]["kl_with_bc_model_coef"]

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
        policy=MlpPolicy, 
        env=env, 
        bc_trained_algo=None,
        kl_coef_with_bc=KL_WITH_BC_MODEL_COEF,
        seed=RL_SEED,
        batch_size=PPO_BATCH_SIZE,
        gamma=GAMMA,
        n_steps=2048,  # 采样时每个环境采样的step数
        n_epochs=5,  # 采样的数据在训练中重复使用的次数
        policy_kwargs=policy_kwargs,
        use_sde=True,  # 使用state dependant exploration
        normalize_advantage=True,
        learning_rate=linear_schedule(3e-4),
    )


# strategy for save policy，有两个选择：（1）根据最大的prob_true_act保存；（2）根据最小的loss保存。
# 因为bc.train()方法的on_batch_end是没有参数的回调函数，所以这里使用闭包，通过一个外部变量记录最高的prob_true_act
def on_best_act_prob_save(algo: PPO, validation_transitions: TransitionsMinimal, sb3_logger: Logger):
    best_prob = PROB_TRUE_ACT_THRESHOLD  # 预估一个初始值，不然训练开始阶段会浪费大量的时间在存储模型上！！！
    def calc_func():
        algo.policy.set_training_mode(mode=False)
        
        nonlocal best_prob
        
        obs = util.safe_to_tensor(validation_transitions.obs).to("cuda")
        acts = util.safe_to_tensor(validation_transitions.acts).to("cuda")
        _, log_prob, entropy = algo.policy.evaluate_actions(obs, acts)
        prob_true_act = th.exp(log_prob).mean()
        if prob_true_act > best_prob:
            sb3_logger.info(f"update prob true act from {best_prob} to {prob_true_act}!")
            best_prob = prob_true_act

            # save policy
            checkpoint_save_dir = Path(__file__).parent / "checkpoints_sb3" / "bc" / EXPERIMENT_NAME
            if not checkpoint_save_dir.exists():
                checkpoint_save_dir.mkdir()

            algo.save(str(checkpoint_save_dir / POLICY_FILE_SAVE_NAME))

        algo.policy.set_training_mode(mode=True)
    return calc_func

def on_best_loss_save(algo: PPO, validation_transitions: TransitionsMinimal, loss_calculator: bc.BehaviorCloningLossCalculator, sb3_logger: Logger):
    min_loss = LOSS_THRESHOLD  # 预估一个初始值，不然训练开始阶段会浪费大量的时间在存储模型上！！！
    def calc_func():
        algo.policy.set_training_mode(mode=False)
        
        nonlocal min_loss
        
        obs = util.safe_to_tensor(validation_transitions.obs).to("cuda")
        acts = util.safe_to_tensor(validation_transitions.acts).to("cuda")
        
        metrics: bc.BCTrainingMetrics = loss_calculator(policy=algo.policy, obs=obs, acts=acts)
        cur_loss = metrics.loss
        if cur_loss < min_loss:
            sb3_logger.info(f"update loss from {min_loss} to {cur_loss}!")
            min_loss = cur_loss

            # save policy
            checkpoint_save_dir = Path(__file__).parent / "checkpoints_sb3" / "bc" / EXPERIMENT_NAME
            if not checkpoint_save_dir.exists():
                checkpoint_save_dir.mkdir()

            algo.save(str(checkpoint_save_dir / POLICY_FILE_SAVE_NAME))

        algo.policy.set_training_mode(mode=True)
    return calc_func


def train():

    sb3_logger: Logger = configure(folder=str((PROJECT_ROOT_DIR / "my_rl_logs" / "bc" / EXPERIMENT_NAME).absolute()), format_strings=['stdout', 'log', 'csv', 'tensorboard'])

    vec_env = get_vec_env_dense_reward(
        num_process=32, 
        step_frequence=STEP_FREQUENCE,
        max_simulate_time=MAX_SIMULATE_TIME,
        available_targets_dir=PROJECT_ROOT_DIR / "my_logs_for_parallel" / EXPERT_DATA_DIR,
        sample_target_noise_std=[0., 0., 0.],
    )
    
    algo_ppo = get_ppo_algo(vec_env)
    sb3_logger.info(str(algo_ppo.policy))
    # print(algo_ppo.policy)

    rng = np.random.default_rng(SEED)

    # train_transitions, validation_transitions, test_transitions = load_data(
    #     data_dir=EXPERT_DATA_DIR, 
    #     my_logger=sb3_logger,
    #     train_size=0.9,
    #     validation_size=0.05,
    #     test_size=0.05
    # )  # 10hz的数据划分结果，train: 745643, validation: 41425, test: 41425

    if (not args.train_on_difficulty_task_set) and (not args.train_on_chi_set):

        sb3_logger.info("load data from: " + str(PROJECT_ROOT_DIR / "runF16" / "algorithm" / "data_sb3" / EXPERT_DATA_CACHE_DIR / EXPERT_DATA_DIR))

        train_transitions, validation_transitions, test_transitions = load_data_from_cache(
            PROJECT_ROOT_DIR / "runF16" / "algorithm" / "data_sb3" / EXPERT_DATA_CACHE_DIR / EXPERT_DATA_DIR,
            train_size=0.96,
            validation_size=0.02,
            test_size=0.02,
            shuffle=True,
        )
    else:
        if args.train_on_difficulty_task_set:
            train_transitions, validation_transitions, test_transitions = load_data_by_difficulty(
                res_file_dir=PROJECT_ROOT_DIR / "runF16" / "algorithm" / "data_sb3" / "label_difficulty",
                res_file_name="res_with_difficulty.csv",
                data_dir=PROJECT_ROOT_DIR / "my_logs_for_parallel" / "10hz_10_5_5",
                difficulty=args.difficulty,
                task_set_capacity=args.task_set_capacity,
                train_size=0.96,
                validation_size=0.02,
                test_size=0.02,
                shuffle=True
            )
        if args.train_on_chi_set:
            train_transitions, validation_transitions, test_transitions = load_data_by_chi(
                res_file_dir=PROJECT_ROOT_DIR / "runF16" / "algorithm" / "data_sb3" / "label_difficulty",
                res_file_name="res_with_difficulty.csv",
                data_dir=PROJECT_ROOT_DIR / "my_logs_for_parallel" / "10hz_10_5_5",
                target_chi=args.target_chi,
                train_size=0.96,
                validation_size=0.02,
                test_size=0.02,
                shuffle=True
            )

    sb3_logger.info(f"train_set: obs size, {train_transitions.obs.shape}, act size, {train_transitions.acts.shape}")
    sb3_logger.info(f"validation_set: obs size, {validation_transitions.obs.shape}, act size, {validation_transitions.acts.shape}")
    sb3_logger.info(f"test_set: obs size, {test_transitions.obs.shape}, act size, {test_transitions.acts.shape}")

    bc_trainer = bc.BC(
        observation_space=vec_env.observation_space,
        action_space=vec_env.action_space,
        policy=algo_ppo.policy,
        batch_size=BC_BATCH_SIZE,
        ent_weight=BC_ENT_WEIGHT,
        l2_weight=BC_L2_WEIGHT,
        demonstrations=train_transitions,
        rng=rng,
        custom_logger=HierarchicalLogger(sb3_logger)
    )

    # train
    bc_trainer.train(
        n_epochs=TRAIN_EPOCHS,
        # on_batch_end=on_best_act_prob_save(algo_ppo, validation_transitions, sb3_logger),
        on_batch_end=on_best_loss_save(algo_ppo, validation_transitions, bc_trainer.loss_calculator, sb3_logger) if USE_LOSS_CALLBACK else on_best_act_prob_save(algo_ppo, validation_transitions, sb3_logger),
    )

    # evaluate with environment
    reward, _, success_rate = evaluate_policy_with_success_rate(algo_ppo.policy, vec_env, n_eval_episodes=1000)
    sb3_logger.info("Reward after BC: ", reward)
    sb3_logger.info("Success rate: ", success_rate)

    # 最终的policy在测试集上的prob_true_act / loss
    if USE_LOSS_CALLBACK:
        test_on_loss(algo_ppo.policy, test_transitions, bc_trainer.loss_calculator, sb3_logger, "训练结束时的策略", "测试集")
    else:
        test_on_prob_true_act(algo_ppo.policy, test_transitions, sb3_logger, "训练结束时的策略", "测试集")

    return sb3_logger, validation_transitions, test_transitions, bc_trainer


def test_on_prob_true_act(
        policy: MlpPolicy, 
        test_transitions: TransitionsMinimal, 
        sb3_logger: Logger, 
        policy_descreption: str, dataset_descreption: str
    ):
    policy.set_training_mode(mode=False)

    obs = util.safe_to_tensor(test_transitions.obs).to("cuda")
    acts = util.safe_to_tensor(test_transitions.acts).to("cuda")
    _, log_prob, entropy = policy.evaluate_actions(obs, acts)
    prob_true_act = th.exp(log_prob).mean()
    sb3_logger.info(f"{policy_descreption}在{dataset_descreption}上的prob_true_act: {prob_true_act}.")


def test_on_loss(
        policy: MlpPolicy, 
        test_transitions: TransitionsMinimal, 
        loss_calculator: bc.BehaviorCloningLossCalculator, 
        sb3_logger: Logger, 
        policy_descreption: str, dataset_descreption: str
    ):
    policy.set_training_mode(mode=False)

    obs = util.safe_to_tensor(test_transitions.obs).to("cuda")
    acts = util.safe_to_tensor(test_transitions.acts).to("cuda")
    
    metrics: bc.BCTrainingMetrics = loss_calculator(policy=policy, obs=obs, acts=acts)
    sb3_logger.info(f"{policy_descreption}在{dataset_descreption}上的loss: {metrics.loss}.")


if __name__ == "__main__":
    sb3_logger, validation_transitions, test_transitions, bc_trainer = train()

    # sb3_logger: Logger = configure(folder=str((PROJECT_TOOR_DIR / "my_rl_logs" / "bc" / EXPERIMENT_NAME).absolute()), format_strings=['stdout', 'log', 'csv'])
    # train_transitions, validation_transitions, test_transitions = load_data(
    #     data_dir=EXPERT_DATA_DIR, 
    #     my_logger=sb3_logger,
    #     train_size=0.9,
    #     validation_size=0.05,
    #     test_size=0.05
    # )

    policy_save_dir = Path(__file__).parent / "checkpoints_sb3" / "bc" / EXPERIMENT_NAME
    algo_ppo = PPOWithBCLoss.load(str((policy_save_dir / POLICY_FILE_SAVE_NAME).absolute()))

    if USE_LOSS_CALLBACK:
        test_on_loss(algo_ppo.policy, validation_transitions, bc_trainer.loss_calculator, sb3_logger, "最优策略", "验证集")
        test_on_loss(algo_ppo.policy, test_transitions, bc_trainer.loss_calculator, sb3_logger, "最优策略", "测试集")
    else:
        test_on_prob_true_act(algo_ppo.policy, validation_transitions, sb3_logger, "最优策略", "验证集")
        test_on_prob_true_act(algo_ppo.policy, test_transitions, sb3_logger, "最优策略", "测试集")

  

    # model = PPO("MlpPolicy", vec_env, verbose=1)
    # model.learn(total_timesteps=25_000)

    # obs = vec_env.reset()
    # for _ in range(1000):
    #     action, _states = model.predict(obs)
    #     obs, rewards, dones, info = vec_env.step(action)
    #     vec_env.render()