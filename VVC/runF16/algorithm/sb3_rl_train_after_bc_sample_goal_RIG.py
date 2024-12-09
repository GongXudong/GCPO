import gymnasium as gym
import numpy as np
from pathlib import Path
import logging
import time
from copy import deepcopy
import argparse
import sys
import torch as th

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecCheckNan
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure, Logger
from stable_baselines3.common.callbacks import EvalCallback

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from runF16.env.guidenv import GuideEnv
from runF16.env.guidenv_dense_reward_sample_goal_from_dist_RIG import GuideEnvDenseRewardSampleGoalFromDistRIG
from runF16.env.utils.my_wrappers import ScaledActionWrapper, ScaledObservationWrapper
from runF16.algorithm.models.sb3_model import PPOWithBCLoss
from runF16.algorithm.utils.sb3_eval_callback import MyEvalCallback, MyEvalCallbackSTAT
from runF16.algorithm.utils.sb3_evaluate_policy import evaluate_policy_with_success_rate
from runF16.env.utils.sb3_utils import get_vec_env, get_vec_env_dense_reward, get_vec_env_dense_reward_small_divide_by_chi, get_vec_env_dense_reward_small_divide_by_difficulty, get_vec_env_ponential_angle_and_velocity, get_vec_env_dense_reward_sample_goal_from_dist_rig
from runF16.utils_my.my_log import get_logger
from runF16.configs.load_config import load_config
from runF16.algorithm.utils.sb3_schedule import linear_schedule

parser = argparse.ArgumentParser(description="传入配置文件")
parser.add_argument("--config_file_name", type=str, help="配置文件名", default="no_framestack_configs/seed1/config_10hz_loss_128_128_sample_goal.json")
parser.add_argument("--reward_type", type=str, help="奖励类型(dense或者ponential)", default="dense")
parser.add_argument("--env_type", type=str, help="奖励类型(origin, small_chi, small_difficulty)", default="origin")
parser.add_argument("--target_chi", type=int, default=0, help="env_type为small时，使用的target_chi")  # env_type为small_chi时，设置该值
parser.add_argument("--difficulty", type=int, default=0, help="训练要使用的任务难度")  # env_type为small_difficulty时，设置该值
parser.add_argument("--task_set_capacity", type=int, default=100, help="训练要使用的任务数")  # env_type为small_difficulty时，设置该值
args = parser.parse_args()

custom_config = load_config(args.config_file_name)


STEP_FREQUENCE = custom_config["step_frequence"]

MAX_SIMULATE_TIME = custom_config["env"]['max_simulate_time']
REWARD_SCALAR_B = custom_config["env"].get("reward_scalar_b", 0.5)
SAMPLE_REACHABLE_TARGETS = custom_config["env"].get("sample_reachable_target", True)
GOAL_DIST_ADD_COEF = custom_config["env"].get("goal_dist_add_coef", 0.5)

EXPERT_DATA_DIR = custom_config["rollout"]["expert_data_dir"]

BC_EXPERIMENT_NAME = custom_config["bc"]["experiment_name"]
BC_POLICY_FILE_NAME = custom_config["bc"]["policy_file_save_name"]
BC_POLICY_AFTER_VALUE_HEAD_TRAINED_FILE_NAME = custom_config["bc"]["policy_after_value_head_trained_file_save_name"]

RL_EXPERIMENT_NAME = custom_config["rl_bc"]["experiment_name"]
SEED = custom_config["rl_bc"]["seed"]
SEED_FOR_LOAD_ALGO = custom_config["rl_bc"]["seed_for_load_algo"]
NET_ARCH = custom_config["rl_bc"]["net_arch"]
PPO_BATCH_SIZE = custom_config["rl_bc"]["batch_size"]
GAMMA = custom_config["rl_bc"]["gamma"]
ACTIVATE_VALUE_HEAD_TRAIN_STEPS = custom_config["rl_bc"]["activate_value_head_train_steps"]
RL_TRAIN_STEPS = custom_config["rl_bc"]["train_steps"]
RL_ENT_COEF = custom_config["rl_bc"].get("ent_coef", 0.0)
RL_LR_RATE = custom_config["rl_bc"].get("lr", 3e-4)
ROLLOUT_PROCESS_NUM = custom_config["rl_bc"]["rollout_process_num"]
EVALUATE_PROCESS_NUM = custom_config["rl_bc"].get("evaluate_process_num", 32)
CALLBACK_PROCESS_NUM = custom_config["rl_bc"].get("callback_process_num", 32)
EVALUATE_ON_ALL_TASKS = custom_config["rl_bc"].get("evaluate_on_all_tasks", False)
EVALUATE_FREQUENCE = custom_config["rl_bc"].get("evaluate_frequence", 2048)
EVALUATE_NUMS_IN_EVALUATION = custom_config["rl_bc"].get("evaluate_nums_in_evaluation", 30)
EVALUATE_NUMS_IN_CALLBACK = custom_config["rl_bc"].get("evaluate_nums_in_callback", 3)
KL_WITH_BC_MODEL_COEF = custom_config["rl_bc"]["kl_with_bc_model_coef"]
KL_ANNEALING = custom_config["rl_bc"].get("kl_annealing", False)
GOAL_CNT_ANNEALING_COEF = custom_config["rl_bc"].get("goal_cnt_annealing_coef", 0.9)

np.seterr(all="raise")  # 检查nan

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
        seed=SEED,
        kl_coef_with_bc=linear_schedule(KL_WITH_BC_MODEL_COEF) if KL_ANNEALING else KL_WITH_BC_MODEL_COEF,
        batch_size=PPO_BATCH_SIZE,  # PPO Mini Batch Size, PPO每次更新使用的数据量
        gamma=GAMMA,
        ent_coef=RL_ENT_COEF,
        n_steps=2048,  # 采样时每个环境采样的step数，PPO每次训练收集的数据量是n_steps * num_envs
        n_epochs=5,  # 采样的数据在训练中重复使用的次数
        policy_kwargs=policy_kwargs,
        use_sde=True,  # 使用state dependant exploration,
        normalize_advantage=True,
        learning_rate=linear_schedule(RL_LR_RATE),
    )


def train():
    
    sb3_logger: Logger = configure(folder=str((PROJECT_ROOT_DIR / "my_rl_logs" / "rl" / RL_EXPERIMENT_NAME).absolute()), format_strings=['stdout', 'log', 'csv', 'tensorboard'])

    env_config_dict_in_training = {
        "num_process": ROLLOUT_PROCESS_NUM, 
        "logger": None, 
        "step_frequence": STEP_FREQUENCE,
        "max_simulate_time": MAX_SIMULATE_TIME,
        "gamma": GAMMA,
        "reward_scale_factor_b": REWARD_SCALAR_B,  # GuideEnvDenseReward中使用，取值小于1效果相当于在数值上放大了奖励，大于1相当于缩小了奖励
        "available_targets_dir": PROJECT_ROOT_DIR / "my_logs_for_parallel" / EXPERT_DATA_DIR,
        "sample_target_noise_std": [0., 0., 0.],
        "flag_str": "Train"
    }

    # 创建临时env，只为了获取success_goal_cnt_dict和fail_goal_cnt_dict
    # 这两个字典用于训练环境采样goal，在训练过程中的测试中，更新这两个字典中的统计信息
    tmp_env = GuideEnvDenseRewardSampleGoalFromDistRIG(
        env_config=env_config_dict_in_training,
    )
    goal_cnt_dict = tmp_env.goal_cnt_dict

    # 训练中根据goal的分布采样（成功率低的）goal
    env_config_dict_in_training.update({
        "sample_from_dist": True,
        "goal_cnt_dict": goal_cnt_dict,
    })

    if args.reward_type == "dense":
        if args.env_type == "origin":
            get_env_class = get_vec_env_dense_reward_sample_goal_from_dist_rig
        elif args.env_type == "small_chi":
            get_env_class = get_vec_env_dense_reward_small_divide_by_chi
            env_config_dict_in_training.update({
                "target_chi": args.target_chi
            })
        elif args.env_type == "small_difficulty":
            get_env_class = get_vec_env_dense_reward_small_divide_by_difficulty
            env_config_dict_in_training.update({
                "available_targets_dir": PROJECT_ROOT_DIR / "runF16" / "algorithm" / "data_sb3" / "label_difficulty",
                "available_targets_filename": "res_with_difficulty.csv",
                "use_fixed_target": False, 
                "difficulty": args.difficulty,
                "task_set_capacity": args.task_set_capacity
            })
        else:
            print("环境类型只能是origin, small_chi或者small_difficulty")
    elif args.reward_type == "ponential":
        get_env_class = get_vec_env_ponential_angle_and_velocity
    else:
        print("奖励类型只能是dense或者ponential")
    
    env_num_used_in_eval = EVALUATE_PROCESS_NUM
    env_config_dict_in_eval = env_config_dict_in_training.copy()
    env_config_dict_in_eval.update({
        "sample_reachable_target": False if EVALUATE_ON_ALL_TASKS else True,
        "sample_from_dist": False,
        "debug": False,
        "flag_str": "Evaluate",
        "num_process": env_num_used_in_eval
    })

    env_num_used_in_callback = CALLBACK_PROCESS_NUM
    env_config_dict_in_callback = env_config_dict_in_training.copy()
    # 测试中，均匀采样goal，更新目标成功与失败的次数
    env_config_dict_in_callback.update({
        "sample_reachable_target": False if EVALUATE_ON_ALL_TASKS else True,
        "sample_from_dist": False,
        "debug": False,
        "flag_str": "Callback",
        "num_process": env_num_used_in_callback
    })

    get_env_class = get_vec_env_dense_reward_sample_goal_from_dist_rig
    # 训练使用的环境
    vec_env = VecCheckNan(get_env_class(
        **env_config_dict_in_training
    ))
    # evaluate_policy使用的测试环境
    eval_env = VecCheckNan(get_env_class(
        **env_config_dict_in_eval
    ))
    # 回调函数中使用的测试环境
    eval_env_in_callback = VecCheckNan(get_env_class(
        **env_config_dict_in_callback
    ))

    # load model
    bc_policy_save_dir = Path(__file__).parent / "checkpoints_sb3" / "bc" / BC_EXPERIMENT_NAME
    algo_ppo_for_kl_loss = PPOWithBCLoss.load(str((bc_policy_save_dir / BC_POLICY_FILE_NAME).absolute()))
    algo_ppo_for_kl_loss.policy.set_training_mode(False)
    algo_ppo = PPOWithBCLoss.load(
        str((bc_policy_save_dir / BC_POLICY_FILE_NAME).absolute()), 
        env=vec_env, 
        seed=SEED_FOR_LOAD_ALGO,
        custom_objects={
            "bc_trained_algo": algo_ppo_for_kl_loss,
            "learning_rate": linear_schedule(RL_LR_RATE),
        },
        kl_coef_with_bc=KL_WITH_BC_MODEL_COEF,
    )
    sb3_logger.info(str(algo_ppo.policy))

    # set sb3 logger
    algo_ppo.set_logger(sb3_logger)

    # evaluate
    reward, _, success_rate = evaluate_policy_with_success_rate(
        algo_ppo.policy, 
        eval_env, 
        EVALUATE_NUMS_IN_EVALUATION * env_num_used_in_eval
    )
    sb3_logger.info(f"Reward before RL: {reward}")
    sb3_logger.info(f"Success rate before RL: {success_rate}")

    # train value head
    # print("#########")
    # print(algo_ppo.policy)
    # print("#########")
    for k, v in algo_ppo.policy.named_parameters():
        # print(k)
        if any([ x in k.split('.') for x in ['shared_net', 'policy_net', 'action_net']]):  # 还有一个log_std
            v.requires_grad = False
    # exit(0)
    # for k, v in algo_ppo.policy.named_parameters():
    #     print(k, v.requires_grad)
    
    start_time = time.time()
    algo_ppo.learn(total_timesteps=ACTIVATE_VALUE_HEAD_TRAIN_STEPS)
    sb3_logger.info(f"training value head time: {time.time() - start_time}(s).")

    # save model, value head训练完的
    algo_ppo.save(bc_policy_save_dir / BC_POLICY_AFTER_VALUE_HEAD_TRAINED_FILE_NAME)

    # evaluate
    reward, _, success_rate = evaluate_policy_with_success_rate(
        algo_ppo.policy, 
        eval_env, 
        EVALUATE_NUMS_IN_EVALUATION * env_num_used_in_eval
    )
    sb3_logger.info(f"Reward after training value head: {reward}")
    sb3_logger.info(f"Success rate after training value head: {success_rate}")

    # continue training
    for k, v in algo_ppo.policy.named_parameters():
        if any([ x in k.split('.') for x in ['shared_net', 'policy_net', 'action_net']]):
            v.requires_grad = True

    # for k, v in algo_ppo.policy.named_parameters():
    #     print(k, v.requires_grad)

    # sb3自带的EvalCallback根据最高平均reward保存最优策略；改成MyEvalCallback，根据最高胜率保存最优策略
    eval_callback = MyEvalCallbackSTAT(
        eval_env_in_callback, 
        best_model_save_path=str((Path(__file__).parent / "checkpoints_sb3" / "rl" / RL_EXPERIMENT_NAME).absolute()),
        log_path=str((PROJECT_ROOT_DIR / "my_rl_logs" / "rl" / RL_EXPERIMENT_NAME).absolute()), 
        eval_freq=EVALUATE_FREQUENCE,  # 多少次env.step()评估一次，此处设置为1000，因为VecEnv有72个并行环境，所以实际相当于72*1000次step，评估一次
        n_eval_episodes=EVALUATE_NUMS_IN_CALLBACK * env_num_used_in_callback,  # 每次评估使用多少条轨迹
        deterministic=True, 
        render=False,
        sync_stat_success_dict=True,
        sync_success_dict_env_method_name="set_goal_cnt_dict",
        stat_success_dict=goal_cnt_dict,
        stat_success_annealing_coef=GOAL_CNT_ANNEALING_COEF,
        sync_stat_fail_dict=False,
        training_envs=vec_env
    )

    algo_ppo.learn(total_timesteps=RL_TRAIN_STEPS, callback=eval_callback)

    # evaluate
    reward, _, success_rate = evaluate_policy_with_success_rate(
        algo_ppo.policy, 
        eval_env, 
        EVALUATE_NUMS_IN_EVALUATION * env_num_used_in_eval
    )
    sb3_logger.info(f"Reward after RL: {reward}")
    sb3_logger.info(f"Success rate after RL: {success_rate}")

    # save model
    # rl_policy_save_dir = Path(__file__).parent / "checkpoints_sb3" / "rl" / RL_EXPERIMENT_NAME
    # algo_ppo.save(str(rl_policy_save_dir / RL_POLICY_FILE_NAME))


if __name__ == "__main__":
    train()
