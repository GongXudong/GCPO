import gymnasium as gym
import numpy as np
from pathlib import Path
import logging
from time import time
from copy import deepcopy
import argparse
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecCheckNan, VecEnv, VecMonitor, is_vecenv_wrapped
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure, Logger
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common import type_aliases

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from runF16.env.guidenv_ponential_angle import GuideEnvPonentialAngle
from runF16.env.utils.my_wrappers import ScaledActionWrapper, ScaledObservationWrapper
from runF16.env.utils.my_vec_frame_stack_env import MyVecFrameStack
from runF16.env.utils.sb3_utils import get_vec_env, get_vec_env_ponential_angle_and_velocity, get_vec_env_dense_reward, get_vec_env_dense_reward_small_divide_by_chi, get_vec_env_dense_reward_small_divide_by_difficulty, get_vec_env_dense_reward_filter_by_difficulty
from runF16.algorithm.models.sb3_model import PPOWithBCLoss
from runF16.algorithm.utils.sb3_evaluate_policy import evaluate_policy_with_success_rate
from runF16.utils_my.my_log import get_logger
from runF16.configs.load_config import load_config

parser = argparse.ArgumentParser(description="传入配置文件")
parser.add_argument("--config_file_name", type=str, help="配置文件名", default="framestack_configs/seed1/config_10hz_loss_256_256.json")
parser.add_argument("--reward_type", type=str, help="奖励类型(dense或者ponential)", default="dense")
parser.add_argument("--env_type", type=str, help="奖励类型(origin, small_chi, small_difficulty)", default="origin")
# env_type为origin时，设置该值:
# -3表示所有专家不能完成的任务，
# -2表示所有可枚举的任务（包括专家能完成和专家不能完成的），
# -1表示所有专家能完成的任务
# 1-5表示相应等级难度的任务
parser.add_argument("--task_difficulty", type=int, default=-1, help="测试使用的任务难度")
parser.add_argument("--target_chi", type=int, default=0, help="env_type为small时，使用的target_chi")  # env_type为small_chi时，设置该值
parser.add_argument("--difficulty", type=int, default=0, help="训练要使用的任务难度")  # env_type为small_difficulty时，设置该值
parser.add_argument("--task_set_capacity", type=int, default=100, help="训练要使用的任务数")  # env_type为small_difficulty时，设置该值
parser.add_argument("--algo", type=str, help="输出要测试的算法，rl或bc或rl_bc", default="rl_bc")
parser.add_argument("--add_noise", action="store_true", help="是否在目标速度矢量上增加噪声")
args = parser.parse_args()

custom_config = load_config(args.config_file_name)

RL_EXPERIMENT_NAME = custom_config["rl_bc"]["experiment_name"]
RL_RL_EXPERIMENT_NAME = custom_config["rl_rl"]["experiment_name"] if "rl_rl" in custom_config else "default"
RL_SINGLE_EXPERIMENT_NAME = custom_config["rl"]["experiment_name"]
BC_EXPERIMENT_NAME = custom_config["bc"]["experiment_name"]
SEED = custom_config["rl_bc"]["seed"]
SEED_FOR_LOAD_ALGO = custom_config["rl_bc"]["seed_for_load_algo"]
STEP_FREQUENCE = custom_config["step_frequence"]
BC_POLICY_FILE_NAME = custom_config["bc"]["policy_file_save_name"]
EXPERT_DATA_DIR = custom_config["rollout"]["expert_data_dir"]
NET_ARCH = custom_config["rl_bc"]["net_arch"]
PPO_BATCH_SIZE = custom_config["rl_bc"]["batch_size"]
GAMMA = custom_config["rl_bc"]["gamma"]
ACTIVATE_VALUE_HEAD_TRAIN_STEPS = custom_config["rl_bc"]["activate_value_head_train_steps"]
RL_TRAIN_STEPS = custom_config["rl_bc"]["train_steps"]
ROLLOUT_PROCESS_NUM = custom_config["rl_bc"]["rollout_process_num"]
KL_WITH_BC_MODEL_COEF = custom_config["rl_bc"]["kl_with_bc_model_coef"]

MAX_SIMULATE_TIME = custom_config["env"]['max_simulate_time']


def my_evaluate_policy(policy_save_dir: Path, policy_save_name: str):
    
    env_config_dict_in_eval = {
        "num_process": ROLLOUT_PROCESS_NUM, 
        "logger": None, 
        "step_frequence": STEP_FREQUENCE,
        "max_simulate_time": MAX_SIMULATE_TIME,
        "gamma": GAMMA,
        "reward_scale_factor_b": 1.,  # GuideEnvDenseReward中使用，取值小于1效果相当于在数值上放大了奖励，大于1相当于缩小了奖励
        "available_targets_dir": PROJECT_ROOT_DIR / "my_logs_for_parallel" / EXPERT_DATA_DIR,
        # "sample_target_noise_std": [0., 0., 0.],
        "sample_target_noise_std": [5, 2.5, 2.5] if args.add_noise else [0., 0., 0.],
        "sample_reachable_target": False
    }
    
    if args.reward_type == "dense":
        if args.env_type == "origin":
            get_env_class = get_vec_env_dense_reward_filter_by_difficulty
            env_config_dict_in_eval.update({
                "difficulty": args.task_difficulty
            })
        elif args.env_type == "small_chi":
            get_env_class = get_vec_env_dense_reward_small_divide_by_chi
            env_config_dict_in_eval.update({
                "target_chi": args.target_chi
            })
        elif args.env_type == "small_difficulty":
            get_env_class = get_vec_env_dense_reward_small_divide_by_difficulty
            env_config_dict_in_eval.update({
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

    # evaluate_policy使用的测试环境
    eval_env = VecCheckNan(get_env_class(
        **env_config_dict_in_eval
    ))

    print("Load policy from: ", str((policy_save_dir / policy_save_name).absolute()))

    algo_ppo = PPOWithBCLoss.load(
        str((policy_save_dir / policy_save_name).absolute()),  
        seed=SEED_FOR_LOAD_ALGO,
        kl_coef_with_bc=KL_WITH_BC_MODEL_COEF,
        custom_objects={
            "bc_trained_algo": None,
            "action_space": eval_env.action_space
        }
    )

    # evaluate
    reward, _, success_rate = evaluate_policy_with_success_rate(algo_ppo.policy, eval_env, 1000)
    print(f"Reward: {reward}, success rate: {success_rate}")


if __name__ == "__main__":
    if args.algo == 'bc':
        print("测试bc")
        my_evaluate_policy(PROJECT_ROOT_DIR / "runF16" / "algorithm" / "checkpoints_sb3" / "bc" / BC_EXPERIMENT_NAME, BC_POLICY_FILE_NAME)
    elif args.algo == 'rl':
        print("测试只用rl训练")
        my_evaluate_policy(PROJECT_ROOT_DIR / "runF16" / "algorithm" / "checkpoints_sb3" / "rl_single" / RL_SINGLE_EXPERIMENT_NAME, "best_model")
    elif args.algo == 'rl_bc':
        print("测试bc后继续rl优化")
        my_evaluate_policy(PROJECT_ROOT_DIR / "runF16" / "algorithm" / "checkpoints_sb3" / "rl" / RL_EXPERIMENT_NAME, "best_model")
    elif args.algo == "rl_rl":
        print("测试继续rl优化(以上一轮RL策略作为出发点)")
        my_evaluate_policy(PROJECT_ROOT_DIR / "runF16" / "algorithm" / "checkpoints_sb3" / "rl_rl" / RL_RL_EXPERIMENT_NAME, "best_model")
    else:
        print(f"脚本参数--algo只能是rl, bc, rl_bc, 或者rl_rl")