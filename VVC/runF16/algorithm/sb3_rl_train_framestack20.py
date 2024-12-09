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
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecCheckNan

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from runF16.env.utils.sb3_utils import get_vec_env, get_vec_env_dense_reward, get_vec_env_dense_reward_small_divide_by_chi, get_vec_env_dense_reward_small_divide_by_difficulty, get_vec_env_ponential_angle_and_velocity
from runF16.configs.load_config import load_config
from runF16.env.utils.my_vec_frame_stack_env import MyVecFrameStack
from runF16.algorithm.utils.sb3_schedule import linear_schedule
from runF16.algorithm.utils.sb3_evaluate_policy import evaluate_policy_with_success_rate

parser = argparse.ArgumentParser(description="传入配置文件")
parser.add_argument("--config_file_name", type=str, help="配置文件名", default="framestack_configs/seed1/config_10hz_loss_128_128.json")
parser.add_argument("--reward_type", type=str, help="奖励类型(dense或者ponential)", default="dense")
parser.add_argument("--env_type", type=str, help="奖励类型(origin, small_chi, small_difficulty)", default="origin")
parser.add_argument("--target_chi", type=int, default=0, help="env_type为small时，使用的target_chi")
parser.add_argument("--difficulty", type=int, default=0, help="训练要使用的任务难度")
parser.add_argument("--task_set_capacity", type=int, default=100, help="训练要使用的任务数")
args = parser.parse_args()

custom_config = load_config(args.config_file_name)

STEP_FREQUENCE = custom_config["step_frequence"]

MAX_SIMULATE_TIME = custom_config["env"]['max_simulate_time']
REWARD_SCALAR_B = custom_config["env"].get("reward_scalar_b", 0.5)
SAMPLE_REACHABLE_TARGETS = custom_config["env"].get("sample_reachable_target", True)
N_STACK = custom_config["env"]['n_stack']
OBS_CNT = custom_config["env"]['obs_cnt']
TARGET_CNT = custom_config["env"]['target_cnt']

EXPERT_DATA_DIR = custom_config["rollout"]["expert_data_dir"]

RL_EXPERIMENT_NAME = custom_config["rl"]["experiment_name"]
SEED = custom_config["rl"]["seed"]

GAMMA = custom_config["rl_bc"]["gamma"]
NET_ARCH = custom_config["rl_bc"]["net_arch"]
PPO_BATCH_SIZE = custom_config["rl_bc"]["batch_size"]

RL_TRAIN_STEPS = custom_config["rl_bc"]["train_steps"]
ROLLOUT_PROCESS_NUM = custom_config["rl"]["rollout_process_num"]



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

    return PPO(
        policy=MlpPolicy, 
        env=env, 
        seed=SEED,
        batch_size=PPO_BATCH_SIZE,
        gamma=GAMMA,
        n_steps=2048,  # 采样时每个环境采样的step数
        n_epochs=5,  # 采样的数据在训练中重复使用的次数
        policy_kwargs=policy_kwargs,
        use_sde=True,  # 使用state dependant exploration,
        normalize_advantage=True,
        learning_rate=linear_schedule(3e-4),
    )

def train():

    sb3_logger: Logger = configure(folder=str((PROJECT_ROOT_DIR / "my_rl_logs" / "rl_single_framestack20" / RL_EXPERIMENT_NAME).absolute()), format_strings=['stdout', 'log', 'csv', 'tensorboard'])

    env_config_dict_in_training = {
        "num_process": ROLLOUT_PROCESS_NUM, 
        "logger": None, 
        "step_frequence": STEP_FREQUENCE,
        "max_simulate_time": MAX_SIMULATE_TIME,
        "gamma": GAMMA,
        "reward_scale_factor_b": 0.5,  # GuideEnvDenseReward中使用，取值小于1效果相当于在数值上放大了奖励，大于1相当于缩小了奖励
        "available_targets_dir": PROJECT_ROOT_DIR / "my_logs_for_parallel" / EXPERT_DATA_DIR,
        "sample_target_noise_std": [0., 0., 0.],
    }

    if args.reward_type == "dense":
        if args.env_type == "origin":
            get_env_class = get_vec_env_dense_reward
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

    env_num_used_in_eval = 32
    env_config_dict_in_eval = env_config_dict_in_training.copy()
    env_config_dict_in_eval.update({"num_process": env_num_used_in_eval})

    env_num_used_in_callback = 32
    env_config_dict_in_callback = env_config_dict_in_training.copy()
    env_config_dict_in_callback.update({"num_process": env_num_used_in_callback})

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

    # 使用FrameStack的话，需要加上下面这一行；不需要的话，注释掉
    vec_env = MyVecFrameStack(
        venv=vec_env,
        n_stack=N_STACK,
        obs_cnt=OBS_CNT,
        target_cnt=TARGET_CNT
    )
    eval_env = MyVecFrameStack(
        venv=eval_env,
        n_stack=N_STACK,
        obs_cnt=OBS_CNT,
        target_cnt=TARGET_CNT
    )
    eval_env_in_callback = MyVecFrameStack(
        venv=eval_env_in_callback,
        n_stack=N_STACK,
        obs_cnt=OBS_CNT,
        target_cnt=TARGET_CNT
    )

    algo_ppo = get_ppo_algo(vec_env)
    sb3_logger.info(str(algo_ppo.policy))

    # set sb3 logger
    algo_ppo.set_logger(sb3_logger)

    eval_callback = EvalCallback(
        eval_env_in_callback, 
        best_model_save_path=str((Path(__file__).parent / "checkpoints_sb3" / "rl_single_framestack20" / RL_EXPERIMENT_NAME).absolute()),
        log_path=str((PROJECT_ROOT_DIR / "my_rl_logs" / "rl_single_framestack20" / RL_EXPERIMENT_NAME).absolute()), 
        eval_freq=2048,
        n_eval_episodes=3*env_num_used_in_callback,
        deterministic=True, 
        render=False,
    )

    algo_ppo.learn(total_timesteps=RL_TRAIN_STEPS, callback=eval_callback)

    # evaluate
    reward, _, success_rate = evaluate_policy_with_success_rate(algo_ppo.policy, eval_env, 30*env_num_used_in_eval)
    sb3_logger.info(f"Reward after RL: {reward}")
    sb3_logger.info(f"Success rate after RL: {success_rate}")


if __name__ == "__main__":
    train()