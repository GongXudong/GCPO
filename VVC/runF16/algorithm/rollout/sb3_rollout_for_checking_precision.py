from pathlib import Path
import sys
import argparse
from copy import deepcopy
import pandas as pd
import numpy as np
from tqdm import tqdm

from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import VecCheckNan

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from runF16.algorithm.models.sb3_model import PPOWithBCLoss
from runF16.configs.load_config import load_config
from runF16.env.utils.sb3_utils import get_vec_env_ponential_angle, get_vec_env_ponential_angle_and_velocity, get_vec_env_dense_reward, get_vec_env_dense_reward_small_divide_by_chi, get_vec_env_dense_reward_small_divide_by_difficulty
from runF16.env.guidenv_ponential_angle import GuideEnvPonentialAngle
from runF16.env.guidenv_dense_reward import GuideEnvDenseReward
from runF16.env.guidenv_dense_reward_filter_by_difficulty import GuideEnvDenseRewardFilteredByDifficulty
from runF16.env.guidenv_dense_reward_small_divide_by_chi import GuideEnvDenseRewardSmallDivideByChi
from runF16.env.guidenv_dense_reward_small_divide_by_difficulty import GuideEnvDenseRewardSmallDivideByDifficulty
from runF16.env.utils.my_wrappers import ScaledActionWrapper, ScaledObservationWrapper

parser = argparse.ArgumentParser(description="传入配置文件")
parser.add_argument("--config_file_name", type=str, help="配置文件名", default="framestack_configs/seed1/config_10hz_loss_256_256.json")
parser.add_argument("--reward_type", type=str, help="奖励类型(dense或者ponential)", default="dense")
parser.add_argument("--algo", type=str, help="输出要测试的算法，rl或bc或rl_bc", default="rl_bc")
parser.add_argument("--env_type", type=str, help="环境类型(origin, small_chi, small_difficulty)", default="small_difficulty")
# env_type为origin时，设置该值:
# -3表示所有专家不能完成的任务，
# -2表示所有可枚举的任务（包括专家能完成和专家不能完成的），
# -1表示所有专家能完成的任务
# 1-5表示相应等级难度的任务
parser.add_argument("--task_difficulty", type=int, default=-1, help="测试使用的任务难度")
parser.add_argument("--target_chi", type=int, default=0, help="env_type为small时，使用的target_chi")
parser.add_argument("--difficulty", type=int, default=0, help="训练要使用的任务难度")
parser.add_argument("--task_set_capacity", type=int, default=100, help="训练要使用的任务数")
parser.add_argument("--save_file_name", type=str, default="res_for_checking_precision.csv", help="保存结果的文件名")

parser.add_argument("--target_v", type=int, default=200, help="目标速度")
parser.add_argument("--target_mu", type=int, default=20, help="目标mu")
args = parser.parse_args()

custom_config = load_config(args.config_file_name)

RL_EXPERIMENT_NAME = custom_config["rl_bc"]["experiment_name"]
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

MAX_SIMULATE_TIME = custom_config["env"]['max_simulate_time']


def get_ppo_algo(env):
    policy_kwargs = dict(
        net_arch=dict(
            pi=NET_ARCH,
            vf=deepcopy(NET_ARCH)
        )
    )

    return PPOWithBCLoss(
        policy=MlpPolicy, 
        env=env, 
        seed=SEED,
        batch_size=PPO_BATCH_SIZE,
        gamma=GAMMA,
        n_steps=2048,  # 采样时每个环境采样的step数
        n_epochs=5,  # 采样的数据在训练中重复使用的次数
        policy_kwargs=policy_kwargs
    )

def termination_shotcut(termination_str: str):
    if termination_str.find("reach_target") != -1:
        return "reach target"
    if termination_str.find("timeout") != -1:
        return "timeout"
    if termination_str.find("move_away") != -1:
        return "continuous move away"
    if termination_str.find("roll") != -1:
        return "continuous roll"
    if termination_str.find("crash") != -1:
        return "crash"
    if termination_str.find("extreme") != -1:
        return "extreme state"
    if termination_str.find("negative") != -1:
        return "negative overload"

def rollout(policy_dir: Path, debug: bool=False, model_save_name: str="best_model"):
    
    env_config = {
        "num_process": 1, 
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
            env_class = GuideEnvDenseRewardFilteredByDifficulty
            env_config.update({
                "difficulty": args.task_difficulty
            })
        elif args.env_type == "small_chi":
            env_class = GuideEnvDenseRewardSmallDivideByChi
            env_config.update({
                "use_fixed_target": False, 
                "target_chi": args.target_chi
            })
        elif args.env_type == "small_difficulty":
            env_class = GuideEnvDenseRewardSmallDivideByDifficulty
            env_config.update({
                "available_targets_dir": PROJECT_ROOT_DIR / "runF16" / "algorithm" / "data_sb3" / "label_difficulty",
                "available_targets_filename": "res_with_difficulty.csv",
                "use_fixed_target": False, 
                "difficulty": args.difficulty,
                "task_set_capacity": args.task_set_capacity
            })
        else:
            print("环境类型只能是origin, small_chi或者small_difficulty")
    else:
        print("奖励类型只能是dense或者ponential")
    
    helper_env = env_class(env_config=env_config)
    scaled_obs_env = ScaledObservationWrapper(helper_env)
    scaled_act_env = ScaledActionWrapper(scaled_obs_env)

    algo_ppo = PPOWithBCLoss.load(
        str((policy_dir / model_save_name).absolute()), 
        custom_objects={
            "observation_space": scaled_act_env.observation_space,
            "action_space": scaled_act_env.action_space
        }
    )
    algo_ppo.policy.set_training_mode(False)

    res_dict = {
        "v": [],
        "mu": [],
        "chi": [],
        "error_value": [],
        "difficulty_calc_by_error": [],
        "expert length": [],
        "length": [],
        "termination": [],
        "achieved v": [],
        "achieved mu": [],
        "achieved chi": [],
    }

    # 枚举任务
    
    for index, target in tqdm(helper_env.available_targets.iterrows(), total=helper_env.available_targets.shape[0]):
        # 为环境设置任务
        # target_v, target_mu, target_chi, expert_length = target["v"], target["mu"], target["chi"], target["length"]
        helper_env.use_fixed_target = True
        tmp_noise = helper_env.sample_noise()
        helper_env.target_v = target["v"] + tmp_noise[0]
        helper_env.target_mu = target["mu"] + tmp_noise[1]
        helper_env.target_chi = target["chi"] + tmp_noise[2]
        helper_env.target_expert_steps = target["length"]

        # 采样
        obs, info = scaled_act_env.reset()
        terminate = False
        s_index = 0
        while not terminate:
            action, _ = algo_ppo.predict(observation=obs, deterministic=True)
            tmp_obs = scaled_obs_env.inverse_scale_state(obs[-11:])
            obs, reward, terminate, truncated, info = scaled_act_env.step(action=action)
            s_index += 1

        res_dict["v"].append(target["v"])
        res_dict["mu"].append(target["mu"])
        res_dict["chi"].append(target["chi"])
        res_dict["error_value"].append(target["error_value"])
        res_dict["difficulty_calc_by_error"].append(target["difficulty_calc_by_error"])
        res_dict["expert length"].append(target["length"])
        res_dict["achieved v"].append(tmp_obs[3])
        res_dict["achieved mu"].append(tmp_obs[4])
        res_dict["achieved chi"].append(tmp_obs[5])
        res_dict["length"].append(s_index)
        res_dict["termination"].append(termination_shotcut(info["termination"]))

    res_df = pd.DataFrame(data=res_dict)
    res_df.to_csv(PROJECT_ROOT_DIR / "runF16" / "algorithm" / "rollout" / "cache" / args.save_file_name, index=False)


if __name__ == "__main__":
    if args.algo == 'bc':
        print("测试bc")
        bc_policy_save_dir = PROJECT_ROOT_DIR / "runF16" / "algorithm" / "checkpoints_sb3" / "bc" / BC_EXPERIMENT_NAME
        rollout(bc_policy_save_dir, model_save_name="bc_checkpoint")
    elif args.algo == 'rl':
        print("测试只用rl训练")
        rl_policy_save_dir = PROJECT_ROOT_DIR / "runF16" / "algorithm" / "checkpoints_sb3" / "rl_single" / RL_SINGLE_EXPERIMENT_NAME
        rollout(rl_policy_save_dir)
    elif args.algo == 'rl_bc':
        print("测试bc后继续rl优化")
        rl_policy_save_dir = PROJECT_ROOT_DIR / "runF16" / "algorithm" / "checkpoints_sb3" / "rl" / RL_EXPERIMENT_NAME
        rollout(rl_policy_save_dir)
    else:
        print(f"脚本参数--algo只能是rl, bc, 或者rl_bc")