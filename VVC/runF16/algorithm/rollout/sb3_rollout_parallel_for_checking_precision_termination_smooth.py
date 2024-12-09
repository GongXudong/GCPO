from pathlib import Path
import sys
import argparse
from copy import deepcopy
import pandas as pd
import numpy as np
from tqdm import tqdm
from ray.util.multiprocessing import Pool

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
from runF16.utils_my.smoothness.smoothness_measure import smoothness_measure_by_delta


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

def rollout(
    policy_dir_str: str, 
    expert_data_dir: str,
    target_goals_v: list,
    target_goals_mu: list,
    target_goals_chi: list,
    error_value_list: list,
    difficulty_calc_by_error_list: list,
    expert_length_list: list,
    debug: bool=False, 
    model_save_name: str="best_model"
):
    policy_dir = Path(policy_dir_str)
    target_goals = pd.DataFrame({
        "v": target_goals_v,
        "mu": target_goals_mu,
        "chi": target_goals_chi,
        "error_value": error_value_list,
        "difficulty_calc_by_error": difficulty_calc_by_error_list,
        "length": expert_length_list,
    })

    env_config = {
        "num_process": 1, 
        "logger": None, 
        "step_frequence": STEP_FREQUENCE,
        "max_simulate_time": MAX_SIMULATE_TIME,
        "gamma": GAMMA,
        "reward_scale_factor_b": 0.5,  # GuideEnvDenseReward中使用，取值小于1效果相当于在数值上放大了奖励，大于1相当于缩小了奖励
        "available_targets_dir": PROJECT_ROOT_DIR / "my_logs_for_parallel" / expert_data_dir,
        "available_targets_filename": "res.csv",
        "use_fixed_target": False, 
        "sample_target_noise_std": [0., 0., 0.],
        "difficulty": -2  # 从所有任务上采样
    }

    env_class = GuideEnvDenseRewardFilteredByDifficulty
    
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
        "smooth_a_p": [],
        "smooth_a_nz": [],
        "smooth_a_pla": [],
        "smooth_s_phi": [],
        "smooth_s_theta": [],
        "smooth_s_psi": [],
        "smooth_s_v": [],
        "smooth_s_mu": [],
        "smooth_s_chi": [],
        "smooth_s_p": [],
        "smooth_s_h": [],
    }

    # 枚举任务
    
    for index, target in tqdm(target_goals.iterrows(), total=target_goals.shape[0]):
        # 为环境设置任务
        # target_v, target_mu, target_chi, expert_length = target["v"], target["mu"], target["chi"], target["length"]
        helper_env.use_fixed_target = True
        tmp_noise = helper_env.sample_noise()
        helper_env.target_v = target["v"] + tmp_noise[0]
        helper_env.target_mu = target["mu"] + tmp_noise[1]
        helper_env.target_chi = target["chi"] + tmp_noise[2]

        traj = {
            "time": [],
            "s_phi": [],
            "s_theta": [],
            "s_psi": [],
            "s_v": [],
            "s_mu": [],
            "s_chi": [],
            "s_p": [],
            "s_h": [],
            "a_p": [],
            "a_nz": [],
            "a_pla": [],
            "a_rud": []
        }

        # 采样
        obs, info = scaled_act_env.reset()
        terminate = False
        s_index = 0
        while not terminate:
            action, _ = algo_ppo.predict(observation=obs, deterministic=True)
            tmp_obs = scaled_obs_env.inverse_scale_state(obs[-11:])
            obs, reward, terminate, truncated, info = scaled_act_env.step(action=action)
            
            traj["time"].append(s_index * 1. / STEP_FREQUENCE)
            traj["s_phi"].append(info["origin_obs"]["phi"])
            traj["s_theta"].append(info["origin_obs"]["theta"])
            traj["s_psi"].append(info["origin_obs"]["psi"])
            traj["s_v"].append(info["origin_obs"]["v"])
            traj["s_mu"].append(info["origin_obs"]["mu"])
            traj["s_chi"].append(info["origin_obs"]["chi"])
            traj["s_p"].append(info["origin_obs"]["p"])
            traj["s_h"].append(info["origin_obs"]["h"])
            traj["a_p"].append(info["action"]["p"])
            traj["a_nz"].append(info["action"]["nz"])
            traj["a_pla"].append(info["action"]["pla"])
            traj["a_rud"].append(info["action"]["rud"])
            
            s_index += 1

        traj_df = pd.DataFrame(data=traj, columns=["time", "s_phi", "s_theta", "s_psi", "s_v", "s_mu", "s_chi", "s_p", "s_h", "a_p", "a_nz", "a_pla", "a_rud"])

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
        res_dict["smooth_a_p"].append(smoothness_measure_by_delta(traj_df, measure_columns=["a_p"])[0])
        res_dict["smooth_a_nz"].append(smoothness_measure_by_delta(traj_df, measure_columns=["a_nz"])[0])
        res_dict["smooth_a_pla"].append(smoothness_measure_by_delta(traj_df, measure_columns=["a_pla"])[0])
        res_dict["smooth_s_phi"].append(smoothness_measure_by_delta(traj_df, measure_columns=["s_phi"])[0])
        res_dict["smooth_s_theta"].append(smoothness_measure_by_delta(traj_df, measure_columns=["s_theta"])[0])
        res_dict["smooth_s_psi"].append(smoothness_measure_by_delta(traj_df, measure_columns=["s_psi"])[0])
        res_dict["smooth_s_v"].append(smoothness_measure_by_delta(traj_df, measure_columns=["s_v"])[0])
        res_dict["smooth_s_mu"].append(smoothness_measure_by_delta(traj_df, measure_columns=["s_mu"])[0])
        res_dict["smooth_s_chi"].append(smoothness_measure_by_delta(traj_df, measure_columns=["s_chi"])[0])
        res_dict["smooth_s_p"].append(smoothness_measure_by_delta(traj_df, measure_columns=["s_p"])[0])
        res_dict["smooth_s_h"].append(smoothness_measure_by_delta(traj_df, measure_columns=["s_h"])[0])

    return res_dict

# python runF16/algorithm/rollout/sb3_rollout_parallel_for_checking_precision_termination_smooth.py --algo rl_bc --cur-expert-data-dir test --process-num 4 --config-file-name no_framestack_iter_2_configs/seed2/config_10hz_loss_128_128.json --save-file-name test.csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="传入配置文件")
    parser.add_argument("--config-file-name", type=str, help="配置文件名", default="no_framestack_configs/seed2/config_10hz_loss_128_128.json")
    parser.add_argument("--algo", type=str, help="", default="rl_bc")
    parser.add_argument("--save-file-name", type=str, help="", default="res_for_checking_precision_rl_bc_2.csv")
    parser.add_argument("--cur-expert-data-dir", type=str, help="当前专家数据目录", default="10hz_10_5_5_iter_2")
    parser.add_argument("--process-num", type=int, help="使用的采样进程数", default=10)
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

    if args.algo == 'bc':
        print("测试bc")
        policy_save_dir = PROJECT_ROOT_DIR / "runF16" / "algorithm" / "checkpoints_sb3" / "bc" / BC_EXPERIMENT_NAME
        model_name = "bc_checkpoint"
    elif args.algo == 'rl':
        print("测试只用rl训练")
        policy_save_dir = PROJECT_ROOT_DIR / "runF16" / "algorithm" / "checkpoints_sb3" / "rl_single" / RL_SINGLE_EXPERIMENT_NAME
        model_name = "best_model"
    elif args.algo == 'rl_bc':
        print("测试bc后继续rl优化")
        policy_save_dir = PROJECT_ROOT_DIR / "runF16" / "algorithm" / "checkpoints_sb3" / "rl" / RL_EXPERIMENT_NAME
        model_name = "best_model"
    elif args.algo == 'rl_rl':
        print("测试rl后继续rl优化")
        policy_save_dir = PROJECT_ROOT_DIR / "runF16" / "algorithm" / "checkpoints_sb3" / "rl_rl" / custom_config["rl_rl"]["experiment_name"]
        model_name = "best_model"
    else:
        print(f"脚本参数--algo只能是rl, bc, 或者rl_bc")
    
    with Pool(processes=args.process_num) as pool:
        res_df = pd.read_csv(PROJECT_ROOT_DIR / "my_logs_for_parallel" / args.cur_expert_data_dir / "res.csv")
        # 设置要分成的份数
        n = args.process_num
        # 计算每份的行数
        chunk_size = len(res_df) // n
        # 分割DataFrame
        chunks = [res_df.iloc[i*chunk_size:(i+1)*chunk_size] for i in range(n)]
        # 如果不能完全均分，处理剩余的数据
        if len(res_df) % n != 0:
            # 将剩余的数据分配到最后一个chunk
            last_chunk = res_df.iloc[n*chunk_size:]
            chunks[-1] = pd.concat([chunks[-1], last_chunk])

        res = pool.starmap(
            rollout,
            [[
                str(policy_save_dir),
                str(args.cur_expert_data_dir),
                list(target.v),
                list(target.mu),
                list(target.chi),
                list(target.error_value),
                list(target.difficulty_calc_by_error),
                list(target["length"]),
                False,
                model_name,
             ] for target in chunks]
        )

        res_df = pd.concat([pd.DataFrame(tmp) for tmp in res])
        res_df.to_csv(PROJECT_ROOT_DIR / "runF16" / "algorithm" / "rollout" / "cache" / args.save_file_name, index=False)
