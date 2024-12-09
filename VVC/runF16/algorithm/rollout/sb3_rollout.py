from pathlib import Path
import sys
import argparse
from copy import deepcopy
import pandas as pd
import numpy as np

from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import VecCheckNan

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from runF16.algorithm.models.sb3_model import PPOWithBCLoss
from runF16.configs.load_config import load_config
from runF16.env.utils.sb3_utils import get_vec_env_ponential_angle, get_vec_env_ponential_angle_and_velocity, get_vec_env_dense_reward
from runF16.env.guidenv_ponential_angle import GuideEnvPonentialAngle
from runF16.env.utils.my_wrappers import ScaledActionWrapper, ScaledObservationWrapper

parser = argparse.ArgumentParser(description="传入配置文件")
parser.add_argument("--config_file_name", type=str, help="配置文件名", default="framestack_configs/seed1/config_10hz_loss_256_256.json")
parser.add_argument("--reward_type", type=str, help="奖励类型(dense或者ponential)", default="dense")
parser.add_argument("--algo", type=str, help="输出要测试的算法，rl或bc或rl_bc", default="rl_bc")
parser.add_argument("--save_acmi", action="store_true", help="是否将轨迹存储成acmi文件")
parser.add_argument("--use_fixed_target", action="store_true", help="是否使用固定目标")
parser.add_argument("--target_v", type=int, default=200, help="目标速度")
parser.add_argument("--target_mu", type=int, default=20, help="目标mu")
parser.add_argument("--target_chi", type=int, default=30, help="目标chi")
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


def rollout(policy_dir: Path, debug: bool=False, model_save_name: str="best_model"):
    if args.reward_type == "dense":
        get_env_class = get_vec_env_dense_reward
    elif args.reward_type == "ponential":
        get_env_class = get_vec_env_ponential_angle_and_velocity
    else:
        print("奖励类型只能是dense或者ponential")
     
    vec_env = get_env_class(
        num_process=1, 
        logger=None, 
        step_frequence=STEP_FREQUENCE,
        max_simulate_time=MAX_SIMULATE_TIME,
        gamma=GAMMA,
        reward_scale_factor_b=0.5,
        available_targets_dir=PROJECT_ROOT_DIR / "my_logs_for_parallel" / EXPERT_DATA_DIR,
        sample_target_noise_std=[0., 0., 0.],
        use_fixed_target=args.use_fixed_target,
        target_v=args.target_v,
        target_mu=args.target_mu,
        target_chi=args.target_chi
    )
    
    helper_env = GuideEnvPonentialAngle(env_config={})
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

    obs_list, act_list, reward_list, ve_vn_vh_list, timestamps = [], [], [], [], []

    tackview_data = {
        'ISO time': [],
        'id': [],
        'Longitude': [],
        'Latitude': [],
        'Altitude': [],
        'Roll': [],
        'Pitch': [],
        'Yaw': [],
    }

    obs = vec_env.reset()
    # ve_vn_vh_list.append([0., 200., 0.])
    # reward_list.append(0.)
    terminates = [False]
    s_index = 0
    while not terminates[0]:
        timestamps.append(s_index * 1. / STEP_FREQUENCE)

        actions, _ = algo_ppo.predict(observation=obs, deterministic=True)
        tmp_obs = scaled_obs_env.inverse_scale_state(obs[0][-11:])
        tmp_act = scaled_act_env.inverse_scale_action(actions[0])
        if debug:
            print(f"state = {tmp_obs}, action = {tmp_act}")
        obs_list.append(tmp_obs)
        act_list.append(tmp_act)
        obs, rewards, terminates, infos = vec_env.step(actions=actions)
        reward_list.append(rewards[0])
        ve_vn_vh_list.append([infos[0]["origin_obs"]["ve"], infos[0]["origin_obs"]["vn"], infos[0]["origin_obs"]["vh"]])
        
        tackview_data["ISO time"].append(s_index * 1. / STEP_FREQUENCE)
        tackview_data["id"].append('101')
        tackview_data["Longitude"].append(infos[0]["origin_obs"]["lon"])
        tackview_data["Latitude"].append(infos[0]["origin_obs"]["lat"])
        tackview_data["Altitude"].append(infos[0]["origin_obs"]["h"])
        tackview_data["Roll"].append(infos[0]["origin_obs"]["phi"])
        tackview_data["Pitch"].append(infos[0]["origin_obs"]["theta"])
        tackview_data["Yaw"].append(infos[0]["origin_obs"]["psi"])

        print()
        print(tmp_obs)
        # print(scaled_obs_env.inverse_scale_state(obs[0][-11:]))
        print(infos[0]['origin_obs'])
        print(infos[0]['rewards'])
        # exit(0)
        
        if debug:
            print(infos[0])
            print(f"reward = {rewards[0]}")
        
        s_index += 1
    
    obs_list = np.array(obs_list)
    act_list = np.array(act_list)
    ve_vn_vh_list = np.array(ve_vn_vh_list)
    df = pd.DataFrame(data={
        "time": timestamps,
        "s_phi": obs_list[:, 0], 
        "s_theta": obs_list[:, 1], 
        "s_psi": obs_list[:, 2], 
        "s_v": obs_list[:, 3], 
        "s_mu": obs_list[:, 4], 
        "s_chi": obs_list[:, 5], 
        "s_p": obs_list[:, 6], 
        "s_h": obs_list[:, 7], 
        "s_ve": ve_vn_vh_list[:, 0],
        "s_vn": ve_vn_vh_list[:, 1],
        "s_vh": ve_vn_vh_list[:, 2],
        "a_p": act_list[:, 0], 
        "a_nz": act_list[:, 1], 
        "a_pla": act_list[:, 2],
        "reward": reward_list,
        "target_v": obs_list[:, 8], 
        "target_mu": obs_list[:, 9], 
        "target_chi": obs_list[:, 10],
    })
    df.to_csv(Path(__file__).parent / 'test.csv', index=False)

    tackview_df = pd.DataFrame(data=tackview_data)
    tackview_file = PROJECT_ROOT_DIR / "runF16" / "algorithm" / "rollout" / f"tackview_data_{obs_list[-1][8]:.0f}_{obs_list[-1][9]:.0f}_{obs_list[-1][10]:.0f}.txt.acmi"

    if args.save_acmi:
        with open(tackview_file, mode='w', encoding='utf-8-sig') as f:
            f.write("FileType=text/acmi/tacview\n")
            f.write("FileVersion=2.1\n")
            f.write("0,ReferenceTime=2023-05-01T00:00:00Z\n")

        with open(tackview_file, mode='a', encoding='utf-8-sig') as f:
            for index, row in tackview_df.iterrows():
                f.write(f"#{row['ISO time']:.2f}\n")
                out_str = f"{row['id']},T={row['Longitude']}|{row['Latitude']}|{row['Altitude']}|{row['Roll']}|{row['Pitch']}|{row['Yaw']},Name=F16,Color=Red\n"
                f.write(out_str)

    tackview_df.to_csv(Path(__file__).parent / 'tackview_data.csv', index=False)


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