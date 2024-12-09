from pathlib import Path
import sys
import argparse
from copy import deepcopy
import pandas as pd
import numpy as np

from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import VecCheckNan
from stable_baselines3.common.evaluation import evaluate_policy

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from runF16.algorithm.models.sb3_model import PPOWithBCLoss
from runF16.configs.load_config import load_config
from runF16.env.utils.sb3_utils import get_vec_env_ponential_angle, get_vec_env_ponential_angle_and_velocity, get_vec_env_dense_reward
from runF16.env.guidenv_ponential_angle_and_velocity import GuideEnvPonentialAngleAndVelocity
from runF16.env.guidenv_dense_reward import GuideEnvDenseReward
from runF16.env.utils.my_wrappers import ScaledActionWrapper, ScaledObservationWrapper

parser = argparse.ArgumentParser(description="传入配置文件")
parser.add_argument("--config_file_name", type=str, help="配置文件名", default="framestack_configs/seed1/config_10hz_loss_128_128.json")
parser.add_argument("--reward_type", type=str, help="奖励类型(dense或者ponential)", default="dense")
# parser.add_argument("--algo", type=str, help="输出要测试的算法，rl或bc或rl_bc", default="rl_bc")
parser.add_argument("--task_difficulty", type=int, default=-1, help="测试使用的任务难度")
parser.add_argument("--task_num", type=int, default=100, help="测试使用的任务数量")
parser.add_argument("--res_save_filename", type=str, default='res.csv', help="存储结果的文件名")
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


def rollout(bc_policy_dir: Path, rl_bc_policy_dir: Path, debug: bool=False, bc_model_save_name: str="bc_checkpoint", rl_bc_model_save_name: str="best_model"):
    if args.reward_type == "dense":
        env_class = GuideEnvDenseReward
    elif args.reward_type == "ponential":
        env_class = GuideEnvPonentialAngleAndVelocity
    else:
        print("奖励类型只能是dense或者ponential")
    
    helper_env = env_class(env_config={
        "step_frequence": STEP_FREQUENCE,
        "max_simulate_time": MAX_SIMULATE_TIME,
        "gamma": GAMMA,
        "reward_scale_factor_b": 0.5,
        "available_targets_dir": PROJECT_ROOT_DIR / "my_logs_for_parallel" / EXPERT_DATA_DIR,
        "sample_target_noise_std": [0., 0., 0.],
        "use_fixed_target": args.use_fixed_target,
        "target_v": args.target_v,
        "target_mu": args.target_mu,
        "target_chi": args.target_chi
    })
    scaled_obs_env = ScaledObservationWrapper(helper_env)
    scaled_act_env = ScaledActionWrapper(scaled_obs_env)

    bc_algo_ppo = PPOWithBCLoss.load(
        str((bc_policy_dir / bc_model_save_name).absolute()),
        custom_objects={
            "observation_space": scaled_act_env.observation_space,
            "action_space": scaled_act_env.action_space,
            "bc_trained_algo": None
        }
    )

    rl_bc_algo_ppo = PPOWithBCLoss.load(
        str((rl_bc_policy_dir / rl_bc_model_save_name).absolute()), 
        custom_objects={
            "observation_space": scaled_act_env.observation_space,
            "action_space": scaled_act_env.action_space,
            "bc_trained_algo": None
        }
    )
    rl_bc_algo_ppo.policy.set_training_mode(False)

    if args.use_fixed_target:
        bc_rewards, bc_ep_lengths = evaluate_policy(bc_algo_ppo, scaled_act_env, n_eval_episodes=1, deterministic=True, return_episode_rewards=True)
        rl_bc_rewards, rl_bc_ep_lengths = evaluate_policy(rl_bc_algo_ppo, scaled_act_env, n_eval_episodes=1, deterministic=True, return_episode_rewards=True)
        print("bc: ", bc_rewards[0], bc_ep_lengths[0])
        print("rl_bc: ", rl_bc_rewards[0], rl_bc_ep_lengths[0])
    else:
        tmp_res_filename = PROJECT_ROOT_DIR / "my_logs_for_parallel" / EXPERT_DATA_DIR / "res.csv"
        targets_df = pd.read_csv(tmp_res_filename)
        if "difficulty_calc_by_error" not in targets_df.columns:
            targets_df = pd.read_csv(PROJECT_ROOT_DIR / "runF16" / "algorithm" / "data_sb3" / "label_difficulty" / "res_with_difficulty.csv")
        
        if args.task_difficulty == -1:
            # 所有专家能完成的轨迹
            targets_df = targets_df[targets_df.length > 0]
        elif args.task_difficulty == -2:
            # 所有轨迹，50715个
            pass
        elif args.task_difficulty == -3:
            targets_df = targets_df[targets_df.length == 0]
        elif args.task_difficulty >= 0:
            targets_df = targets_df[targets_df.difficulty_calc_by_error == args.task_difficulty]
        else:
            print(f"task_difficulty的取值只能是[-3, -2, -1, 1, 2, 3, 4, 5]")

        print(f"测试空间任务数：{len(targets_df)}")

        helper_env.use_fixed_target = True
        res = {
            'v': [],
            'mu': [],
            'chi': [],
            'expert length': [],
            'bc': [],
            'bc_success': [],
            'rl_bc': [],
            'rl_bc_success': []
        }

        def rollout_one_trajactory(algo):
            # 采样
            obs, info = scaled_act_env.reset()
            terminate = False
            s_index = 0
            while not terminate:
                action, _ = algo.predict(observation=obs, deterministic=True)
                tmp_obs = scaled_obs_env.inverse_scale_state(obs[-11:])
                obs, reward, terminate, truncated, info = scaled_act_env.step(action=action)
                s_index += 1
            
            return s_index, info

        for item in np.random.randint(low=0, high=len(targets_df), size=args.task_num):
            tmp = targets_df.iloc[item]
            target_v, target_mu, target_chi, expert_len = tmp['v'], tmp['mu'], tmp['chi'], tmp['length']
            helper_env.target_v = target_v
            helper_env.target_mu = target_mu
            helper_env.target_chi = target_chi
            helper_env.target_expert_steps = expert_len
            
            # bc_rewards, bc_ep_lengths = evaluate_policy(bc_algo_ppo, scaled_act_env, n_eval_episodes=1, deterministic=True, return_episode_rewards=True)
            # rl_bc_rewards, rl_bc_ep_lengths = evaluate_policy(rl_bc_algo_ppo, scaled_act_env, n_eval_episodes=1, deterministic=True, return_episode_rewards=True)
            
            bc_ep_length, bc_info = rollout_one_trajactory(bc_algo_ppo)
            rl_bc_ep_length, rl_bc_info = rollout_one_trajactory(rl_bc_algo_ppo)

            # if bc_info["is_success"] and rl_bc_info["is_success"]:
            res["v"].append(target_v)
            res["mu"].append(target_mu)
            res["chi"].append(target_chi)
            res["expert length"].append(expert_len)
            res["bc"].append(bc_ep_length)
            res["bc_success"].append(int(bc_info["is_success"]))
            res["rl_bc"].append(rl_bc_ep_length)
            res["rl_bc_success"].append(int(rl_bc_info["is_success"]))
            
            # print(target_v, target_mu, target_chi, expert_len, bc_rewards[0], bc_ep_lengths[0], rl_bc_rewards[0], rl_bc_ep_lengths[0])
        res_df = pd.DataFrame(data=res)
        print(res_df)
        res_df.to_csv(PROJECT_ROOT_DIR / "runF16" / "algorithm" / "rollout" / args.res_save_filename, index=False)


if __name__ == "__main__":

    rollout(
        bc_policy_dir=PROJECT_ROOT_DIR / "runF16" / "algorithm" / "checkpoints_sb3" / "bc" / BC_EXPERIMENT_NAME,
        bc_model_save_name="bc_checkpoint",
        rl_bc_policy_dir=PROJECT_ROOT_DIR / "runF16" / "algorithm" / "checkpoints_sb3" / "rl" / RL_EXPERIMENT_NAME,
        rl_bc_model_save_name="best_model"
    )
