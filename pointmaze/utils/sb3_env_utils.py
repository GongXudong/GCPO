from typing import List

import gymnasium as gym
import sys
from pathlib import Path

from stable_baselines3.common.utils import set_random_seed
from utils.sb3_env_wrappers import ScaledObservationWrapper
from sklearn.preprocessing import StandardScaler

import minari
from imitation.data import rollout
from imitation.data.types import TransitionsMinimal, Transitions, Trajectory

PROJECT_ROOT_DIR = Path(__file__).parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from utils import load_data
from utils.non_markovian_reward_wrapper import NMRWrapper
from utils.non_markovian_reward_jump_wrapper import NMRJumpWrapper
from utils.mega_wrapper import MEGAWrapper


def make_env(
        env_id: str, 
        rank: int, seed: int = 0, 
        use_mega: bool = False, mega_smaple_N: int = 10, 
        wrap_with_nmr: bool = False, 
        nmr_jump: bool = True,
        nmr_jump_cnt: int = 2,
        nmr_length: int = 10, 
        nmr_use_original_reward: float = 0., nmr_non_terminal_reward: float = 0., nmr_terminal_reward: float = 1.,
        scale_obs: bool = False, 
        dataset_minari_id: str="", 
        **kwargs
    ):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = gym.make(env_id, **kwargs)
        
        # 注意wrapper的顺序：ScaledObservationWrapper(NMRWrapper(MEGAWrapper(env)))

        if use_mega:
            print(f"wrap env with MEGA, {mega_smaple_N}")
            env = MEGAWrapper(env, sample_N=mega_smaple_N)

        if wrap_with_nmr:
            if nmr_jump:
                print(f"wrap env with NMR Jump, {nmr_jump_cnt}, {nmr_use_original_reward}, {nmr_non_terminal_reward}, {nmr_terminal_reward}")
                env=NMRJumpWrapper(env, jump_cnt_target=nmr_jump_cnt, use_original_reward=nmr_use_original_reward, non_terminal_reward=nmr_non_terminal_reward, terminal_reward=nmr_terminal_reward)
            else:
                print(f"wrap env with NMR, {nmr_length}, {nmr_use_original_reward}, {nmr_non_terminal_reward}, {nmr_terminal_reward}")
                env = NMRWrapper(env, nmr_length=nmr_length, use_original_reward=nmr_use_original_reward, non_terminal_reward=nmr_non_terminal_reward, terminal_reward=nmr_terminal_reward)

        if scale_obs:
            trajs: List[Trajectory] = load_data.load_data(dataset_minari_id)
            transitions = rollout.flatten_trajectories(trajs)
            
            scaled_obs, scaler = load_data.scale_obs(transitions.obs)
            env = ScaledObservationWrapper(env=env, scaler=scaler)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


def make_env_from_minari(minari_dataset_id: str, rank: int, seed: int=0, scale_obs: bool = False, is_eval: bool = False, **kwargs):

    def _init():
        minari_dataset, trajs = load_data.load_data2(minari_dataset_id)
        env = minari_dataset.recover_environment(eval_env=is_eval, **kwargs)
        if scale_obs:
            transitions = rollout.flatten_trajectories(trajs)
            
            scaled_obs, scaler = load_data.scale_obs(transitions.obs)
            env = ScaledObservationWrapper(env=env, scaler=scaler)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init