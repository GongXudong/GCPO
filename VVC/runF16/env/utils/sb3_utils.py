import logging
import sys
from pathlib import Path
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from runF16.env.guidenv import GuideEnv
from runF16.env.guidenv_ponential_angle import GuideEnvPonentialAngle
from runF16.env.guidenv_scaled_ponential import GuideEnvScaledPonential
from runF16.env.guidenv_ponential_angle_and_velocity import GuideEnvPonentialAngleAndVelocity
from runF16.env.guidenv_dense_reward import GuideEnvDenseReward
from runF16.env.guidenv_dense_reward_small_divide_by_chi import GuideEnvDenseRewardSmallDivideByChi
from runF16.env.guidenv_dense_reward_small_divide_by_difficulty import GuideEnvDenseRewardSmallDivideByDifficulty
from runF16.env.guidenv_dense_reward_filter_by_difficulty import GuideEnvDenseRewardFilteredByDifficulty
from runF16.env.guidenv_dense_reward_sample_goal_from_dist import GuideEnvDenseRewardSampleGoalFromDist
from runF16.env.guidenv_dense_reward_sample_goal_from_dist_MEGA import GuideEnvDenseRewardSampleGoalFromDistMEGA
from runF16.env.guidenv_dense_reward_sample_goal_from_dist_RIG import GuideEnvDenseRewardSampleGoalFromDistRIG
from runF16.env.guidenv_dense_reward_sample_goal_from_dist_DISCERN import GuideEnvDenseRewardSampleGoalFromDistDISCERN
from runF16.env.guidenv_dense_reward_MR import GuideEnvDenseRewardMR
from runF16.env.guidenv_dense_reward_MR_sample_goal_from_dist_MEGA import GuideEnvDenseRewardMRSampleGoalFromDistMEGA
from runF16.env.utils.my_wrappers import ScaledActionWrapper, ScaledObservationWrapper


def make_env(rank: int, seed: int = 0, **kwargs):
    """
    Utility function for multiprocessed env.

    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = GuideEnv(env_config={
            # "max_simulate_time": 120,
            # "gamma": 0.9995,  # 0.9995^2000 = 0.3678, 0.999^2000 = 0.1352
            # "sample_target_noise_std": [0., 0., 0.],
            **kwargs
        })
        env = ScaledActionWrapper(ScaledObservationWrapper(env))
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

def get_vec_env(num_process: int=10, **kwargs):
    return SubprocVecEnv([make_env(rank=i, **kwargs) for i in range(num_process)])

def make_env_dense_reward(rank: int, seed: int = 0, **kwargs):
    """
    放大shaping reward的环境

    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = GuideEnvDenseReward(env_config={
            # "max_simulate_time": 120,
            # "gamma": 0.9995,  # 0.9995^2000 = 0.3678, 0.999^2000 = 0.1352
            # "sample_target_noise_std": [0., 0., 0.],
            # "dense_reward_b": 1.,
            **kwargs
        })
        env = ScaledActionWrapper(ScaledObservationWrapper(env))
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

def get_vec_env_dense_reward(num_process: int=10, **kwargs):
    return SubprocVecEnv([make_env_dense_reward(rank=i, **kwargs) for i in range(num_process)])

def make_env_dense_reward_sample_goal_from_dist(rank: int, seed: int = 0, **kwargs):
    """
    放大shaping reward的环境

    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = GuideEnvDenseRewardSampleGoalFromDist(env_config={
            # "max_simulate_time": 120,
            # "gamma": 0.9995,  # 0.9995^2000 = 0.3678, 0.999^2000 = 0.1352
            # "sample_target_noise_std": [0., 0., 0.],
            # "dense_reward_b": 1.,
            **kwargs
        })
        env = ScaledActionWrapper(ScaledObservationWrapper(env))
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

def get_vec_env_dense_reward_sample_goal_from_dist(num_process: int=10, **kwargs):
    return SubprocVecEnv([make_env_dense_reward_sample_goal_from_dist(rank=i, **kwargs) for i in range(num_process)])

def make_env_dense_reward_MR(rank: int, seed: int = 0, **kwargs):
    """
    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = GuideEnvDenseRewardMR(
            env_config={
                # "max_simulate_time": 120,
                # "gamma": 0.9995,  # 0.9995^2000 = 0.3678, 0.999^2000 = 0.1352
                # "sample_target_noise_std": [0., 0., 0.],
                # "dense_reward_b": 1.,
                **kwargs
            },
        )
        env = ScaledActionWrapper(ScaledObservationWrapper(env))
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

def get_vec_env_dense_reward_MR(num_process: int=10, **kwargs):
    return SubprocVecEnv([make_env_dense_reward_MR(rank=i, **kwargs) for i in range(num_process)])

def make_env_dense_reward_MR_sample_goal_from_dist_mega(rank: int, seed: int = 0, **kwargs):
    """
    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = GuideEnvDenseRewardMRSampleGoalFromDistMEGA(
            env_config={
                # "max_simulate_time": 120,
                # "gamma": 0.9995,  # 0.9995^2000 = 0.3678, 0.999^2000 = 0.1352
                # "sample_target_noise_std": [0., 0., 0.],
                # "dense_reward_b": 1.,
                **kwargs
            },
            sample_from_dist=kwargs["sample_from_dist"],
            goal_cnt_dict=kwargs["goal_cnt_dict"],
            n_when_sampling_goal=kwargs["n_when_sampling_goal"]
        )
        env = ScaledActionWrapper(ScaledObservationWrapper(env))
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

def get_vec_env_dense_reward_MR_sample_goal_from_dist_mega(num_process: int=10, **kwargs):
    return SubprocVecEnv([make_env_dense_reward_MR_sample_goal_from_dist_mega(rank=i, **kwargs) for i in range(num_process)])

def make_env_dense_reward_sample_goal_from_dist_mega(rank: int, seed: int = 0, **kwargs):
    """
    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = GuideEnvDenseRewardSampleGoalFromDistMEGA(
            env_config={
                # "max_simulate_time": 120,
                # "gamma": 0.9995,  # 0.9995^2000 = 0.3678, 0.999^2000 = 0.1352
                # "sample_target_noise_std": [0., 0., 0.],
                # "dense_reward_b": 1.,
                **kwargs
            },
            sample_from_dist=kwargs["sample_from_dist"],
            goal_cnt_dict=kwargs["goal_cnt_dict"],
            n_when_sampling_goal=kwargs["n_when_sampling_goal"]
        )
        env = ScaledActionWrapper(ScaledObservationWrapper(env))
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

def get_vec_env_dense_reward_sample_goal_from_dist_mega(num_process: int=10, **kwargs):
    return SubprocVecEnv([make_env_dense_reward_sample_goal_from_dist_mega(rank=i, **kwargs) for i in range(num_process)])

def make_env_dense_reward_sample_goal_from_dist_rig(rank: int, seed: int = 0, **kwargs):
    """
    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = GuideEnvDenseRewardSampleGoalFromDistRIG(
            env_config={
                # "max_simulate_time": 120,
                # "gamma": 0.9995,  # 0.9995^2000 = 0.3678, 0.999^2000 = 0.1352
                # "sample_target_noise_std": [0., 0., 0.],
                # "dense_reward_b": 1.,
                **kwargs
            },
            sample_from_dist=kwargs["sample_from_dist"],
            goal_cnt_dict=kwargs["goal_cnt_dict"],
        )
        env = ScaledActionWrapper(ScaledObservationWrapper(env))
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

def get_vec_env_dense_reward_sample_goal_from_dist_rig(num_process: int=10, **kwargs):
    return SubprocVecEnv([make_env_dense_reward_sample_goal_from_dist_rig(rank=i, **kwargs) for i in range(num_process)])

def make_env_dense_reward_sample_goal_from_dist_discern(rank: int, seed: int = 0, **kwargs):
    """
    放大shaping reward的环境

    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = GuideEnvDenseRewardSampleGoalFromDistDISCERN(
            env_config={
                # "max_simulate_time": 120,
                # "gamma": 0.9995,  # 0.9995^2000 = 0.3678, 0.999^2000 = 0.1352
                # "sample_target_noise_std": [0., 0., 0.],
                # "dense_reward_b": 1.,
                **kwargs
            },
            sample_from_dist=kwargs["sample_from_dist"],
            goal_cnt_dict=kwargs["goal_cnt_dict"],
        )
        env = ScaledActionWrapper(ScaledObservationWrapper(env))
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

def get_vec_env_dense_reward_sample_goal_from_dist_discern(num_process: int=10, **kwargs):
    return SubprocVecEnv([make_env_dense_reward_sample_goal_from_dist_discern(rank=i, **kwargs) for i in range(num_process)])

def make_env_dense_reward_small_divide_by_chi(rank: int, seed: int = 0, **kwargs):
    """
    放大shaping reward的环境

    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = GuideEnvDenseRewardSmallDivideByChi(env_config={
            # "max_simulate_time": 120,
            # "gamma": 0.9995,  # 0.9995^2000 = 0.3678, 0.999^2000 = 0.1352
            # "sample_target_noise_std": [0., 0., 0.],
            # "dense_reward_b": 1.,
            **kwargs
        })
        env = ScaledActionWrapper(ScaledObservationWrapper(env))
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

def get_vec_env_dense_reward_small_divide_by_chi(num_process: int=10, **kwargs):
    return SubprocVecEnv([make_env_dense_reward_small_divide_by_chi(rank=i, **kwargs) for i in range(num_process)])

def make_env_dense_reward_small_divide_by_difficulty(rank: int, seed: int = 0, **kwargs):
    """
    放大shaping reward的环境

    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = GuideEnvDenseRewardSmallDivideByDifficulty(env_config={
            # "max_simulate_time": 120,
            # "gamma": 0.9995,  # 0.9995^2000 = 0.3678, 0.999^2000 = 0.1352
            # "sample_target_noise_std": [0., 0., 0.],
            # "dense_reward_b": 1.,
            **kwargs
        })
        env = ScaledActionWrapper(ScaledObservationWrapper(env))
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

def get_vec_env_dense_reward_small_divide_by_difficulty(num_process: int=10, **kwargs):
    return SubprocVecEnv([make_env_dense_reward_small_divide_by_difficulty(rank=i, **kwargs) for i in range(num_process)])

def make_env_dense_reward_filter_by_difficulty(rank: int, seed: int = 0, **kwargs):
    """
    放大shaping reward的环境

    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = GuideEnvDenseRewardFilteredByDifficulty(env_config={
            # "max_simulate_time": 120,
            # "gamma": 0.9995,  # 0.9995^2000 = 0.3678, 0.999^2000 = 0.1352
            # "sample_target_noise_std": [0., 0., 0.],
            # "dense_reward_b": 1.,
            **kwargs
        })
        env = ScaledActionWrapper(ScaledObservationWrapper(env))
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

def get_vec_env_dense_reward_filter_by_difficulty(num_process: int=10, **kwargs):
    return SubprocVecEnv([make_env_dense_reward_filter_by_difficulty(rank=i, **kwargs) for i in range(num_process)])

def make_env_ponential_angle(rank: int, seed: int = 0, **kwargs):
    """
    放大shaping reward的环境

    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = GuideEnvPonentialAngle(env_config={
            # "max_simulate_time": 120,
            # "gamma": 0.9995,  # 0.9995^2000 = 0.3678, 0.999^2000 = 0.1352
            # "sample_target_noise_std": [0., 0., 0.],
            **kwargs
        })
        env = ScaledActionWrapper(ScaledObservationWrapper(env))
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

def get_vec_env_ponential_angle(num_process: int=10, **kwargs):
    return SubprocVecEnv([make_env_ponential_angle(rank=i, **kwargs) for i in range(num_process)])

def make_env_scaled_ponential(rank: int, seed: int = 0, **kwargs):
    """
    放大shaping reward的环境

    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = GuideEnvScaledPonential(env_config={
            # "max_simulate_time": 120,
            # "gamma": 0.9995,  # 0.9995^2000 = 0.3678, 0.999^2000 = 0.1352
            # "sample_target_noise_std": [0., 0., 0.],
            "scale_ponential_reward_coef": 100.,
            **kwargs
        })
        env = ScaledActionWrapper(ScaledObservationWrapper(env))
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

def get_vec_env_scaled_ponential(num_process: int=10, **kwargs):
    return SubprocVecEnv([make_env_scaled_ponential(rank=i, **kwargs) for i in range(num_process)])

def make_env_ponential_angle_and_velocity(rank: int, seed: int = 0, **kwargs):
    """
    放大shaping reward的环境

    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = GuideEnvPonentialAngleAndVelocity(env_config={
            # "max_simulate_time": 120,
            # "gamma": 0.9995,  # 0.9995^2000 = 0.3678, 0.999^2000 = 0.1352
            # "sample_target_noise_std": [0., 0., 0.],
            **kwargs
        })
        env = ScaledActionWrapper(ScaledObservationWrapper(env))
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

def get_vec_env_ponential_angle_and_velocity(num_process: int=10, **kwargs):
    return SubprocVecEnv([make_env_ponential_angle_and_velocity(rank=i, **kwargs) for i in range(num_process)])