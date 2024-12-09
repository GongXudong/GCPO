from typing import Union
import numpy as np
from collections import namedtuple
from pathlib import Path
import logging
import sys

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from runF16.env.rewards.reward_base import RewardBase
from runF16.utils_my.geometry_utils import angle_of_2_3d_vectors


class DenseRewardBasedOnAngleAndVelocity(RewardBase):
    """基于飞机速度矢量的大小与方向的过程奖励。

    Args:
        RewardBase (_type_): _description_
    """
    def __init__(self, b: float = 1., angle_scale: float = 180., velocity_scale: float = 180., log_history_reward: bool = True, my_logger: logging.Logger = None) -> None:
        self.b = b
        self.angle_scale = angle_scale
        self.velocity_scale = velocity_scale
        super().__init__(is_potential=False, log_history_reward=log_history_reward, my_logger=my_logger)
    
    def get_reward(self, state: Union[namedtuple, np.ndarray], **kwargs) -> float:
        assert "next_ve" in kwargs, "调用前，需要传入next_ve"
        assert "next_vn" in kwargs, "调用前，需要传入next_vn"
        assert "next_vh" in kwargs, "调用前，需要传入next_vh"
        assert "next_state" in kwargs, "调用前，需要把namedtuple类型的next_state放入函数的参数中"

        plane_current_velocity_vector = [
            kwargs["next_ve"], 
            kwargs["next_vn"], 
            kwargs["next_vh"]
        ]
        target_velocity_vector = [
            state[-3] * np.cos(np.deg2rad(state[-2])) * np.sin(np.deg2rad(state[-1])), 
            state[-3] * np.cos(np.deg2rad(state[-2])) * np.cos(np.deg2rad(state[-1])),
            state[-3] * np.sin(np.deg2rad(state[-2])),
        ]
        angle = angle_of_2_3d_vectors(plane_current_velocity_vector, target_velocity_vector)
        angle_base_reward = - np.power(angle / self.angle_scale, self.b)

        # 奖励公式参考论文“Reinforcement learning for UAV attitude control”的4.3节
        next_state = kwargs["next_state"]
        velocity_error = np.abs(next_state.target_v - next_state.v)
        cliped_velocity_error = np.clip(velocity_error / self.velocity_scale, a_min=0., a_max=1.)
        velocity_base_reward = - np.power(cliped_velocity_error, self.b)

        return 0.5 * (angle_base_reward + velocity_base_reward)

    def reset(self):
        super().reset()