from abc import ABC, abstractmethod
import numpy as np
from collections import namedtuple
from typing import Union, Callable
import logging
from pathlib import Path
import sys

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from runF16.env.rewards.reward_base import RewardBase
from runF16.utils_my.geometry_utils import angle_of_2_3d_vectors


class PonentialRewardBasedOnAngle(RewardBase, ABC):
    """基于速度矢量方向的shaping reward

    Args:
        RewardBase (_type_): _description_
        ABC (_type_): _description_
    """

    def __init__(self, b: float=1., gamma: float=0.99, log_history_reward: bool = True, my_logger: logging.Logger = None) -> None:
        self.b = b
        self.gamma = gamma
        super().__init__(is_potential=False, log_history_reward=log_history_reward, my_logger=my_logger)
    
    def get_reward(self, state: Union[namedtuple, np.ndarray], **kwargs) -> float:
        assert "ve" in kwargs, "调用前，需要传入ve"
        assert "vn" in kwargs, "调用前，需要传入vn"
        assert "vh" in kwargs, "调用前，需要传入vh"
        assert "next_ve" in kwargs, "调用前，需要传入next_ve"
        assert "next_vn" in kwargs, "调用前，需要传入next_vn"
        assert "next_vh" in kwargs, "调用前，需要传入next_vh"
        assert "next_state" in kwargs, "调用前，需要把namedtuple类型的next_state放入函数的参数中"
        assert "done" in kwargs, ""

        next_state = kwargs["next_state"]
        done = kwargs["done"]
        ve, vn, vh = kwargs["ve"], kwargs["vn"], kwargs["vh"]
        next_ve, next_vn, next_vh = kwargs["next_ve"], kwargs["next_vn"], kwargs["next_vh"]

        reward = self.gamma * (0. if done else self.phi(next_state, next_ve, next_vn, next_vh)) - self.phi(state, ve, vn, vh)
        
        return self._process(new_reward=reward)

    def phi(self, state: namedtuple, ve: float, vn: float, vh: float):
        plane_current_velocity_vector = [
            ve, 
            vn, 
            vh
        ]

        target_velocity_vector = [
            state.target_v * np.cos(np.deg2rad(state.target_mu)) * np.sin(np.deg2rad(state.target_chi)), 
            state.target_v * np.cos(np.deg2rad(state.target_mu)) * np.cos(np.deg2rad(state.target_chi)),
            state.target_v * np.sin(np.deg2rad(state.target_mu)),
        ]

        angle = angle_of_2_3d_vectors(plane_current_velocity_vector, target_velocity_vector)

        return -np.power(angle / 180., self.b)

    def reset(self):
        super().reset()