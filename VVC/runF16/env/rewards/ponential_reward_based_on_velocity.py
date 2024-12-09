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


class PonentialRewardBasedOnVelocity(RewardBase, ABC):
    """基于速度矢量大小的shaping reward

    Args:
        RewardBase (_type_): _description_
        ABC (_type_): _description_
    """

    def __init__(self, b: float=1., scale: float=100., gamma: float=0.99, log_history_reward: bool = True, my_logger: logging.Logger = None) -> None:
        self.b = b
        self.gamma = gamma
        self.scale = scale
        super().__init__(is_potential=False, log_history_reward=log_history_reward, my_logger=my_logger)
    
    def get_reward(self, state: Union[namedtuple, np.ndarray], **kwargs) -> float:
        assert "next_state" in kwargs, "调用前，需要把namedtuple类型的next_state放入函数的参数中"
        assert "done" in kwargs, ""

        next_state = kwargs["next_state"]
        done = kwargs["done"]

        reward = self.gamma * (0. if done else self.phi(next_state)) - self.phi(state)
        
        return self._process(new_reward=reward)

    def phi(self, state: namedtuple):
        delta_v = np.abs(state.target_v - state.v)
        return -np.power(delta_v / self.scale, self.b)

    def reset(self):
        super().reset()
