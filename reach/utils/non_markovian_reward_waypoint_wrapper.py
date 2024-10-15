from __future__ import annotations
from gymnasium import ObservationWrapper, ActionWrapper, Env, spaces, Wrapper
from gymnasium.core import Env
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity
from typing import Any, SupportsFloat, TypeVar, Union, Dict, List
import numpy as np
from pathlib import Path
import sys
from copy import deepcopy

from stable_baselines3.common.env_util import is_wrapped

# from gymnasium core.py
ObsType = TypeVar("ObsType")
WrapperObsType = TypeVar("WrapperObsType")
ActType = TypeVar("ActType")
WrapperActType = TypeVar("WrapperActType")

PROJECT_ROOT_DIR = Path(__file__).parent.parent

if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))
    

class NMRWayPointWrapper(Wrapper):
    """与MEGAWrapper一起使用时，注意使用顺序：NMRJumpWrapper(MEGAWrapper(env))
    只用在0-1reward环境上

    Args:
        Wrapper (_type_): _description_
    """
    def __init__(self, env: Env, target_waypoint_delta: np.ndarray = np.array([0.0, 0.0, 0.1]), use_original_reward: bool = False, non_terminal_reward: float=0., terminal_reward: float=1.):
        super().__init__(env)
        self.reach_way_point: bool = False
        self.target_waypoint_delta = target_waypoint_delta
        self.use_original_reward = use_original_reward
        self.non_terminal_reward = non_terminal_reward
        self.terminal_reward = terminal_reward

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        
        # 重置waypoint
        self.waypoint = self.unwrapped.task.goal + self.target_waypoint_delta
        self.reach_way_point = False

        return obs, info

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info =  super().step(action)
        
        # 判断是否曾经到达waypoint
        if not self.reach_way_point:
            self.reach_way_point = bool(self.unwrapped.task.is_success(obs["achieved_goal"], self.waypoint))
            # if self.reach_way_point:
                # print(f"reach way point: {self.waypoint} by {obs['achieved_goal']}")

        # 判断terminated
        new_terminated = False
        reach_goal = bool(self.unwrapped.task.is_success(obs["achieved_goal"], self.unwrapped.task.get_goal()))
        if self.reach_way_point and reach_goal:
            # print(f"reach target {self.unwrapped.task.get_goal()} by {obs['achieved_goal']}")
            new_terminated = True

        info["is_success"] = new_terminated

        if self.use_original_reward:
            new_reward = reward
        else:
            new_reward = self.terminal_reward if new_terminated else self.non_terminal_reward

        return obs, new_reward, new_terminated, truncated, info
