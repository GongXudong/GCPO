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

from utils.mega_wrapper import MEGAWrapper
    

class NMRJumpWrapper(Wrapper):
    """与MEGAWrapper一起使用时，注意使用顺序：NMRJumpWrapper(MEGAWrapper(env))
    只用在0-1reward环境上

    Args:
        Wrapper (_type_): _description_
    """
    def __init__(self, env: Env, jump_cnt_target: int = 2, use_original_reward: bool = False, non_terminal_reward: float=0., terminal_reward: float=1.):
        super().__init__(env)
        self.success_info_history = []
        self.jump_cnt_target = jump_cnt_target
        self.jump_cnt = 0
        self.use_original_reward = use_original_reward
        self.non_terminal_reward = non_terminal_reward
        self.terminal_reward = terminal_reward

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        
        # 重置success历史列表
        self.success_info_history = [info["success"]]
        self.jump_cnt = 0

        return obs, info

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info =  super().step(action)
        
        if (self.success_info_history[-1] == False) and (terminated == True):
            self.jump_cnt += 1
            # print(f"step cnt: {len(self.success_info_history)}, jump cnt: {self.jump_cnt}, last terminated: {self.success_info_history[-1]}, current terminated: {terminated}")
        
        self.success_info_history.append(terminated)

        if self.jump_cnt < self.jump_cnt_target:
            new_terminated = False
        else:
            new_terminated = True
            # print(f"\033[32m jump_cnt: {self.jump_cnt}, episode length: {len(self.success_info_history)}\033[0m")

        info["success"] = new_terminated
        info["is_success"] = new_terminated

        if self.use_original_reward:
            new_reward = reward
        else:
            new_reward = self.terminal_reward if new_terminated else self.non_terminal_reward

        return obs, new_reward, new_terminated, truncated, info

    def sync_evaluation_stat(self, evaluation_stat: List):
        """NMRWrapper与MEGAWrapper嵌套使用时，因为MEGAWrapper在内层，所以使用此函数把evaluation信息传递给MEGAWrapper

        Args:
            evaluation_stat (List): _description_
        """
        if is_wrapped(self.env, MEGAWrapper):
            self.env.sync_evaluation_stat(evaluation_stat)
