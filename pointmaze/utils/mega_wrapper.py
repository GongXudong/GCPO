from __future__ import annotations
from gymnasium import ObservationWrapper, ActionWrapper, Env, spaces, Wrapper
from gymnasium.core import Env
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity
from typing import Any, TypeVar, Union, Dict, List
import numpy as np
from pathlib import Path
import sys
from copy import deepcopy

# from gymnasium core.py
ObsType = TypeVar("ObsType")
WrapperObsType = TypeVar("WrapperObsType")
ActType = TypeVar("ActType")
WrapperActType = TypeVar("WrapperActType")
    

class MEGAWrapper(Wrapper):

    def __init__(self, env: Env, sample_N: int, kde_kernel: str="gaussian", kde_bandwidth: float=0.2):
        super().__init__(env)
        self.evaluation_stat: List = []
        self.kde_kernel: str = kde_kernel
        self.kde_bandwidth: float = kde_bandwidth
        self.kde: KernelDensity = None
        self.sample_N = sample_N
        self.need_re_estimate_kde_flag: bool = True

        self.kde_score_threshold: float = 0.0

    def sync_evaluation_stat(self, evaluation_stat: List[Dict]):
        # evaluation_stat中的item：{"goal": xx, "weight": 1.}

        # print(f"sync evaluation statistic: {evaluation_stat}")
        print(f"sync evaluation stat list!!!!!!!!!!!!!!!!!! length: {len(evaluation_stat)}")

        self.evaluation_stat = deepcopy(evaluation_stat)
        self.set_re_estimate()

    def set_re_estimate(self):
        self.need_re_estimate_kde_flag = True
    
    def estimate_kde(self):
        if self.need_re_estimate_kde_flag:
            self.kde = KernelDensity(kernel=self.kde_kernel, bandwidth=self.kde_bandwidth)
            self.kde.fit(
                [item["goal"] for item in self.evaluation_stat],
                sample_weight=[item["weight"] for item in self.evaluation_stat]
            )
            self.kde_score_threshold = np.min(np.exp(self.kde.score_samples([item["goal"] for item in self.evaluation_stat])))
            self.need_re_estimate_kde_flag = False
    
    def sample_goal(self):
        
        if len(self.evaluation_stat) == 0:
            tmp_goal = self.unwrapped.generate_target_goal()
            tmp_noise_goal = self.unwrapped.add_xy_position_noise(tmp_goal)
            print(f"sample from random: {tmp_noise_goal}")
            return tmp_noise_goal
        else:

            self.estimate_kde()

            # sample N candidate goals
            candidate_goals = [self.unwrapped.generate_target_goal() for i in range(self.sample_N)]  # goal's xy
            candidate_noise_goals = [self.unwrapped.add_xy_position_noise(tmp_goal) for tmp_goal in candidate_goals]  # goal's xy
            
            # compute candidate goals' KDE score
            candidate_goal_scores = np.exp(self.kde.score_samples(candidate_noise_goals))

            # print(candidate_goal_scores)

            # kde_score低于self.kde_score_threshold的goal，认为是无法完成的goal，不采样这些目标
            candidate_goal_scores_backup = candidate_goal_scores.copy()
            candidate_goal_scores[candidate_goal_scores < self.kde_score_threshold] = np.Infinity

            if np.all(candidate_goal_scores==np.Infinity):
                # 处理所有采样点的score都小于threshold的情况！！！！
                candidate_goal_index = np.argmax(candidate_goal_scores_backup)
                print(f"\033[31m find max: point {candidate_noise_goals[candidate_goal_index]} with score {candidate_goal_scores_backup[candidate_goal_index]}\033[0m from {candidate_goal_scores_backup}, score_threshold: {self.kde_score_threshold}")
            else:
                candidate_goal_index = np.argmin(candidate_goal_scores)
                print(f"\033[32m find min: point {candidate_noise_goals[candidate_goal_index]} with score {candidate_goal_scores[candidate_goal_index]}\033[0m from {candidate_goal_scores}, score_threshold: {self.kde_score_threshold}")

            return candidate_noise_goals[candidate_goal_index]


    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)

        # sample mega goal
        mega_goal = self.sample_goal()
        
        # modify obs and info
        self.unwrapped.goal = mega_goal
        obs["desired_goal"] = mega_goal.copy()

        info["success"] = bool(
            np.linalg.norm(obs["achieved_goal"] - self.unwrapped.goal) <= 0.45
        )
        info["is_success"] = info["success"]

        return obs, info

