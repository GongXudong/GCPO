from pathlib import Path
import sys
from copy import deepcopy
import numpy as np
import math

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from runF16.env.guidenv_dense_reward import GuideEnvDenseReward

class GuideEnvDenseRewardSampleGoalFromDist(GuideEnvDenseReward):

    def __init__(self, env_config: dict, sample_from_dist: bool=True, success_goal_cnt_dict: dict={}, fail_goal_cnt_dict: dict={}, goal_dist_add_coef: float=0.5) -> None:
        """_summary_

        Args:
            env_config (dict): _description_
            success_goal_dist (dict, optional): {"goal": cnt}. Defaults to {}. goal的形式为：v_mu_chi，如200_10_130.
            fail_goal_dist (dict, optional): {"goal": cnt}. Defaults to {}. goal的形式为：v_mu_chi，如200_10_130.
        """
        super().__init__(env_config)
        self.sample_from_dist: bool = sample_from_dist

        if success_goal_cnt_dict == {} and fail_goal_cnt_dict == {}:
            self.init_goal_dict()
            # self.success_goal_cnt_dict.update(success_goal_cnt_dict)
            # self.fail_goal_cnt_dict.update(fail_goal_cnt_dict)
        else:
            self.success_goal_cnt_dict = success_goal_cnt_dict
            self.fail_goal_cnt_dict = fail_goal_cnt_dict
        
        # 上一次采样目标使用的分布
        self.prev_goal_sample_dist = np.array([])

        self.goal_dist_add_coef = goal_dist_add_coef
    
    def init_goal_dict(self):
        res_dict = {}
        for index, target in self.available_targets.iterrows():
            tmp_v, tmp_mu, tmp_chi = int(target["v"].item()), int(target["mu"].item()), int(target["chi"].item())
            res_dict[f"{tmp_v}_{tmp_mu}_{tmp_chi}"] = 0
        
        self.success_goal_cnt_dict = res_dict
        self.fail_goal_cnt_dict = deepcopy(res_dict)

    def set_success_goal_cnt_dict(self, target_dict: dict):
        self.success_goal_cnt_dict.update(target_dict)

    def set_fail_goal_cnt_dict(self, target_dict: dict):
        self.fail_goal_cnt_dict.update(target_dict)

    def sample_target(self):
        
        if self.sample_from_dist:
            
            # print(f"In env, sample from dist: {sum(list(self.success_goal_cnt_dict.values()))}, {sum(list(self.fail_goal_cnt_dict.values()))}")

            keys = list(self.success_goal_cnt_dict.keys())
            success_values = np.array(list(self.success_goal_cnt_dict.values()))
            fail_values = np.array(list(self.fail_goal_cnt_dict.values()))
            
            tmp_success_values_sum = sum(success_values)
            if math.isclose(tmp_success_values_sum, 0.0, rel_tol=1e-6):
                success_goal_dist = (success_values + 1.) / len(keys)
            else:
                success_goal_dist = success_values / tmp_success_values_sum
            inverse_success_goal_dist = (1. - success_goal_dist) / (len(keys) - 1)
            
            tmp_fail_values_sum = sum(fail_values)
            if math.isclose(tmp_fail_values_sum, 0.0, rel_tol=1e-6):
                fail_goal_dist = (fail_values + 1.) / len(keys)
            else:
                fail_goal_dist = fail_values / tmp_fail_values_sum

            added_sample_dist = self.goal_dist_add_coef * inverse_success_goal_dist + (1. - self.goal_dist_add_coef) * fail_goal_dist
            self.prev_goal_sample_dist = added_sample_dist

            sampled_goal_str = np.random.choice(a=keys, p=added_sample_dist, size=1)
            tmp = [int(item) for item in sampled_goal_str[0].split('_')]

            print(f"sample {tmp[0]}, {tmp[1]}, {tmp[2]} from dist, prob: {added_sample_dist[keys.index(sampled_goal_str[0])]}, success cnt: {self.success_goal_cnt_dict[sampled_goal_str[0]]}, fail cnt: {self.fail_goal_cnt_dict[sampled_goal_str[0]]}.")

            self.target_v = tmp[0]
            self.target_mu = tmp[1]
            self.target_chi = tmp[2]
            # return tmp[0], tmp[1], tmp[2]
        else:
            print("In env, sample from uniform...")
            super().sample_target()