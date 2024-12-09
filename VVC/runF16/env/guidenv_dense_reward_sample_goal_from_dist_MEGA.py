from pathlib import Path
import sys
from copy import deepcopy
import numpy as np
import math

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from runF16.env.guidenv_dense_reward import GuideEnvDenseReward

class GuideEnvDenseRewardSampleGoalFromDistMEGA(GuideEnvDenseReward):
    """更多采样achieved goal分布上概率低的目标用于训练

    Args:
        GuideEnvDenseReward (_type_): _description_
    """
    def __init__(self, env_config: dict, sample_from_dist: bool=True, goal_cnt_dict: dict={}, n_when_sampling_goal: int=10) -> None:
        """_summary_

        Args:
            env_config (dict): _description_
            success_goal_dist (dict, optional): {"goal": cnt}. Defaults to {}. goal的形式为：v_mu_chi，如200_10_130.
        """
        super().__init__(env_config)
        self.sample_from_dist: bool = sample_from_dist
        self.n_when_sampling_goal = n_when_sampling_goal
        self.goal_cnt_dict = self.init_goal_dict() if goal_cnt_dict == {} else goal_cnt_dict
    
    def init_goal_dict(self):
        res_dict = {}
        for index, target in self.available_targets.iterrows():
            tmp_v, tmp_mu, tmp_chi = int(target["v"].item()), int(target["mu"].item()), int(target["chi"].item())
            res_dict[f"{tmp_v}_{tmp_mu}_{tmp_chi}"] = 0
        
        return res_dict

    def set_goal_cnt_dict(self, target_dict: dict):
        self.goal_cnt_dict.update(target_dict)

    def sample_target(self):
        
        if self.sample_from_dist:
            
            # print(f"In env, sample from dist: {sum(list(self.success_goal_cnt_dict.values()))}, {sum(list(self.fail_goal_cnt_dict.values()))}")

            keys = list(self.goal_cnt_dict.keys())
            success_values = np.array(list(self.goal_cnt_dict.values()))
            
            tmp_success_values_sum = sum(success_values)
            if math.isclose(tmp_success_values_sum, 0.0, rel_tol=1e-6):
                # success_goal_cnt_dict中所有值都为0
                if self.debug_mode:
                    print("choose randomly!!!")
                sampled_goal_str = np.random.choice(keys, size=1)[0]
            else:
                # success_goal_cnt_dict中有目标的值大于0
                while True:
                    found_valid_goal = False
                    min_value = 1e9
                    sampled_goal_str = ""

                    sampled_goal_str_arr = np.random.choice(a=keys, size=self.n_when_sampling_goal)
                    
                    for tmp_goal_str in sampled_goal_str_arr:
                        tmp_goal_value = self.goal_cnt_dict[tmp_goal_str]
                        if tmp_goal_value > 0.:
                            found_valid_goal = True
                            if tmp_goal_value < min_value:
                                min_value = tmp_goal_value
                                sampled_goal_str = tmp_goal_str
                    
                    if found_valid_goal:
                        break

            tmp = [int(item) for item in sampled_goal_str.split('_')]

            if self.debug_mode:
                print(f"sample in {self.flag_str}, {tmp[0]}, {tmp[1]}, {tmp[2]} from dist, value: {self.goal_cnt_dict[sampled_goal_str]}.")

            self.target_v = tmp[0]
            self.target_mu = tmp[1]
            self.target_chi = tmp[2]
            # return tmp[0], tmp[1], tmp[2]
        else:
            # print(f"Sample in {self.flag_str}, from uniform...")
            super().sample_target()