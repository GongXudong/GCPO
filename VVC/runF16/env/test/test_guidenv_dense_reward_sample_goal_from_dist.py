import unittest
from pathlib import Path
import sys
from copy import deepcopy

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecCheckNan

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from runF16.env.guidenv_dense_reward_sample_goal_from_dist import GuideEnvDenseRewardSampleGoalFromDist
from runF16.env.utils.sb3_utils import get_vec_env_dense_reward_sample_goal_from_dist


class GuideEnvDenseRewardSmallTest(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.env = GuideEnvDenseRewardSampleGoalFromDist(
            env_config={"use_fixed_target": False}
        )

    def test_initial_goal_dict(self):
        print("test init goal dict: ")
        self.env.success_goal_cnt_dict["300_-10_155"] += 1
        
        print(len(self.env.success_goal_cnt_dict.keys()))
        print(self.env.success_goal_cnt_dict["300_-10_155"], self.env.fail_goal_cnt_dict["300_-10_155"])

    def test_sample_goal(self):
        print("test sample goal: ")
        for ky in self.env.success_goal_cnt_dict.keys():
            if ky != "300_-10_155":
                self.env.success_goal_cnt_dict[ky] = 1
            else:
                self.env.fail_goal_cnt_dict[ky] = 1
        
        self.env.sample_target()
        res = (self.env.target_v, self.env.target_mu, self.env.target_chi)
        print(self.env.prev_goal_sample_dist)
        print(res)

    def test_multi_envs(self):
        print("test multi envs: ") 
        
        self.env1 = GuideEnvDenseRewardSampleGoalFromDist(
            env_config={"use_fixed_target": False}
        )
        self.env1.success_goal_cnt_dict["300_-10_155"] += 1

        success_dict_cnt = self.env1.success_goal_cnt_dict
        fail_dict_cnt = self.env1.fail_goal_cnt_dict

        self.env2 = GuideEnvDenseRewardSampleGoalFromDist(
            env_config={"use_fixed_target": False},
            success_goal_cnt_dict=success_dict_cnt,
            fail_goal_cnt_dict=fail_dict_cnt
        )
        print(self.env2.success_goal_cnt_dict["300_-10_155"])
        
        self.env2.success_goal_cnt_dict["300_-10_155"] += 1

        print(self.env1.success_goal_cnt_dict["300_-10_155"])
    
    def test_multi_envs2(self):
        print("test multi envs 2: ")
        env_config_dict_in_training = {
            "num_process": 2, 
            "logger": None, 
            "step_frequence": 10,
            "max_simulate_time": 40,
            "gamma": 0.995,
            "reward_scale_factor_b": 0.5,  # GuideEnvDenseReward中使用，取值小于1效果相当于在数值上放大了奖励，大于1相当于缩小了奖励
            "available_targets_dir": PROJECT_ROOT_DIR / "my_logs_for_parallel" / "10hz_10_5_5",
            "sample_target_noise_std": [0., 0., 0.],
        }

        # 创建临时env，只为了获取success_goal_cnt_dict和fail_goal_cnt_dict
        # 这两个字典用于训练环境采样goal，在训练过程中的测试中，更新这两个字典中的统计信息
        tmp_env = GuideEnvDenseRewardSampleGoalFromDist(
            env_config=env_config_dict_in_training,
            success_goal_cnt_dict={},
            fail_goal_cnt_dict={},
            goal_dist_add_coef=0.5
        )
        success_goal_cnt_dict = tmp_env.success_goal_cnt_dict
        fail_goal_cnt_dict = tmp_env.fail_goal_cnt_dict

        env_config_dict_in_training.update({
            "sample_from_dist": True,
            "success_goal_cnt_dict": success_goal_cnt_dict,
            "fail_goal_cnt_dict": fail_goal_cnt_dict,
            "goal_dist_add_coef": 0.5
        })

        vec_env = VecCheckNan(get_vec_env_dense_reward_sample_goal_from_dist(
            **env_config_dict_in_training
        ))
        print(type(vec_env.venv))

        tmp_dict = {
            "300_-20_130": 1
        }

        vec_env.set_attr("success_goal_cnt_dict", tmp_dict)
        vec_env.set_attr("fail_goal_cnt_dict", tmp_dict)

        print("-----------------")
        res = vec_env.get_attr("fail_goal_cnt_dict")
        print(type(res), res)
        print("-----------------")
        fail_goal_cnt_dict.update({
            "300_-10_155": 1
        })
        fail_goal_cnt_dict["300_-10_155"] += 1
        vec_env.set_attr("success_goal_cnt_dict", success_goal_cnt_dict)
        vec_env.set_attr("fail_goal_cnt_dict", fail_goal_cnt_dict)
        res2 = vec_env.get_attr("fail_goal_cnt_dict")
        print(type(res2), res2)


if __name__ == "__main__":
    unittest.main()