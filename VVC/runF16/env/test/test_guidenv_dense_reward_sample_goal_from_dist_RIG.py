import unittest
from pathlib import Path
import sys
from copy import deepcopy

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecCheckNan

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from runF16.env.guidenv_dense_reward_sample_goal_from_dist_RIG import GuideEnvDenseRewardSampleGoalFromDistRIG
from runF16.env.utils.sb3_utils import get_vec_env_dense_reward_sample_goal_from_dist


class GuideEnvDenseRewardSmallTest(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.env = GuideEnvDenseRewardSampleGoalFromDistRIG(
            env_config={"use_fixed_target": False},
            goal_cnt_dict={
                "100_0_0": 1,
                "100_0_5": 2,
                "100_0_-5": 3
            }
        )

    def test_sample_goal(self):
        print("test sample goal: ")

        res_cnt = {
            "100_0_0": 0,
            "100_0_5": 0,
            "100_0_-5": 0
        }
    
        for i in range(100):
            self.env.sample_target()
            res = (self.env.target_v, self.env.target_mu, self.env.target_chi)
            res_cnt[f"{str(self.env.target_v)}_{str(self.env.target_mu)}_{str(self.env.target_chi)}"] += 1

        print(res_cnt)


if __name__ == "__main__":
    unittest.main()