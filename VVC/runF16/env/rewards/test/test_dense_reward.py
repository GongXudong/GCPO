import unittest
import numpy as np
from pathlib import Path
import sys
import math

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from runF16.env.rewards.dense_reward import DenseReward
from runF16.env.guidenv import GuideEnv
from runF16.utils_my.geometry_utils import angle_of_2_3d_vectors


class DenseReardTest(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.dense_reward = DenseReward(b=1.)
        
        self.state_var_type = GuideEnv.get_state_vars()
        
    def test_1(self):
        """积分长度达到要求，精度也达到要求，返回1.
        """
        state = self.state_var_type(phi=0., theta=0., psi=0., v=200., mu=20., chi=30., p=0., h=0., target_v=200., target_mu=0., target_chi=0.)
        res = self.dense_reward.get_reward(state, ve=0., vn=1., vh=0., next_ve=0., next_vn=1., next_vh=0.)
        self.assertAlmostEqual(res, 0.)

    def test_2(self):
        """积分长度达到要求，但精度没达到要求，返回0.
        """
        state = self.state_var_type(phi=0., theta=0., psi=0., v=200., mu=20., chi=30., p=0., h=0., target_v=200., target_mu=30., target_chi=60.)
        res = self.dense_reward.get_reward(state, ve=1., vn=1., vh=1.,  next_ve=1., next_vn=1., next_vh=1.)
        angle = angle_of_2_3d_vectors([150., 50 * np.sqrt(3.), 100.], [1, 1, 1])
        self.assertAlmostEqual(res, -angle/180.)
    
    # def test_3(self):
    #     """精度达到要求，但积分长度没达到要求，返回0.
    #     """
    #     state_list = [self.state_var_type(phi=0., theta=0., psi=0., v=200., mu=20., chi=30., p=0., h=0., target_v=200., target_mu=20., target_chi=30.) for i in range(90)]
    #     res = self.reach_target_reward.get_reward(state_list[-1], state_list=state_list)
    #     self.assertAlmostEqual(res, 0.)


if __name__ == "__main__":
    unittest.main()