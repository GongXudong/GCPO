import unittest
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from runF16.env.rewards.reach_target_reward import ReachTargetReward
from runF16.env.guidenv import GuideEnv


class ReachTargetReardTest(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.reach_target_reward = ReachTargetReward(
            step_frequence=100, integral_time_length=1, 
            v_threshold=10., mu_threshold=1., chi_threshold=1.)
        
        self.state_var_type = GuideEnv.get_state_vars()
        
    def test_1(self):
        """积分长度达到要求，精度也达到要求，返回1.
        """
        state_list = [self.state_var_type(phi=0., theta=0., psi=0., v=200., mu=20., chi=30., p=0., h=0., target_v=200., target_mu=20., target_chi=30.) for i in range(110)]
        res = self.reach_target_reward.get_reward(state_list[-1], state_list=state_list)
        self.assertAlmostEqual(res, 1.)

    def test_2(self):
        """积分长度达到要求，但精度没达到要求，返回0.
        """
        state_list = [self.state_var_type(phi=0., theta=0., psi=0., v=189., mu=20., chi=30., p=0., h=0., target_v=200., target_mu=20., target_chi=30.) for i in range(110)]
        res = self.reach_target_reward.get_reward(state_list[-1], state_list=state_list)
        self.assertAlmostEqual(res, 0.)
    
    def test_3(self):
        """精度达到要求，但积分长度没达到要求，返回0.
        """
        state_list = [self.state_var_type(phi=0., theta=0., psi=0., v=200., mu=20., chi=30., p=0., h=0., target_v=200., target_mu=20., target_chi=30.) for i in range(90)]
        res = self.reach_target_reward.get_reward(state_list[-1], state_list=state_list)
        self.assertAlmostEqual(res, 0.)


if __name__ == "__main__":
    unittest.main()