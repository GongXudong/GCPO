import unittest
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from runF16.env.terminations.reach_target_termination2 import ReachTargetTermination2
from runF16.env.guidenv import GuideEnv


class ReachTargetTerminationTest(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        env_config = {
            "use_fixed_target": False,
            "target_v": 200, 
            "target_mu": 20, 
            "target_chi": 30,
            "h0": 5000,
            "v0": 200,
            "step_frequence": 10,
            "max_simulate_time": 40,
            "gamma": 0.995,
            "available_targets_dir": Path(__file__).parent.parent.parent / "my_logs_for_parallel" / "10_5_5",
            "available_targets_filename": "res.csv",
            "sample_target_noise_std": [5, 0.5, 0.5],  # noise std for [v, mu, chi]
            "my_logger": None
        }
        self.reach_target_termination = ReachTargetTermination2(
            integral_time_length=1,
            v_threshold=10., 
            angle_threshold=1.,
            termination_reward=1.,
            env_config=env_config
        )
        self.state_var_type = GuideEnv.get_state_vars()
    
    def test_1(self):
        """积分长度达到要求，精度也达到要求，返回True
        """
        episode_length = 11
        state_list = [self.state_var_type(phi=0., theta=0., psi=0., v=200., mu=19.1, chi=30., p=0., h=0., target_v=200., target_mu=20., target_chi=30.) for i in range(episode_length)]
        res = self.reach_target_termination.get_termination(state_list[-1], state_list=state_list)
        self.assertTrue(res[0])
        self.assertFalse(res[1])

        res2 = self.reach_target_termination.get_termination_and_reward(state_list[-1], state_list=state_list, step_cnt=100)
        self.assertTrue(res2[0])
        self.assertFalse(res2[1])
        self.assertAlmostEqual(res2[2], 1.)

    def test_2(self):
        """积分长度达到要求，但速度大小精度没达到要求，返回False
        """
        episode_length = 11
        state_list = [self.state_var_type(phi=0., theta=0., psi=0., v=189., mu=20., chi=30., p=0., h=0., target_v=200., target_mu=20., target_chi=30.) for i in range(episode_length)]
        res = self.reach_target_termination.get_termination(state_list[-1], state_list=state_list)
        self.assertFalse(res[0])
        self.assertFalse(res[1])

        res2 = self.reach_target_termination.get_termination_and_reward(state_list[-1], state_list=state_list, step_cnt=100)
        self.assertFalse(res2[0])
        self.assertFalse(res2[1])
        self.assertAlmostEqual(res2[2], 0.)

    def test_3(self):
        """积分长度达到要求，但速度方向精度没达到要求，返回False
        """
        episode_length = 11
        state_list = [self.state_var_type(phi=0., theta=0., psi=0., v=200., mu=21.1, chi=30., p=0., h=0., target_v=200., target_mu=20., target_chi=30.) for i in range(episode_length)]
        res = self.reach_target_termination.get_termination(state_list[-1], state_list=state_list)
        self.assertFalse(res[0])
        self.assertFalse(res[1])

        res2 = self.reach_target_termination.get_termination_and_reward(state_list[-1], state_list=state_list, step_cnt=100)
        self.assertFalse(res2[0])
        self.assertFalse(res2[1])
        self.assertAlmostEqual(res2[2], 0.)

    def test_4(self):
        """积分长度没达到要求，返回False
        """
        episode_length = 9
        state_list = [self.state_var_type(phi=0., theta=0., psi=0., v=200., mu=20., chi=30., p=0., h=0., target_v=200., target_mu=20., target_chi=30.) for i in range(episode_length)]
        res = self.reach_target_termination.get_termination(state_list[-1], state_list=state_list)
        self.assertFalse(res[0])
        self.assertFalse(res[1])

        res2 = self.reach_target_termination.get_termination_and_reward(state_list[-1], state_list=state_list, step_cnt=100)
        self.assertFalse(res2[0])
        self.assertFalse(res2[1])
        self.assertAlmostEqual(res2[2], 0.)


if __name__ == "__main__":
    unittest.main()