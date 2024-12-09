import unittest
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from runF16.env.terminations.timeout_termination import TimeoutTermination
from runF16.env.guidenv import GuideEnv


class TimeoutTerminationTest(unittest.TestCase):

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
        self.timeout_termination = TimeoutTermination(
            termination_reward=-1.,
            env_config=env_config
        )
        self.state_var_type = GuideEnv.get_state_vars()
    
    def test_1(self):
        tested_step_cnt = 410
        state = self.state_var_type(phi=0., theta=0., psi=0., v=200., mu=20., chi=30., p=0., h=0., target_v=200., target_mu=20., target_chi=30.)
        res = self.timeout_termination.get_termination(state=state, step_cnt=tested_step_cnt)
        self.assertTrue(res[0])
        self.assertTrue(res[1])

        res2 = self.timeout_termination.get_termination_and_reward(state=state, step_cnt=tested_step_cnt)
        self.assertTrue(res2[0])
        self.assertTrue(res2[1])
        self.assertAlmostEqual(res2[2], -1.)
    
    def test_2(self):
        tested_step_cnt = 500
        state = self.state_var_type(phi=0., theta=0., psi=0., v=200., mu=20., chi=30., p=0., h=0., target_v=200., target_mu=20., target_chi=30.)
        res = self.timeout_termination.get_termination(state=state, step_cnt=tested_step_cnt)
        self.assertTrue(res[0])
        self.assertTrue(res[1])

        res2 = self.timeout_termination.get_termination_and_reward(state=state, step_cnt=tested_step_cnt)
        self.assertTrue(res2[0])
        self.assertTrue(res2[1])
        self.assertAlmostEqual(res2[2], -1.)
    
    def test_3(self):
        tested_step_cnt = 398
        state = self.state_var_type(phi=0., theta=0., psi=0., v=200., mu=20., chi=30., p=0., h=0., target_v=200., target_mu=20., target_chi=30.)
        res = self.timeout_termination.get_termination(state=state, step_cnt=tested_step_cnt)
        self.assertFalse(res[0])
        self.assertFalse(res[1])

        res2 = self.timeout_termination.get_termination_and_reward(state=state, step_cnt=tested_step_cnt)
        self.assertFalse(res2[0])
        self.assertFalse(res2[1])
        self.assertAlmostEqual(res2[2], 0.)

if __name__ == "__main__":
    unittest.main()