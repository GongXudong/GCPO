import unittest
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from runF16.env.terminations.extreme_state_termination import ExtremeStateTermination
from runF16.env.guidenv import GuideEnv

class CrashTerminationTest(unittest.TestCase):

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
        self.extreme_state_termination = ExtremeStateTermination(
            v_max=1000., 
            p_max=500.,
            termination_reward=-1.,
            is_termination_reward_based_on_steps_left=False,
            env_config=env_config
        )
        self.state_var_type = GuideEnv.get_state_vars()

    def test_1(self):
        tested_v = 1100.
        state = self.state_var_type(phi=0., theta=0., psi=0., v=tested_v, mu=20., chi=30., p=0., h=5000., target_v=200., target_mu=20., target_chi=30.)
        res = self.extreme_state_termination.get_termination(state=state)
        self.assertTrue(res[0])
        self.assertTrue(res[1])

        res2 = self.extreme_state_termination.get_termination_and_reward(state=state, step_cnt=100)
        self.assertTrue(res2[0])
        self.assertTrue(res2[1])
        self.assertAlmostEqual(res2[2], -1.)

    def test_2(self):
        tested_v = 999.
        state = self.state_var_type(phi=0., theta=0., psi=0., v=tested_v, mu=20., chi=30., p=0., h=5000., target_v=200., target_mu=20., target_chi=30.)
        res = self.extreme_state_termination.get_termination(state=state)
        self.assertFalse(res[0])
        self.assertFalse(res[1])

        res2 = self.extreme_state_termination.get_termination_and_reward(state=state, step_cnt=100)
        self.assertFalse(res2[0])
        self.assertFalse(res2[1])
        self.assertAlmostEqual(res2[2], 0.)

    def test_3(self):
        tested_p = -410.
        state = self.state_var_type(phi=0., theta=0., psi=0., v=200., mu=20., chi=30., p=tested_p, h=5000., target_v=200., target_mu=20., target_chi=30.)
        res = self.extreme_state_termination.get_termination(state=state)
        self.assertFalse(res[0])
        self.assertFalse(res[1])

        res2 = self.extreme_state_termination.get_termination_and_reward(state=state, step_cnt=100)
        self.assertFalse(res2[0])
        self.assertFalse(res2[1])
        self.assertAlmostEqual(res2[2], 0.)
    
    def test_4(self):
        tested_p = -510.
        state = self.state_var_type(phi=0., theta=0., psi=0., v=200., mu=20., chi=30., p=tested_p, h=5000., target_v=200., target_mu=20., target_chi=30.)
        res = self.extreme_state_termination.get_termination(state=state)
        self.assertTrue(res[0])
        self.assertTrue(res[1])

        res2 = self.extreme_state_termination.get_termination_and_reward(state=state, step_cnt=100)
        self.assertTrue(res2[0])
        self.assertTrue(res2[1])
        self.assertAlmostEqual(res2[2], -1.)
    
    def test_5(self):
        tested_p = 510.
        state = self.state_var_type(phi=0., theta=0., psi=0., v=200., mu=20., chi=30., p=tested_p, h=5000., target_v=200., target_mu=20., target_chi=30.)
        res = self.extreme_state_termination.get_termination(state=state)
        self.assertTrue(res[0])
        self.assertTrue(res[1])

        res2 = self.extreme_state_termination.get_termination_and_reward(state=state, step_cnt=100)
        self.assertTrue(res2[0])
        self.assertTrue(res2[1])
        self.assertAlmostEqual(res2[2], -1.)


if __name__ == "__main__":
    unittest.main()