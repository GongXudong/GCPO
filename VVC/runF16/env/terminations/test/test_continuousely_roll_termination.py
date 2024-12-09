import unittest
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from runF16.env.terminations.continuousely_roll_termination import ContinuouselyRollTermination
from runF16.env.guidenv import GuideEnv


class ContinuouselyRollTerminationTest(unittest.TestCase):

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
        self.continuousely_roll_termination = ContinuouselyRollTermination(
            continuousely_roll_threshold=720., 
            termination_reward=-1.,
            is_termination_reward_based_on_steps_left=False,
            env_config=env_config
        )

        self.state_var_type = GuideEnv.get_state_vars()

    def test_1(self):
        phi_arr = list(range(0, 180, 10)) + list(range(-180, 0, 10)) + list(range(0, 180, 10)) + list(range(-180, 0, 10)) + list(range(0, 180, 10))
        print(len(phi_arr))
        state_list = [self.state_var_type(phi=phi, theta=0., psi=0., v=200., mu=0., chi=30., p=0., h=0., target_v=200., target_mu=20., target_chi=30.) for phi in phi_arr]

        self.continuousely_roll_termination.reset()
        for i in range(len(state_list) - 1):
            state = state_list[i]
            next_state = state_list[i+1]
            res = self.continuousely_roll_termination.get_termination(state=state, next_state=next_state)
            if res[0]:
                break
        
        self.assertTrue(res[0])
        self.assertFalse(res[1])
        
        self.continuousely_roll_termination.reset()
        for i in range(len(state_list) - 1):
            state = state_list[i]
            next_state = state_list[i+1]
            res2 = self.continuousely_roll_termination.get_termination_and_reward(state=state, next_state=next_state, step_cnt=100)
            if res2[0]:
                break
        
        self.assertTrue(res2[0])
        self.assertFalse(res2[1])
        self.assertAlmostEqual(res2[2], -1.)
        

    def test_2(self):
        phi_arr = list(range(0, 180, 10)) + list(range(-180, 0, 10)) + list(range(0, 180, 10)) + [160] + list(range(-180, 0, 10)) + list(range(0, 180, 10))

        # for i in phi_arr:
        #     print(i)
        
        state_list = [self.state_var_type(phi=phi, theta=0., psi=0., v=200., mu=0., chi=30., p=0., h=0., target_v=200., target_mu=20., target_chi=30.) for phi in phi_arr]

        self.continuousely_roll_termination.reset()
        for i in range(len(state_list) - 1):
            state = state_list[i]
            next_state = state_list[i+1]
            res = self.continuousely_roll_termination.get_termination(state=state, next_state=next_state)
            if res[0]:
                break
        
        self.assertFalse(res[0])
        self.assertFalse(res[1])
        
        self.continuousely_roll_termination.reset()
        for i in range(len(state_list) - 1):
            state = state_list[i]
            next_state = state_list[i+1]
            res2 = self.continuousely_roll_termination.get_termination_and_reward(state=state, next_state=next_state, step_cnt=100)
            if res2[0]:
                break
        
        self.assertFalse(res2[0])
        self.assertFalse(res2[1])
        self.assertAlmostEqual(res2[2], 0.)


if __name__ == "__main__":
    unittest.main()