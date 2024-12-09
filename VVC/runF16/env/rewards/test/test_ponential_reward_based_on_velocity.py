import unittest
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from runF16.env.rewards.ponential_reward_based_on_velocity import PonentialRewardBasedOnVelocity
from runF16.env.guidenv import GuideEnv

class PonentialRewardBasedOnAngleTest(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()

        state_min = GuideEnv.get_state_lower_bounds()
        state_max = GuideEnv.get_state_higher_bounds()
        self.state_var_type = GuideEnv.get_state_vars()
        self.ponential_reward = PonentialRewardBasedOnVelocity(
            b=1.,
            # gamma=0.9999,
            gamma=0.99,
            scale=100.
        )

    def test_1(self):
        state = self.state_var_type(phi=0., theta=0., psi=0., v=170, mu=10., chi=20., p=0., h=0., target_v=200., target_mu=20., target_chi=30.)
        next_state = self.state_var_type(phi=0., theta=0., psi=0., v=180, mu=15., chi=25., p=0., h=0., target_v=200., target_mu=20., target_chi=30.)
        reward = self.ponential_reward.get_reward(state=state, next_state=next_state, done=False)

        reward_test = - self.ponential_reward.gamma * (next_state.target_v - next_state.v) / self.ponential_reward.scale + (state.target_v - state.v) / self.ponential_reward.scale

        self.assertAlmostEqual(reward, reward_test) 


if __name__ == "__main__":
    unittest.main()