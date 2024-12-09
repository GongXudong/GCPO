import unittest
import numpy as np
from pathlib import Path
import sys
from functools import partial
from gymnasium import spaces

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from runF16.env.guidenv import GuideEnv
from runF16.env.utils.my_wrappers import ScaledObservationWrapper, ScaledActionWrapper


my_all_close = partial(np.allclose, atol=1.e-8)


class EnvTest(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.env = GuideEnv(env_config={})
        self.wrappered_env = ScaledActionWrapper(ScaledObservationWrapper(self.env))
    
    def test_env_observation_space_1(self):
        self.assertEqual(
            self.env.observation_space, 
            spaces.Box(
                low=np.array(GuideEnv.get_state_lower_bounds(), dtype=np.float32), 
                high=np.array(GuideEnv.get_state_higher_bounds(), dtype=np.float32), 
            )
        )
    
    def test_env_observation_space_2(self):
        self.assertEqual(
            self.wrappered_env.unwrapped.observation_space,
            spaces.Box(
                low=np.array(GuideEnv.get_state_lower_bounds(), dtype=np.float32), 
                high=np.array(GuideEnv.get_state_higher_bounds(), dtype=np.float32), 
            )
        )
    
    def test_env_action_space_1(self):
        self.assertEqual(
            self.env.action_space,
            spaces.Box(
                low=np.array(GuideEnv.get_action_lower_bounds(), dtype=np.float32),
                high=np.array(GuideEnv.get_action_higher_bounds(), dtype=np.float32)
            )
        )
    
    def test_env_action_space_2(self):
        self.assertEqual(
            self.wrappered_env.unwrapped.action_space,
            spaces.Box(
                low=np.array(GuideEnv.get_action_lower_bounds(), dtype=np.float32),
                high=np.array(GuideEnv.get_action_higher_bounds(), dtype=np.float32)
            )
        )

    def test_wrappered_env_observation_space(self):
        # print(self.wrappered_env.observation_space)
        self.assertEqual(
            self.wrappered_env.observation_space, 
            spaces.Box(low=0., high=1., shape=(11,))
        )

    def test_wrappered_env_action_space(self):
        # print(self.wrappered_env.action_space)
        self.assertEqual(
            self.wrappered_env.action_space, 
            spaces.Box(low=0., high=1., shape=(3,))
        )

    def test_scale_state_1(self):
        state_var_type = GuideEnv.get_state_vars()
        state = state_var_type(phi=0., theta=0., psi=0., v=500., mu=0., chi=0., p=0, h=10000., target_v=500., target_mu=0., target_chi=0.)
        scaled_state = self.wrappered_env.scale_state(np.array(state))

        ref = np.array([0.5] * 11).reshape((1, -1))
        self.assertTrue(my_all_close(scaled_state, ref, atol=1.e-8))

    def test_inverse_scale_action_1(self):
        action_var_type = GuideEnv.get_action_vars()
        action = action_var_type(p=0.5, nz=0.5, pla=0.5)
        inverse_scaled_action = self.wrappered_env.inverse_scale_action(np.array(action))
        ref = np.array([0., 2.5, 0.5]).reshape((1, -1))
        self.assertTrue(my_all_close(inverse_scaled_action, ref))


if __name__ == "__main__":
    unittest.main()