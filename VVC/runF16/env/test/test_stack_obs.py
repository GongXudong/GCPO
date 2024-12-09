import unittest
from pathlib import Path
import sys
import numpy as np
from gymnasium import spaces

from stable_baselines3.common.vec_env import VecEnv, VecFrameStack, StackedObservations

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from runF16.env.guidenv_ponential_angle import GuideEnvPonentialAngle
from runF16.env.utils.my_wrappers import ScaledActionWrapper, ScaledObservationWrapper
from runF16.env.utils.sb3_utils import get_vec_env_ponential_angle

OBS_CNT = 8

class StackObsTest(unittest.TestCase):
    """测试StackedObservations，num_envs=1的使用方法

    Args:
        unittest (_type_): _description_
    """
    def setUp(self) -> None:
        super().setUp()

        env = GuideEnvPonentialAngle(env_config={
            "available_targets_dir": PROJECT_ROOT_DIR / "my_logs_for_parallel" / "10hz_10_5_5",
            "available_targets_filename": "res.csv",
            "my_logger": None
        })
        self.env = ScaledActionWrapper(ScaledObservationWrapper(env))

        self.stacked_obs = StackedObservations(
            num_envs=1, 
            n_stack=3, 
            # venv.observation_space, 
            observation_space=spaces.Box(
                low=env.observation_space.low[:OBS_CNT], 
                high=env.observation_space.high[:OBS_CNT]
            )
        )

    def test_1(self):
        origin_state, info = self.env.reset()
        print(np.expand_dims(origin_state[:OBS_CNT], axis=0))
        state = self.stacked_obs.reset(np.expand_dims(origin_state[:OBS_CNT], axis=0))
        print(state)
        for i in range(2):
            state, info = self.stacked_obs.update(np.expand_dims(origin_state[:OBS_CNT], axis=0), [False], [{}])
        print(type(state))
        print(np.squeeze(state, axis=0))

if __name__ == "__main__":
    unittest.main()