import unittest
from pathlib import Path
import sys
import numpy as np
from gymnasium import spaces

from stable_baselines3.common.vec_env import VecEnv

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from runF16.env.guidenv_ponential_angle import GuideEnvPonentialAngle
from runF16.env.utils.my_wrappers import ScaledActionWrapper, ScaledObservationWrapper
from runF16.env.utils.sb3_utils import get_vec_env_ponential_angle
from runF16.env.utils.my_vec_frame_stack_env import MyVecFrameStack

STEP_FREQUENCE = 10
GAMMA = 0.999
EXPERT_DATA_DIR = "10hz_10_5_5"
NUM_PROCESS = 2
N_STACK = 3

class MyVecFrameStackEnvTest(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()

        self.vec_envs = get_vec_env_ponential_angle(
            num_process=NUM_PROCESS, 
            logger=None, 
            step_frequence=STEP_FREQUENCE,
            max_simulate_time=120,
            gamma=GAMMA,
            available_targets_dir=PROJECT_ROOT_DIR / "my_logs_for_parallel" / EXPERT_DATA_DIR,
            sample_target_noise_std=[0., 0., 0.],
        )
        self.vec_frame_stack_envs = MyVecFrameStack(
            venv=self.vec_envs,
            n_stack=N_STACK,
            obs_cnt=8,
            target_cnt=3
        )

    def test_obs_space_1(self):
        """验证观测空间
        """
        s1 = self.vec_frame_stack_envs.observation_space
        self.assertEqual(s1, spaces.Box(low=0., high=1., shape=(8 * N_STACK + 3,)))

    def test_obs_1(self):
        """验证初始化环境时，stack的观测全部为初始观测。
        """
        state = self.vec_frame_stack_envs.reset()
        self.assertTrue(np.allclose(state[:, :8], state[:, 8:16]))
        self.assertTrue(np.allclose(state[:, 8:16], state[:, 16:24]))
    
    def test_obs_2(self):
        
        state = self.vec_frame_stack_envs.reset()
        acts = [self.vec_frame_stack_envs.action_space.sample() for i in range(NUM_PROCESS)]
        
        state2, rewards, dones, infos = self.vec_frame_stack_envs.step(actions=acts)
        
        self.assertTrue(np.allclose(state[:, 16:24], state2[:, 8:16]))

    def test_obs_3(self):
        state = self.vec_frame_stack_envs.reset()
        acts = [self.vec_frame_stack_envs.action_space.sample() for i in range(NUM_PROCESS)]
        
        state2, rewards, dones, infos = self.vec_frame_stack_envs.step(actions=acts)
        state3, rewards, dones, infos = self.vec_frame_stack_envs.step(actions=acts)

        self.assertTrue(np.allclose(state[:, 16:24], state2[:, 8:16]))
        self.assertTrue(np.allclose(state2[:, 16:24], state3[:, 8:16]))
        self.assertTrue(np.allclose(state2[:, 8:16], state3[:, :8]))

if __name__ == "__main__":
    unittest.main()