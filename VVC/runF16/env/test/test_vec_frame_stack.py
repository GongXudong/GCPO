import unittest
from pathlib import Path
import sys
import numpy as np

from stable_baselines3.common.vec_env import VecEnv, VecFrameStack

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from runF16.env.guidenv_ponential_angle import GuideEnvPonentialAngle
from runF16.env.utils.my_wrappers import ScaledActionWrapper, ScaledObservationWrapper
from runF16.env.utils.sb3_utils import get_vec_env_ponential_angle

STEP_FREQUENCE = 10
GAMMA = 0.999
EXPERT_DATA_DIR = "10hz_10_5_5"
NUM_PROCESS = 2
N_STACK = 3


class SB3VecFrameStackTest(unittest.TestCase):

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
        self.vec_frame_stack_envs = VecFrameStack(
            venv=self.vec_envs,
            n_stack=N_STACK
        )  # obs shape 变成了 (num_process, 11 * n_stack)，观测中开始时的缺省值是0！！！

    def test_obs_shape_1(self):
        """测试观测空间
        """
        states = self.vec_frame_stack_envs.reset()
        print(states.shape)
        self.assertTupleEqual(states.shape, (NUM_PROCESS, 11 * N_STACK))
        # print(states[0])

    def test_obs_1(self):
        """第一回合，前两个位置的观测均为0，第三个位置的观测是系统当前观测
        """
        states = self.vec_frame_stack_envs.reset()
        self.assertTrue(np.allclose(states[0, :22], [0.] * 11 * (N_STACK-1)))

    def test_obs_2(self):
        """第二回合
        """
        states = self.vec_frame_stack_envs.reset()
        acts = [self.vec_frame_stack_envs.action_space.sample() for i in range(NUM_PROCESS)]
        acts = np.array(acts)
        # print("acts: ", acts.shape)
        states2, rewards, dones, infos = self.vec_frame_stack_envs.step(actions=acts)
        self.assertTrue(np.allclose(states2[0, :11], [0.]*11))
        self.assertTrue(np.allclose(states[0, 22:], states2[0, 11:22]))
    
    def test_obs_3(self):
        """第三回合
        """
        states = self.vec_frame_stack_envs.reset()
        acts = [self.vec_frame_stack_envs.action_space.sample() for i in range(NUM_PROCESS)]
        acts = np.array(acts)
        # print("acts: ", acts.shape)
        states2, rewards, dones, infos = self.vec_frame_stack_envs.step(actions=acts)
        states3, rewards, dones, infos = self.vec_frame_stack_envs.step(actions=acts)
        self.assertTrue(np.allclose(states2[0, 22:], states3[0, 11:22]))
        self.assertTrue(np.allclose(states2[0, 11:22], states3[0, :11]))
        self.assertTrue(np.allclose(states[0, 22:], states3[0, :11]))
        

if __name__ == "__main__":
    unittest.main()