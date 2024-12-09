import unittest
from pathlib import Path
import sys

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from runF16.env.utils.sb3_utils import get_vec_env


class TestSB3VecEnv(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()

        self.num_process = 10
        self.env = get_vec_env(num_process=self.num_process)

    def test_1(self):
        print(self.env.observation_space)
        print(self.env.action_space)

    def test_2(self):
        states = self.env.reset()
        # print(states.shape)
        self.assertTrue(states.shape, (self.num_process, ) + self.env.observation_space.shape)
    
    def test_3(self):
        states = self.env.reset()
        actions = [self.env.action_space.sample() for i in range(self.num_process)]
        states, rewards, terminates, infos = self.env.step(actions=actions)
        self.assertTrue(states.shape, (self.num_process, ) + self.env.observation_space.shape)
        self.assertTrue(rewards.shape, (self.num_process, ))
        self.assertTrue(terminates.shape, (self.num_process, ))
        self.assertTrue(isinstance(infos, tuple))


if __name__ == "__main__":
    unittest.main()