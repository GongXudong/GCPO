import unittest
from pathlib import Path
import sys

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from runF16.env.guidenv_dense_reward_filter_by_difficulty import GuideEnvDenseRewardFilteredByDifficulty

class GuideEnvDenseRewardFilteredByDifficultyTest(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()

    def test_difficulty_0_num(self):
        env = GuideEnvDenseRewardFilteredByDifficulty(env_config={"difficulty": -1})
        print(env.available_targets)

    def test_difficulty_1_num(self):
        env = GuideEnvDenseRewardFilteredByDifficulty(env_config={"difficulty": 1})
        print(env.available_targets)
    
    def test_difficulty_2_num(self):
        env = GuideEnvDenseRewardFilteredByDifficulty(env_config={"difficulty": 2})
        print(env.available_targets)
    
    def test_difficulty_3_num(self):
        env = GuideEnvDenseRewardFilteredByDifficulty(env_config={"difficulty": 3})
        print(env.available_targets)
    
    def test_difficulty_4_num(self):
        env = GuideEnvDenseRewardFilteredByDifficulty(env_config={"difficulty": 4})
        print(env.available_targets)

    def test_difficulty_5_num(self):
        env = GuideEnvDenseRewardFilteredByDifficulty(env_config={"difficulty": 5})
        print(env.available_targets)

if __name__ == "__main__":
    unittest.main()