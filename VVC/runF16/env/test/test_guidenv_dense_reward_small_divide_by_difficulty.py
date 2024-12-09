import unittest
from pathlib import Path
import sys

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from runF16.env.guidenv_dense_reward_small_divide_by_difficulty import GuideEnvDenseRewardSmallDivideByDifficulty

class GuideEnvDenseRewardSmallTest(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.env = GuideEnvDenseRewardSmallDivideByDifficulty(env_config={
            "available_targets_dir": PROJECT_ROOT_DIR / "runF16" / "algorithm" / "data_sb3" / "label_difficulty",
            "available_targets_filename": "res_with_difficulty.csv",
            "use_fixed_target": False, 
            "difficulty": 0 ,
            "task_set_capacity": 100,
        })

    def test_target_1(self):
        print(self.env.available_targets)

if __name__ == "__main__":
    unittest.main()