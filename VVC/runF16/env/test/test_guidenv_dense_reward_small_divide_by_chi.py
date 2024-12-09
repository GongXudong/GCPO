import unittest
from pathlib import Path
import sys

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from runF16.env.guidenv_dense_reward_small_divide_by_chi import GuideEnvDenseRewardSmallDivideByChi

class GuideEnvDenseRewardSmallTest(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.env = GuideEnvDenseRewardSmallDivideByChi(env_config={"use_fixed_target": False, "target_chi": 170})

    def test_target_chi_num(self):
        print(self.env.available_targets)

if __name__ == "__main__":
    unittest.main()