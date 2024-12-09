from pathlib import Path
import sys

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from runF16.env.guidenv_dense_reward import GuideEnvDenseReward

class GuideEnvDenseRewardSmallDivideByChi(GuideEnvDenseReward):

    def __init__(self, env_config: dict) -> None:
        super().__init__(env_config)
        self.available_targets = self.available_targets[(self.available_targets.chi == self.target_chi) & (self.available_targets.length > 0)]
        print(f"target_chi={self.target_chi}, num={len(self.available_targets)}")