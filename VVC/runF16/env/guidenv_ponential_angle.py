from pathlib import Path
import sys

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from runF16.env.guidenv import GuideEnv
from runF16.env.rewards.ponential_reward_based_on_angle import PonentialRewardBasedOnAngle


class GuideEnvPonentialAngle(GuideEnv):

    def __init__(self, env_config: dict) -> None:
        super().__init__(env_config)
    
    def _prep_reward_funcs(self, env_config):
        self.reward_funcs = [
            PonentialRewardBasedOnAngle(b=env_config.get('reward_scale_factor_b', 1.), gamma=env_config.get("gamma"))
        ]