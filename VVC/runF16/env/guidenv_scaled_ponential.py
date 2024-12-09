from pathlib import Path
import sys

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from runF16.env.guidenv import GuideEnv
from runF16.env.rewards.ponential_reward import ScaledPonentialReward2


class GuideEnvScaledPonential(GuideEnv):

    def __init__(self, env_config: dict) -> None:
        self.scale_ponential_reward_coef = env_config.get('scale_ponential_reward_coef', 100.)
        super().__init__(env_config)
    
    def _prep_reward_funcs(self, env_config):
        self.reward_funcs = [
            ScaledPonentialReward2(scale_coef=self.scale_ponential_reward_coef, gamma=env_config.get("gamma"))
        ]