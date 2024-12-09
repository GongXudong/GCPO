from pathlib import Path
import sys

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from runF16.env.guidenv import GuideEnv
from runF16.env.rewards.dense_reward import DenseReward
from runF16.env.rewards.dense_reward_based_on_angle_and_velocity import DenseRewardBasedOnAngleAndVelocity

from runF16.env.terminations.reach_target_termination import ReachTargetTermination
from runF16.env.terminations.reach_target_termination2 import ReachTargetTermination2
from runF16.env.terminations.crash_termination import CrashTermination
from runF16.env.terminations.timeout_termination import TimeoutTermination
from runF16.env.terminations.continuousely_move_away_termination import ContinuouselyMoveAwayTermination
from runF16.env.terminations.continuousely_move_away_termination2 import ContinuouselyMoveAwayTermination2
from runF16.env.terminations.continuousely_roll_termination import ContinuouselyRollTermination
from runF16.env.terminations.negative_overload_and_big_phi_termination import NegativeOverloadAndBigPhiTermination
from runF16.env.terminations.extreme_state_termination import ExtremeStateTermination

class GuideEnvDenseReward(GuideEnv):

    def __init__(self, env_config: dict) -> None:
        super().__init__(env_config)
    
    def _prep_reward_funcs(self, env_config):
        self.reward_funcs = [
            # DenseReward(b=env_config.get('reward_scale_factor_b', 1.))
            DenseRewardBasedOnAngleAndVelocity(b=env_config.get('reward_scale_factor_b', 1.), angle_scale=180., velocity_scale=100.)
        ]
    
    def _prep_termination_funcs(self, env_config):
        bonus = 0.
        self.termination_funcs = [
            ReachTargetTermination2(integral_time_length=1, v_threshold=10., angle_threshold=1., termination_reward=bonus, env_config=env_config, my_logger=self.logger),
            CrashTermination(h0=0., is_termination_reward_based_on_steps_left=True, env_config=env_config, my_logger=self.logger),
            ExtremeStateTermination(v_max=400., p_max=300., is_termination_reward_based_on_steps_left=True, env_config=env_config, my_logger=self.logger),
            TimeoutTermination(env_config=env_config, my_logger=self.logger),
            ContinuouselyMoveAwayTermination(time_window=2, is_termination_reward_based_on_steps_left=True, env_config=env_config, my_logger=self.logger),
            # ContinuouselyMoveAwayTermination2(time_window=2, ignore_velocity_vector_angle_error=1., is_termination_reward_based_on_steps_left=True, env_config=env_config, my_logger=self.logger),
            ContinuouselyRollTermination(continuousely_roll_threshold=720, is_termination_reward_based_on_steps_left=True, env_config=env_config, my_logger=self.logger),
            NegativeOverloadAndBigPhiTermination(time_window=2, negative_overload_threshold=0., big_phi_threshold=60., is_termination_reward_based_on_steps_left=True, env_config=env_config, my_logger=self.logger),
        ]
