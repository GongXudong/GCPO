from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from runF16.env.guidenv_dense_reward import GuideEnvDenseReward

class GuideEnvDenseRewardFilteredByDifficulty(GuideEnvDenseReward):

    def __init__(self, env_config: dict) -> None:
        super().__init__(env_config)
        # new_target_file = PROJECT_ROOT_DIR / "runF16" / "algorithm" / "data_sb3" / "label_difficulty" / "res_with_difficulty.csv"
        
        self.task_difficulty = env_config.get("difficulty", -1)

        if 'difficulty_calc_by_error' not in self.available_targets.columns:
            new_target_file = PROJECT_ROOT_DIR / "runF16" / "algorithm" / "data_sb3" / "label_difficulty" / "res_with_difficulty_all.csv"
            self.available_targets = pd.read_csv(new_target_file)
        else:
            self.available_targets = self.available_targets = pd.read_csv((self.available_targets_dir / self.available_targets_filename).absolute())

        if self.task_difficulty >= 0:
            self.available_targets = self.available_targets[(self.available_targets.difficulty_calc_by_error == self.task_difficulty) & (self.available_targets.length > 0)]
        elif self.task_difficulty == -1:
            # 目标是专家能完成的轨迹
            self.available_targets = self.available_targets[self.available_targets.length > 0]
        elif self.task_difficulty == -2:
            # 目标是所有枚举出的轨迹
            pass
        elif self.task_difficulty == -3:
            # 目标是所有专家不能完成的轨迹
            self.available_targets = self.available_targets[self.available_targets.length == 0]
        else:
            print(f"task_difficulty的取值只能是[-3, -2, -1, 1, 2, 3, 4, 5]")
        print(f"difficulty={self.task_difficulty}, num={len(self.available_targets)}")