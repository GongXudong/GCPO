from pathlib import Path
import sys

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from runF16.env.guidenv_dense_reward import GuideEnvDenseReward

class GuideEnvDenseRewardSmallDivideByDifficulty(GuideEnvDenseReward):
    """将所有目标按难度从小到大排序，按{task_set_capacity}分成若干份，使用第{difficulty}份作为目标任务集训练
    """
    def __init__(self, env_config: dict) -> None:
        super().__init__(env_config)
        self.available_targets = self.available_targets[self.available_targets.length > 0]
        self.difficulty = env_config.get("difficulty", 0)
        self.task_set_capacity = env_config.get("task_set_capacity", 100)
        
        total_capacity = len(self.available_targets)
        begin_index = min(self.difficulty * self.task_set_capacity, total_capacity)
        end_index = min((self.difficulty + 1) * self.task_set_capacity, total_capacity)

        self.available_targets = self.available_targets.sort_values(by=["error_value", "length", "chi"])
        self.available_targets = self.available_targets[begin_index:end_index]
        
        print(f"sorted by difficulty value, select from {begin_index} to {end_index}, difficulty from {self.available_targets.iloc[0].error_value} to {self.available_targets.iloc[-1].error_value}.")
        