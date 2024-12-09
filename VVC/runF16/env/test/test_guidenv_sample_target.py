import unittest
import numpy as np
from pathlib import Path
import logging
import sys
from functools import partial
from gymnasium import spaces

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from runF16.env.guidenv import GuideEnv
from runF16.env.utils.my_wrappers import ScaledObservationWrapper, ScaledActionWrapper
from runF16.utils_my.my_log import get_logger


my_logger = get_logger(log_file_name="my_rl_logs/logs.log")


class EnvStepTest(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        # 本轮测试不要修改初始条件！！！（需要修改的话，定义新的测试）
        self.origin_env = GuideEnv(env_config={"h0": 5000., "v0": 200, "my_logger": my_logger})
        self.obs_wrappered_env = ScaledObservationWrapper(self.origin_env)
        self.env = ScaledActionWrapper(self.obs_wrappered_env)

    def test_1(self):
        for i in range(10):
            self.env.reset()

if __name__ == "__main__":
    unittest.main()