import unittest
import numpy as np
from pathlib import Path
import sys
from functools import partial
import gymnasium as gym
from collections import namedtuple

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from utils.non_markovian_reward_wrapper import NMRWrapper


class NMRWrapperTest(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()

        tmp_env = gym.make("PointMaze_Large_Diverse_G-v3")
        self.env = NMRWrapper(tmp_env)
    
        # 直接在NMR环境上测试训练的策略

        # 1.在bc上测试
        
        # 不使用NMRWrapper时
        # python evaluate/evaluate_policy.py --config-file-name configs/pointmaze-diverse-g/seed1/sparse_256_256_lambda_1e-1.json --test-algo bc --env-id PointMaze_Large-v3 --eval-process-num 16 --eval-episode-num 100 --seed 5
        # 成功率：0.82，轨迹平均长度：449.11

        # nmr_length=1
        # python evaluate/evaluate_policy.py --config-file-name configs/pointmaze-diverse-g/seed1/sparse_256_256_lambda_1e-1.json --test-algo bc --env-id PointMaze_Large-v3 --eval-process-num 16 --eval-episode-num 100 --seed 5 --wrap-env-with-nmr --nmr-length=1
        # 成功率：0.82，轨迹平均长度：449.11

        # nmr_length=20
        # python evaluate/evaluate_policy.py --config-file-name configs/pointmaze-diverse-g/seed1/sparse_256_256_lambda_1e-1.json --test-algo bc --env-id PointMaze_Large-v3 --eval-process-num 16 --eval-episode-num 100 --seed 5 --wrap-env-with-nmr --nmr-length=20
        # 成功率：0.65，轨迹平均长度：534.68

        # nmr_length=30
        # python evaluate/evaluate_policy.py --config-file-name configs/pointmaze-diverse-g/seed1/sparse_256_256_lambda_1e-1.json --test-algo bc --env-id PointMaze_Large-v3 --eval-process-num 16 --eval-episode-num 100 --seed 5 --wrap-env-with-nmr --nmr-length=30
        # 成功率：0.35，轨迹平均长度：687.66
        
        # nmr_length=50
        # python evaluate/evaluate_policy.py --config-file-name configs/pointmaze-diverse-g/seed1/sparse_256_256_lambda_1e-1.json --test-algo bc --env-id PointMaze_Large-v3 --eval-process-num 16 --eval-episode-num 100 --seed 5 --wrap-env-with-nmr --nmr-length=50
        # 成功率：0.05，轨迹平均长度：782.21

        # 2.在rl_bc上测试
        
        # nmr_length=20
        # python evaluate/evaluate_policy.py --config-file-name configs/pointmaze-diverse-g/seed1/sparse_256_256_lambda_1e-1.json --test-algo rl_bc --env-id PointMaze_Large-v3 --eval-process-num 16 --eval-episode-num 100 --seed 5 --wrap-env-with-nmr --nmr-length=20
        # 成功率：0.48，轨迹平均长度：582.01

        # nmr_length=30
        # python evaluate/evaluate_policy.py --config-file-name configs/pointmaze-diverse-g/seed1/sparse_256_256_lambda_1e-1.json --test-algo rl_bc --env-id PointMaze_Large-v3 --eval-process-num 16 --eval-episode-num 100 --seed 5 --wrap-env-with-nmr --nmr-length=30
        # 成功率：0.18，轨迹平均长度：733.8
        
        # nmr_length=50
        # python evaluate/evaluate_policy.py --config-file-name configs/pointmaze-diverse-g/seed1/sparse_256_256_lambda_1e-1.json --test-algo rl_bc --env-id PointMaze_Large-v3 --eval-process-num 16 --eval-episode-num 100 --seed 5 --wrap-env-with-nmr --nmr-length=50
        # 成功率：0.04，轨迹平均长度：787.78
        
        # nmr_length越长，policy成功率越低，间接测试NMRWrapper的正确性。
        # 且nmr_length设置为1时，结果与不使用NMRWrapper相同，简介说明NMRWrapper的正确性。

if __name__ == "__main__":
    unittest.main()