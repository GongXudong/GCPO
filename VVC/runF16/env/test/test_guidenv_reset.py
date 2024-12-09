import unittest
import numpy as np
from pathlib import Path
import sys
from functools import partial
from gymnasium import spaces
from collections import namedtuple

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from runF16.env.guidenv import GuideEnv
from runF16.env.utils.my_wrappers import ScaledObservationWrapper, ScaledActionWrapper


# def output_state(state: namedtuple):
#     tmp = state._asdict()
#     for k in tmp.keys():
#         print(f"{k}: {tmp[k]}")


class EnvResetTest(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
    
    def test_1(self):
        v0 = 200.
        origin_env = GuideEnv(env_config={"use_fixed_target": True, "target_v": 200., "target_mu": 20., "target_chi": 30., "h0": 5000., "v0": v0})
        env = ScaledActionWrapper(ScaledObservationWrapper(origin_env))

        state, info = env.reset()
        #                  phi, theta,             psi, v,   mu,  chi, p,   h,    target_v, target_mu, target_chi
        # 原env的初始值      0,   2.613220523094039, 0,   200, 0,   0,   0,   5000, 200,      20,        30
        # wrap后env的初始值  0.5, 0.51451789,        0.5, 0.2, 0.5, 0.5, 0.5, 0.25, 0.2,      11/18,     21/36 
        # print(state)
        # print("initial theta: ", origin_env.f16model.getPlaneState()["theta"])
        # print(info)
        self.assertTrue(np.allclose(state, [0.5, 0.51451789, 0.5, 0.2, 0.5, 0.5, 0.5, 0.25, 0.2, 11./18., 21./36.]))
    
    def test_2(self):
        v0 = 300.
        origin_env = GuideEnv(env_config={"use_fixed_target": True, "target_v": 200., "target_mu": 20., "target_chi": 30., "h0": 5000., "v0": v0})
        env = ScaledActionWrapper(ScaledObservationWrapper(origin_env))
        state, info = env.reset()
        # print("initial theta: ", origin_env.f16model.getPlaneState()["theta"])
        # print(info)
        # print(state)
        self.assertTrue(np.allclose(state, [0.5, 0.50223776, 0.5, 0.3, 0.5, 0.5, 0.5, 0.25, 0.2, 11./18., 21./36.]))

    def test_3(self):
        """测试运行几拍后reset
        """
        v0 = 200.
        origin_env = GuideEnv(env_config={"use_fixed_target": True, "target_v": 200., "target_mu": 20., "target_chi": 30., "h0": 5000., "v0": v0})
        env = ScaledActionWrapper(ScaledObservationWrapper(origin_env))

        state, info = env.reset()

        for i in range(100):
            action = env.action_space.sample()
            env.step(action)
        
        state, info = env.reset()
        self.assertTrue(np.allclose(state, [0.5, 0.51451789, 0.5, 0.2, 0.5, 0.5, 0.5, 0.25, 0.2, 11./18., 21./36.]))
        

if __name__ == "__main__":
    unittest.main()