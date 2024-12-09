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
        self.origin_env = GuideEnv(env_config={"use_fixed_target": True, "target_v": 200., "target_mu": 20., "target_chi": 30., "h0": 5000., "v0": 200, "my_logger": my_logger})  
        self.obs_wrappered_env = ScaledObservationWrapper(self.origin_env)
        self.env = ScaledActionWrapper(self.obs_wrappered_env)

    def test_1(self):
        state, info = self.env.reset()

        for i in range(10):
            action = self.env.action_space.sample()
            state, reward, terminated, truncated, info = self.env.step(action)
            # print(f"step: {i}, info: {info}")
    
    def test_2(self):
        state, info = self.env.reset()
        while True:
            action = self.env.action_space.sample()
            state, reward, terminated, truncated, info = self.env.step(action)
            if terminated:
                break
        # print(info)

    def test_3(self):
        """测试连续滚转，还没滚转到720，就触发了continuousely_move_away_termination
        """
        state, info = self.env.reset()

        # action = [5./6., 0.25, 0.,]  # 相当于p=120
        action = np.array([120., -4 + 13./4, 0.])
        action = self.env.scale_action(action_var=action)
        self.assertTrue(np.allclose(action, np.array([5./6., 0.25, 0.])))

        while True:
            state, reward, terminated, truncated, info = self.env.step(action)
            # print(state[0], self.origin_env.termination_funcs[4].accumulate_roll)
            if terminated:
                break
        # print(info)

    def test_4(self):
        """测试连续滚转
        """
        state, info = self.env.reset()

        # action = [5./6., 0., 0.,]  # 相当于p=200， nz=-4
        action = np.array([120., -4., 0.])
        action = self.env.scale_action(action_var=action)
        self.assertTrue(np.allclose(action, np.array([5./6., 0., 0.])))

        while True:
            state, reward, terminated, truncated, info = self.env.step(action)
            # print(state[0], self.origin_env.termination_funcs[4].accumulate_roll)
            if terminated:
                break
        # print(info)

    def test_5(self):
        """测试负过载，2秒钟就触发了continuousely_move_away_termination
        """
        state, info = self.env.reset()

        action = np.array([0, -4., 0.,])  # p=0, nz=-4
        action = self.env.scale_action(action_var=action)
        self.assertTrue(np.allclose(action, np.array([0.5, 0., 0.])))
        
        while True:
            state, reward, terminated, truncated, info = self.env.step(action)
            state = self.obs_wrappered_env.inverse_scale_state(state_var=state)
            # print(f"mu: {state[4]}, chi: {state[5]}")
            if terminated:
                break
        # print(info)
    
    def test_6(self):
        """测试负过载，2秒钟就触发了continuousely_move_away_termination
        """
        state, info = self.env.reset()

        action = np.array([-50, -4., 0.,])  # p=50, nz=-4
        action = self.env.scale_action(action_var=action)
        print(action)
        self.assertTrue(np.allclose(action, np.array([0.361111111, 0., 0.])))
        
        while True:
            state, reward, terminated, truncated, info = self.env.step(action)
            state = self.obs_wrappered_env.inverse_scale_state(state_var=state)
            # print(f"mu: {state[4]}, chi: {state[5]}")
            if terminated:
                break
        # print(info)

    # TODO: 增加测试

if __name__ == "__main__":
    unittest.main()