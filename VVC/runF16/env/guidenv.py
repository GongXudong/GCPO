import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from collections import namedtuple
from typing import List
import logging
from pathlib import Path
import sys
from copy import deepcopy
import gc
import time

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from runF16.env.rewards.reward_base import RewardBase
from runF16.env.rewards.ponential_reward import PonentialReward2

from runF16.env.terminations.termination_base import TerminationBase
from runF16.env.terminations.reach_target_termination import ReachTargetTermination
from runF16.env.terminations.reach_target_termination2 import ReachTargetTermination2
from runF16.env.terminations.reach_target_termination_single_step import ReachTargetTerminationSingleStep
from runF16.env.terminations.crash_termination import CrashTermination
from runF16.env.terminations.timeout_termination import TimeoutTermination
from runF16.env.terminations.continuousely_move_away_termination import ContinuouselyMoveAwayTermination
from runF16.env.terminations.continuousely_move_away_termination2 import ContinuouselyMoveAwayTermination2
from runF16.env.terminations.continuousely_roll_termination import ContinuouselyRollTermination
from runF16.env.terminations.negative_overload_and_big_phi_termination import NegativeOverloadAndBigPhiTermination
from runF16.env.terminations.extreme_state_termination import ExtremeStateTermination

from runF16.f16Classes import ControlLaw, PlaneModel


class GuideEnv(gym.Env):

    def __init__(self, 
        env_config: dict
    ) -> None:
        # target_v, target_mu, target_chi, 
        # h0=5000, v0=200, 
        # step_frequence=100,
        # gamma=0.999,
        # available_targets_dir: "", # 绝对路径
        # my_logger: logging.Logger = None,


        # define spaces
        state_mins = GuideEnv.get_state_lower_bounds()
        state_maxs = GuideEnv.get_state_higher_bounds()
        self.observation_space = spaces.Box(low=np.array(state_mins, dtype=np.float32), high=np.array(state_maxs, dtype=np.float32))  # phi, theta, psi, v, mu, chi, p, h, target_v, target_mu, target_chi

        action_mins = GuideEnv.get_action_lower_bounds()
        action_maxs = GuideEnv.get_action_higher_bounds()
        self.action_space = spaces.Box(low=np.array(action_mins, dtype=np.float32), high=np.array(action_maxs, dtype=np.float32))  # p, nz, pla

        default_config = {
            "use_fixed_target": False,
            "target_v": 0, 
            "target_mu": 0, 
            "target_chi": 0,
            "h0": 5000,
            "v0": 200,
            "step_frequence": 10,
            "max_simulate_time": 40,
            "gamma": 0.995,
            "available_targets_dir": Path(__file__).parent.parent.parent / "my_logs_for_parallel" / "10hz_10_5_5",
            "available_targets_filename": "res.csv",
            "sample_target_noise_std": [5, 0.5, 0.5],  # noise std for [v, mu, chi]
            "my_logger": None,
            "sample_reachable_target": True,
            "debug": True,
            "flag_str": "Train"  # 记录该环境是用于train、evaluate或者callback，仅辅助输出用于debug！！
        }
        default_config.update(env_config)

        self.use_fixed_target = default_config.get("use_fixed_target")
        self.target_v = default_config.get("target_v")
        self.target_mu = default_config.get("target_mu")
        self.target_chi = default_config.get("target_chi")
        self.target_expert_steps = 0

        self.available_targets_dir: Path = default_config.get("available_targets_dir")
        self.available_targets_filename: str = default_config.get("available_targets_filename")

        self.available_targets = pd.read_csv((self.available_targets_dir / self.available_targets_filename).absolute())  # 采样结果，存储了所有轨迹的[target_v, target_mu, target_chi, length]
        
        self.sample_target_noise_std = default_config.get("sample_target_noise_std")
        self.sample_reachable_target = default_config.get("sample_reachable_target")

        if self.sample_reachable_target:
            self.available_targets = self.available_targets[self.available_targets.length > 0]

        self.step_frequence = default_config.get("step_frequence")
        self.max_simulate_time = default_config.get("max_simulate_time")

        self.h0, self.v0 = default_config.get("h0"), default_config.get("v0")

        self.f16cl: ControlLaw = None
        self.f16model: PlaneModel = None

        self.step_cnt = 0
        self.current_obs: dict = None  # 存f16model.getPlaneState()返回的原始观测
        
        # 数据记录
        self.state_list = []
        self.action_list = []

        self.logger: logging.Logger = default_config.get("my_logger")
        # TODO: 不手动设置级别的话，会自动变成WARNING！！！！！
        # self.logger.setLevel("DEBUG")
        # print(f"logger: {self.logger}")
        # TODO: logger在使用ray的时候不起作用！！！
        # tmp_logger = default_config.get("my_logger")
        # self.logger: logging.Logger = ray.get(tmp_logger) if type(tmp_logger) is ObjectRef else tmp_logger
        
        self.debug_mode: bool = default_config.get("debug")
        self.flag_str: str = default_config.get("flag_str")

        self.reward_funcs: List[RewardBase] = []
        self._prep_reward_funcs(default_config)

        self.termination_funcs: List[TerminationBase] = []
        self._prep_termination_funcs(default_config)

    def _prep_reward_funcs(self, env_config: dict):
        self.reward_funcs = [
            # PonentialReward2(gamma=config.get("gamma")),
            PonentialReward2(gamma=env_config.get("gamma")),
        ]

    def _prep_termination_funcs(self, env_config: dict):
        # TODO: 如何设置优先级？？？
        self.termination_funcs = [
            # ReachTargetTermination(integral_time_length=1, v_threshold=10., mu_threshold=3., chi_threshold=3., termination_reward=1., env_config=env_config, my_logger=self.logger),
            ReachTargetTermination2(integral_time_length=1, v_threshold=10., angle_threshold=3., termination_reward=1., env_config=env_config, my_logger=self.logger),
            CrashTermination(h0=0., termination_reward=-1., is_termination_reward_based_on_steps_left=False, env_config=env_config, my_logger=self.logger),
            ExtremeStateTermination(v_max=400., p_max=300., termination_reward=-1., is_termination_reward_based_on_steps_left=False, env_config=env_config, my_logger=self.logger),
            TimeoutTermination(termination_reward=-1., env_config=env_config, my_logger=self.logger),
            ContinuouselyMoveAwayTermination(time_window=2, termination_reward=-1., env_config=env_config, my_logger=self.logger),
            # 使用速度矢量判断的话，相当于条件更严格，更容易触发持续远离目标，会增加训练难度
            # ContinuouselyMoveAwayTermination2(time_window=2, ignore_velocity_vector_angle_error=1., termination_reward=-1., is_termination_reward_based_on_steps_left=False, env_config=env_config, my_logger=self.logger),
            ContinuouselyRollTermination(continuousely_roll_threshold=720, termination_reward=-1., is_termination_reward_based_on_steps_left=False, env_config=env_config, my_logger=self.logger),
            NegativeOverloadAndBigPhiTermination(time_window=2, negative_overload_threshold=0., big_phi_threshold=60., termination_reward=-1., is_termination_reward_based_on_steps_left=False, env_config=env_config, my_logger=self.logger),
        ]

    def reset(self, *, seed = None, options = None):
        """

        Args:
            seed (_type_, optional): _description_. Defaults to None.
            options (_type_, optional): _description_. Defaults to None.
        
        Returns:
            tuple[ObsType, dict[str, Any]]: obs, info
        """
        super().reset(seed=seed, options=options)

        if not self.use_fixed_target:
            self.sample_target()

        self.f16cl = ControlLaw(stepTime=self.step_time)  # 控制律模型内有积分，所以也要重新初始化
        self.f16model = PlaneModel(self.h0, self.v0, stepTime=self.step_time)  # 飞机初始化，会配平，
        
        for t_func in self.termination_funcs:
            t_func.reset()
        
        for r_func in self.reward_funcs:
            r_func.reset()

        self.step_cnt = 0
        self.current_obs = self.f16model.getPlaneState()
        self.state_list = [self.convert_dict_to_state_vars(self.current_obs)]
        self.action_list = []

        info = {}
        info["origin_obs"] = self.current_obs

        return self.state_list[-1], info

    def close(self):
        # 先回收旧的self.f16cl和self.f16model，否则会存在内存持续增长的情况！！！！！
        if self.f16cl is not None:
            # if hasattr(self.f16cl, "f16CL"):
            #     del self.f16cl.f16CL
            del self.f16cl
        if self.f16model is not None:
            # if hasattr(self.f16model, "f16Model"):
            #     del self.f16model.f16Model
            del self.f16model
        gc.collect()

    def step(self, action):
        """_summary_

        Args:
            action (_type_): action是制导律输出的[p, nz, pla]

        Returns:
            tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]: next_obs, reward, terminated, truncated, info

            info = {
                "origin_obs": ,
                "rewards": ,
                "termination": ,
            }
        """
        self.step_cnt += 1
        self.action_list.append(action)

        guide_output = {
            "p": action[0],
            "nz": action[1],
            "pla": action[2],
            "rud": 0.
        }  # [p, nz, pla, rud], rud置为0

        self.f16cl.step(guide_output, self.current_obs)
        control_law_output = self.f16cl.getOutputDict()
        self.f16model.step(control_law_output)
        next_obs = self.f16model.getPlaneState()

        # check obs for NaN!!!!!!!!!
        if np.any(np.isnan(self.convert_dict_to_state_vars(next_obs))):
            tmp = []
            for item_obs, item_act in zip(self.state_list, self.action_list):
                if self.debug_mode:
                    print(item_obs, item_act)
                tmp.append([*item_obs, *item_act])
            tmp_pd = pd.DataFrame(data=tmp, columns=['s_phi', 's_theta', 's_psi', 's_v', 's_mu', 's_chi', 's_p', 's_h', 'target_v', 'target_mu', 'target_chi','a_p', 'a_nz', 'a_pla'])
            tmp_pd.to_csv(str((PROJECT_ROOT_DIR / "my_logs" / 'nan_states.csv').absolute()), index=False)

        self.state_list.append(self.convert_dict_to_state_vars(next_obs))

        terminated, truncated, reward, info = False, False, 0., {}
        info["step"] = self.step_cnt
        info["target_v"] = self.target_v
        info["target_mu"] = self.target_mu
        info["target_chi"] = self.target_chi
        info['is_success'] = False  # https://stable-baselines3.readthedocs.io/en/feat-gymnasium-support/common/logger.html#eval 评估中success_rate要求这个字段

        # 判断是否终止
        for t_func in self.termination_funcs:
            terminated, truncated, reward = t_func.get_termination_and_reward(
                state=self.state_list[-2], 
                next_state=self.state_list[-1],  # ContinuouselyRollTermination,
                step_cnt=self.step_cnt,  # TimeoutTermination,
                state_list=self.state_list,  # ReachTargetTermination, 
                nz=next_obs["nz"],  # NegativeOverloadAndBigPhiTermination,
                ve=self.current_obs["ve"],  # ContinuouselyMoveAwayTermination2
                vn=self.current_obs["vn"],  # ContinuouselyMoveAwayTermination2
                vh=self.current_obs["vh"],  # ContinuouselyMoveAwayTermination2
                goal_v=self.target_v,
                goal_mu=self.target_mu,
                goal_chi=self.target_chi
            )
            if terminated:
                info["termination"] = str(t_func)
                output_str = f"{str(t_func)}。steps: {self.step_cnt}。target: ({self.target_v}, {self.target_mu}, {self.target_chi})。"
                if not self.use_fixed_target:
                    output_str += f"expert steps: {self.target_expert_steps}。"
                
                if isinstance(t_func, ReachTargetTermination) or isinstance(t_func, ReachTargetTermination2) or isinstance(t_func, ReachTargetTerminationSingleStep):
                    info['is_success'] = True

                if self.logger is not None:
                    self.logger.warning('logger, ' +  output_str)
                else:
                    if isinstance(t_func, ReachTargetTermination) or isinstance(t_func, ReachTargetTermination2) or isinstance(t_func, ReachTargetTerminationSingleStep):
                        # reach target, 绿色打印
                        if self.debug_mode:
                            print(f"print, {self.flag_str}, \033[32m{str(t_func)}。\033[0m steps: {self.step_cnt}。target: ({self.target_v:.2f}, {self.target_mu:.2f}, {self.target_chi:.2f})。achieved target: ({next_obs['v']:.2f}, {next_obs['mu']:.2f}, {next_obs['chi']:.2f})。expert steps: {self.target_expert_steps}。")
                    else:
                        # 其它终止条件，红色打印
                        if self.debug_mode:
                            print(f"print, {self.flag_str}, \033[31m{str(t_func)}。\033[0m steps: {self.step_cnt}。target: ({self.target_v:.2f}, {self.target_mu:.2f}, {self.target_chi:.2f})。achieved target: ({next_obs['v']:.2f}, {next_obs['mu']:.2f}, {next_obs['chi']:.2f})。expert steps: {self.target_expert_steps}。")

                if "rewards" not in info:
                    info["rewards"] = {}
                info["rewards"][str(t_func)] = reward
                break

        # 计算奖励
        for r_func in self.reward_funcs:
            tmp_reward = r_func.get_reward(
                state=self.state_list[-2], 
                next_state=self.state_list[-1],
                done=terminated,
                ve=self.current_obs["ve"],  # 用于计算dense reward和基于速度矢量角度的ponential_reward
                vn=self.current_obs["vn"],  # 用于计算dense reward和基于速度矢量角度的ponential_reward
                vh=self.current_obs["vh"],  # 用于计算dense reward和基于速度矢量角度的ponential_reward
                next_ve=next_obs["ve"],  # 用于计算dense reward和基于速度矢量角度的ponential_reward
                next_vn=next_obs["vn"],  # 用于计算dense reward和基于速度矢量角度的ponential_reward
                next_vh=next_obs["vh"],  # 用于计算dense reward和基于速度矢量角度的ponential_reward
            )
            if "rewards" not in info:
                info["rewards"] = {}
            info["rewards"][str(r_func)] = tmp_reward
            reward += tmp_reward / len(self.reward_funcs)

        info["origin_obs"] = self.current_obs
        info["action"] = guide_output

        self.current_obs = deepcopy(next_obs)
        return self.state_list[-1], reward, terminated, truncated, info

    def sample_target(self):
        """从采样结果中，随机挑出一个轨迹长度大于0的(target_v, target_mu, target_chi)，并增加一定的噪声
        """
        if self.use_fixed_target:
            print(f"use fixed goal: {self.target_v}, {self.target_mu}, {self.target_chi}.")
        else:
            if self.sample_reachable_target:
                # 采样专家能完成的轨迹
                while True:
                    tmp = self.available_targets.sample(n=1)
                    if tmp['length'].item() != 0:
                        sampled_v, sampled_mu, sampled_chi = tmp["v"].item(), tmp["mu"].item(), tmp["chi"].item()
                        
                        self.target_v = sampled_v + (2 * np.random.rand() -1.) * self.sample_target_noise_std[0]
                        self.target_mu = sampled_mu + (2 * np.random.rand() -1.) * self.sample_target_noise_std[1]
                        self.target_chi = sampled_chi + (2 * np.random.rand() -1.) * self.sample_target_noise_std[2]
                        self.target_expert_steps = tmp['length'].item()

                        if self.logger is not None:
                            self.logger.info(f"sample target in {self.flag_str}: v={sampled_v}, mu={sampled_mu}, chi={sampled_chi}, length: {tmp['length'].item()}")
                            self.logger.info(f"after add noise: v={self.target_v:.2f}, mu={self.target_mu:.2f}, chi={self.target_chi:.2f}")

                        break
            else:
                # 采样任何一个轨迹

                tmp = self.available_targets.sample(n=1)
                sampled_v, sampled_mu, sampled_chi = tmp["v"].item(), tmp["mu"].item(), tmp["chi"].item()
                        
                self.target_v = sampled_v + (2 * np.random.rand() -1.) * self.sample_target_noise_std[0]
                self.target_mu = sampled_mu + (2 * np.random.rand() -1.) * self.sample_target_noise_std[1]
                self.target_chi = sampled_chi + (2 * np.random.rand() -1.) * self.sample_target_noise_std[2]
                self.target_expert_steps = tmp['length'].item()

                if self.debug_mode:
                    print(f"Sample in {self.flag_str}, randomly, {self.target_v}, {self.target_mu}, {self.target_chi}")
    
    def sample_noise(self):
        """采样噪声
        """
        return (
            (2 * np.random.rand() -1.) * self.sample_target_noise_std[0], 
            (2 * np.random.rand() -1.) * self.sample_target_noise_std[1], 
            (2 * np.random.rand() -1.) * self.sample_target_noise_std[2]
        )

    def convert_dict_to_state_vars(self, state_dict):
        """将仿真器返回的字典类型观测转换为环境定义的观测(GuideEnv.get_state_vars()定义的namedtuple)

        Args:
            state_dict (dict): 键包括：lef, npos, epos, h, alpha, beta, phi, theta, psi, p, q, r, v, vn, ve, vh, nx, ny, nz, ele, ail, rud, thrust, lon, lat, mu, chi

        Returns:
            namedtuple: _description_
        """

        state_vars_type = GuideEnv.get_state_vars()
        return state_vars_type(
            phi=state_dict['phi'], theta=state_dict['theta'], psi=state_dict['psi'], 
            v=state_dict['v'], mu=state_dict['mu'], chi=state_dict['chi'],
            p=state_dict['p'], h=state_dict['h'], 
            target_v=self.target_v, target_mu=self.target_mu, target_chi=self.target_chi
        )

    @property
    def step_time(self):
        return 1. / self.step_frequence

    @staticmethod
    def get_state_vars():
        """学习器使用的观测

        Returns:
            _type_: _description_
        """
        return namedtuple("state_vars", ["phi", "theta", "psi", "v", "mu", "chi", "p", "h", "target_v", "target_mu", "target_chi"])

    @staticmethod
    def get_state_lower_bounds():
        """返回观测的下限

        Returns:
            _type_: _description_
        """
        state_vars_type = GuideEnv.get_state_vars()
        return state_vars_type(phi=-180., theta=-90., psi=-180., v=0., mu=-90., chi=-180., p=-300., h=0., target_v=0., target_mu=-90., target_chi=-180.)

    @staticmethod
    def get_state_higher_bounds():
        """返回观测的上限

        Returns:
            _type_: _description_
        """
        state_vars_type = GuideEnv.get_state_vars()
        return state_vars_type(phi=180., theta=90., psi=180., v=1000., mu=90., chi=180., p=300., h=20000., target_v=1000., target_mu=90., target_chi=180.)

    @staticmethod
    def get_action_vars():
        """学习器输出的动作

        Returns:
            _type_: _description_
        """
        return namedtuple("action_vars", ["p", "nz", "pla"])
    
    @staticmethod
    def get_action_lower_bounds():
        """动作的下限

        Returns:
            _type_: _description_
        """
        action_vars_type = GuideEnv.get_action_vars()
        return action_vars_type(p=-180., nz=-4., pla=0.)

    @staticmethod
    def get_action_higher_bounds():
        """动作的上限

        Returns:
            _type_: _description_
        """
        action_vars_type = GuideEnv.get_action_vars()
        return action_vars_type(p=180., nz=9., pla=1.)


if __name__ == "__main__":
    # from ray.rllib.utils import check_env
    env = GuideEnv(env_config={"target_v": 200, "target_mu": 20, "target_chi": 30})
    # check_env(env)
