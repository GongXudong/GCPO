from gymnasium import ObservationWrapper, ActionWrapper, Env, spaces, Wrapper
from sklearn.preprocessing import StandardScaler
from typing import TypeVar, Union, Dict, List
import numpy as np
from pathlib import Path
import sys

from stable_baselines3.common.env_util import is_wrapped

# from gymnasium core.py
ObsType = TypeVar("ObsType")
WrapperObsType = TypeVar("WrapperObsType")
ActType = TypeVar("ActType")
WrapperActType = TypeVar("WrapperActType")

PROJECT_ROOT_DIR = Path(__file__).parent.parent

if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from utils.mega_wrapper import MEGAWrapper
from utils.non_markovian_reward_wrapper import NMRWrapper


class ScaledObservationWrapper(ObservationWrapper):
    
    def __init__(self, env: Env, scaler: Union[StandardScaler, Dict[str, StandardScaler]]):
        super().__init__(env)

        # 缩放与仿真器无关，只在学习器中使用
        # 送进策略网络的观测，各分量的取值都在[-inf, inf]之间，但是做了标准化       
        if isinstance(env.observation_space, spaces.Dict):
            assert isinstance(scaler, dict), "observation为Dict类型时，scaler也必须是Dict类型！！！"
            self.observation_space = spaces.Dict({
                k: spaces.Box(low=-np.inf, high=np.inf, shape=env.observation_space[k].shape) 
                for k in env.observation_space.keys()
            })
        else: 
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=env.observation_space.shape)
        self.state_scalar: Union[StandardScaler, Dict[str, StandardScaler]] = scaler
    
    def scale_state(self, state_var: Union[np.ndarray, dict]) -> Union[np.ndarray, dict]:
        """将仿真器返回的state缩放到[0, 1]之间
        """
        if isinstance(state_var, np.ndarray):
            if len(state_var.shape) == 1:
                tmp_state_var = state_var.reshape((1, -1))
                return self.state_scalar.transform(tmp_state_var).reshape((-1))
            elif len(state_var.shape) == 2:
                return self.state_scalar.transform(state_var)
            else:
                raise TypeError("state_var只能是1维或者2维！")
        elif isinstance(state_var, dict):
            res_dict = {}
            
            for k in state_var.keys():
                if len(state_var[k].shape) == 1:
                    tmp_state_var = state_var[k].reshape((1, -1))
                    res_dict[k] = self.state_scalar[k].transform(tmp_state_var).reshape((-1))
                elif len(state_var[k].shape) == 2:
                    res_dict[k] = self.state_scalar[k].transform(state_var[k])
                else:
                    raise TypeError(f"state_var[{k}]只能是1维或者2维！")
            
            return res_dict
        else:
            raise NotImplementedError("state_var的类型只能是np.ndarray或者dict")

    def observation(self, observation: ObsType) -> WrapperObsType:
        # 检查observation类型
        if isinstance(observation, np.ndarray) or isinstance(observation, dict):
            return self.scale_state(observation)
        else:
            return self.scale_state(np.array(observation))
    
    def inverse_scale_state(self, state_var: Union[np.ndarray, dict]) -> Union[np.ndarray, dict]:
        """将[0, 1]之间state变回仿真器定义的原始state。用于测试！！！
        """
        if isinstance(state_var, np.ndarray):
            if len(state_var.shape) == 1:
                tmp_state_var = state_var.reshape((1, -1))
                return self.state_scalar.inverse_transform(tmp_state_var).reshape((-1))
            elif len(state_var.shape) == 2:
                return self.state_scalar.inverse_transform(state_var)
            else:
                raise TypeError("state_var只能是1维或者2维！")
        elif isinstance(state_var, dict):
            res_dict = {}

            for k in state_var.keys():
                if len(state_var[k].shape) == 1:
                    tmp_state_var = state_var[k].reshape((1, -1))
                    res_dict[k] = self.state_scalar[k].inverse_transform(tmp_state_var).reshape((-1))
                elif len(state_var[k].shape) == 2:
                    res_dict[k] = self.state_scalar[k].inverse_transform(state_var[k])
                else:
                    raise TypeError(f"state_var[{k}]只能是1维或者2维！")
            
            return res_dict
        else:
            raise NotImplementedError("state_var的类型只能是np.ndarray或者dict")

    def sync_evaluation_stat(self, evaluation_stat: List):
        if is_wrapped(self.env, MEGAWrapper) or is_wrapped(self.env, NMRWrapper):
            self.env.sync_evaluation_stat(evaluation_stat)