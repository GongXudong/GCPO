from typing import Union, Dict, Tuple
from imitation.data.types import TransitionsMinimal, Transitions
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
from time import time
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List
import sys
import minari
from imitation.data import types

import gymnasium

def load_data(env_dataset_id: str) -> List[types.Trajectory]:
    minari_dataset = minari.load_dataset(env_dataset_id)
    
    trajs: List[types.Trajectory] = []
    for episode in minari_dataset.iterate_episodes():
        traj = types.Trajectory(obs=types.DictObs(episode.observations), acts=episode.actions, infos=None, terminal=episode.terminations[-1])
        trajs.append(traj)
    
    return trajs
    
    # obs = np.array(dataset["observations"])
    # act = np.array(dataset["actions"])
    # if "infos/action_log_probs" in dataset:
    #     infos = np.array(dataset["infos/action_log_probs"])
    # else:
    #     infos = np.array([None] * len(obs))
    # return obs, act, infos

def load_data2(env_dataset_id: str) -> Tuple[minari.MinariDataset, List[types.Trajectory]]:
    minari_dataset = minari.load_dataset(env_dataset_id)
    
    trajs: List[types.Trajectory] = []
    for episode in minari_dataset.iterate_episodes():
        traj = types.Trajectory(obs=types.DictObs(episode.observations), acts=episode.actions, infos=None, terminal=episode.terminations[-1])
        trajs.append(traj)
    
    return minari_dataset, trajs

def scale_obs(obs: Union[np.array, types.DictObs]) -> Union[Tuple[np.ndarray, StandardScaler], Tuple[types.DictObs, Dict[str, StandardScaler]]]:
    if isinstance(obs, types.DictObs):
        scaled_obs_dict, scaler_dict = {}, {}
        for k in obs.keys():
            tmp_scaler = StandardScaler()
            tmp_scaled_obs = tmp_scaler.fit_transform(obs._d[k])
            scaled_obs_dict[k] = tmp_scaled_obs
            scaler_dict[k] = tmp_scaler
        return types.DictObs(scaled_obs_dict), scaler_dict
    else:
        scaler = StandardScaler()
        scaled_obs = scaler.fit_transform(obs)
        return scaled_obs, scaler

# def split_data(
#         transitions: types.Transitions,
#         train_size: float=0.9, 
#         validation_size: float=0.05, 
#         test_size: float=0.05,
#         shuffle: bool=True,
#     ) -> Tuple[TransitionsMinimal, TransitionsMinimal, TransitionsMinimal]:

#     obs: Union[np.ndarray, types.DictObs] = transitions.obs
#     acts: np.ndarray = transitions.acts
#     infos: np.ndarray = transitions.infos

#     # 训练集、验证集、测试集划分
#     train_data, tmp_data, train_labels, tmp_labels, train_infos, tmp_infos = train_test_split(
#         obs, acts, infos, 
#         train_size=train_size, 
#         test_size=validation_size + test_size, 
#         shuffle=shuffle,
#         random_state=0,  # 保证每次得到的结果是一样的
#     )

#     validation_data, test_data, validation_labels, test_labels, validation_infos, test_infos = train_test_split(
#         tmp_data, tmp_labels, tmp_infos,
#         train_size=validation_size/(validation_size + test_size),
#         test_size=test_size/(validation_size + test_size),
#         shuffle=shuffle,
#         random_state=0,
#     )

#     return (
#         TransitionsMinimal(obs=train_data, acts=train_labels, infos=train_infos),
#         TransitionsMinimal(obs=validation_data, acts=validation_labels, infos=validation_infos),
#         TransitionsMinimal(obs=test_data, acts=test_labels, infos=test_infos)
#     )

def split_data(
        transitions: types.Transitions,
        train_size: float=0.9, 
        validation_size: float=0.05, 
        test_size: float=0.05,
        shuffle: bool=True,
    ) -> Tuple[Transitions, Transitions, Transitions]:

    obs_ids = np.arange(len(transitions.obs))
    act_ids = np.arange(len(transitions.acts))
    info_ids = np.arange(len(transitions.infos))

    # 训练集、验证集、测试集划分
    train_data, tmp_data, train_labels, tmp_labels, train_infos, tmp_infos = train_test_split(
        obs_ids, act_ids, info_ids, 
        train_size=train_size, 
        test_size=validation_size + test_size, 
        shuffle=shuffle,
        random_state=0,  # 保证每次得到的结果是一样的
    )

    validation_data, test_data, validation_labels, test_labels, validation_infos, test_infos = train_test_split(
        tmp_data, tmp_labels, tmp_infos,
        train_size=validation_size/(validation_size + test_size),
        test_size=test_size/(validation_size + test_size),
        shuffle=shuffle,
        random_state=0,
    )

    return (
        Transitions(obs=transitions.obs[train_data], acts=transitions.acts[train_labels], infos=transitions.infos[train_infos], next_obs=transitions.next_obs[train_data], dones=transitions.dones[train_data]),
        Transitions(obs=transitions.obs[validation_data], acts=transitions.acts[validation_labels], infos=transitions.infos[validation_infos], next_obs=transitions.next_obs[validation_data], dones=transitions.dones[validation_data]),
        Transitions(obs=transitions.obs[test_data], acts=transitions.acts[test_labels], infos=transitions.acts[test_infos], next_obs=transitions.next_obs[test_data], dones=transitions.dones[test_data])
    )
