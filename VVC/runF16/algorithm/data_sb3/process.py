from imitation.data.types import TransitionsMinimal, Transitions
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
from time import time
import logging
from sklearn.model_selection import train_test_split
from typing import Tuple
import argparse
import sys

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from runF16.env.guidenv import GuideEnv
from runF16.env.utils.my_wrappers import ScaledObservationWrapper, ScaledActionWrapper
from runF16.configs.load_config import load_config

parser = argparse.ArgumentParser(description="传入配置文件")
parser.add_argument("--config_file_name", type=str, help="配置文件名", default="no_framestack_configs/seed1/config_10hz_loss_128_128.json")
args = parser.parse_args()

custom_config = load_config(args.config_file_name)

DATA_DIR = custom_config["rollout"]["expert_data_dir"]
CACHE_DATA_DIR = custom_config["bc"]["data_cache_dir"]

def load_data(
        data_dir: str=DATA_DIR, 
        cache_data_dir: str=CACHE_DATA_DIR,
        my_logger: logging.Logger=None, 
        train_size: float=0.9, 
        validation_size: float=0.05, 
        test_size: float=0.05,
        shuffle: bool=True,
    ) -> Tuple[TransitionsMinimal, TransitionsMinimal, TransitionsMinimal]:
    """加载数据，并根据比例划分训练集、验证集、测试集。

    Args:
        data_dir (str, optional): 数据存储目录. Defaults to DATA_DIR.
        my_logger (logging.Logger, optional): 日志器. Defaults to None.
        train_size (float, optional): 训练集比例. Defaults to 0.9.
        validation_size (float, optional): 验证集比例. Defaults to 0.05.
        test_size (float, optional): 测试集比例. Defaults to 0.05.
        shuffle (bool, optional): 是否打乱数据集顺序. Defaults to True.

    Returns:
        Tuple[TransitionsMinimal, TransitionsMinimal, TransitionsMinimal]: 训练集、验证集、测试集
    """
    assert np.allclose([train_size + validation_size + test_size], [1.0]), "训练集、验证集、测试集的比例之和必须为1！"
    
    start_time = time()
    env = GuideEnv(env_config={
        "available_targets_dir": PROJECT_ROOT_DIR / "my_logs_for_parallel" / data_dir,
        "available_targets_filename": "res.csv",
        "my_logger": my_logger
    })
    env = ScaledActionWrapper(ScaledObservationWrapper(env))

    obs = []
    acts = []
    infos = []

    traj_cnt = 0
    traj_file_cnt = 0
    transitions_cnt = 0
    
    for index, row in tqdm(env.available_targets.iterrows(), total=env.available_targets.shape[0]):
        target_v, target_mu, target_chi, cur_length = row["v"], row["mu"], row["chi"], row["length"]   # TODO: 注意，有的res.csv保存了序号，每保存的话，使用row[:4]
        if cur_length > 0:
            # 能够生成轨迹的目标速度矢量
            cur_filename = f"my_f16trace_{int(target_v)}_{int(target_mu)}_{int(target_chi)}.csv"
            cur_file_path = env.available_targets_dir / cur_filename
            transitions_cnt += cur_length
            traj_cnt += 1
            if cur_file_path.exists():
                if my_logger is not None:
                    my_logger.info(f"process file: {cur_filename}")
                else:
                    print(f"process file: {cur_filename}")
                traj_file_cnt += 1
                cur_traj = pd.read_csv(cur_file_path.absolute())

                # 9.960(s) for 13 files with pd.iterrows
                # for index, row in cur_traj.iterrows():
                #     obs.append([*row[1:9], target_v, target_mu, target_chi])
                #     acts.append([*row[9:12]])
                #     infos.append(None)

                # 1.742(s) for 13 files without pd.iterrows
                obs.extend(zip(
                    cur_traj['s_phi'].tolist(),
                    cur_traj['s_theta'].tolist(),
                    cur_traj['s_psi'].tolist(),
                    cur_traj['s_v'].tolist(),
                    cur_traj['s_mu'].tolist(),
                    cur_traj['s_chi'].tolist(),
                    cur_traj['s_p'].tolist(),
                    cur_traj['s_h'].tolist(),
                    [target_v] * cur_traj.count()['time'],
                    [target_mu] * cur_traj.count()['time'],
                    [target_chi] * cur_traj.count()['time'],
                ))
                acts.extend(zip(
                    cur_traj['a_p'].tolist(),
                    cur_traj['a_nz'].tolist(),
                    cur_traj['a_pla'].tolist(),
                ))
                infos.extend([None] * cur_traj.count()['time'])

    # 数据标准化. 这里的标准化最耗时.
    origin_env = GuideEnv(env_config={})
    scaled_obs_env = ScaledObservationWrapper(origin_env)
    scaled_act_env = ScaledActionWrapper(scaled_obs_env)

    scaled_obs = np.array([scaled_obs_env.scale_state(np.array(item)) for item in obs])
    scaled_acts = np.array([scaled_act_env.scale_action(np.array(item)) for item in acts])

    # 缓存标准化后的数据
    if not (Path(__file__).parent / cache_data_dir).exists():
        (Path(__file__).parent / cache_data_dir).mkdir()

    data_save_dir = Path(__file__).parent / cache_data_dir /  data_dir
    if not data_save_dir.exists():
        data_save_dir.mkdir()

    np.save(str((data_save_dir / "normalized_obs").absolute()), scaled_obs)
    np.save(str((data_save_dir / "normalized_acts").absolute()), scaled_acts)
    np.save(str((data_save_dir / "infos").absolute()), np.array(infos))

    # 输出统计量
    if my_logger is not None:
        my_logger.info(f"traj cnt: {traj_file_cnt}, transition(from *.csv) cnt: {len(obs)}, average traj length: {len(obs) / traj_file_cnt}")
        my_logger.info(f"traj cnt: {traj_cnt}, transition(from res.csv) cnt: {transitions_cnt}, average traj length: {transitions_cnt / traj_cnt}")
        my_logger.info(f"process time: {time() - start_time}(s).")
    else:
        print(f"traj cnt: {traj_file_cnt}, transition(from *.csv) cnt: {len(obs)}, average traj length: {len(obs) / traj_file_cnt}")
        print(f"traj cnt: {traj_cnt}, transition(from res.csv) cnt: {transitions_cnt}, average traj length: {transitions_cnt / traj_cnt}")
        print(f"process time: {time() - start_time}(s).")

    # 训练集、验证集、测试集划分
    train_data, tmp_data, train_labels, tmp_labels, train_infos, tmp_infos = train_test_split(
        scaled_obs, scaled_acts, np.array(infos), 
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
    print(f"划分集合后总时间：{time() - start_time}(s).")

    return (
        TransitionsMinimal(obs=train_data, acts=train_labels, infos=train_infos),
        TransitionsMinimal(obs=validation_data, acts=validation_labels, infos=validation_infos),
        TransitionsMinimal(obs=test_data, acts=test_labels, infos=test_infos)
    )


if __name__ =="__main__":
    train_trans, validation_trans, test_trans = load_data("10hz_10_5_5_iter_2_aug_high_quality")
    print(train_trans.obs.shape, train_trans.acts.shape, train_trans.infos.shape)
    print(validation_trans.obs.shape, validation_trans.acts.shape, validation_trans.infos.shape)
    print(test_trans.obs.shape, test_trans.acts.shape, test_trans.infos.shape)
