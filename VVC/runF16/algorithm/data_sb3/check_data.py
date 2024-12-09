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
from time import sleep

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from runF16.env.guidenv import GuideEnv
from runF16.env.utils.my_wrappers import ScaledObservationWrapper, ScaledActionWrapper
from runF16.configs.load_config import load_config

P_MAX = 180.
P_MIN = -180.
NZ_MAX = 9.
NZ_MIN = -4.
PLA_MAX = 1.
PLA_MIN = 0.

def check_data(
        data_dir: str, 
        my_logger: logging.Logger=None, 
    ):
        
    start_time = time()
    env = GuideEnv(env_config={
        "available_targets_dir": PROJECT_ROOT_DIR / "my_logs_for_parallel" / data_dir,
        "available_targets_filename": "res.csv",
        "my_logger": my_logger
    })
    env = ScaledActionWrapper(ScaledObservationWrapper(env))

    traj_cnt = 0
    traj_file_cnt = 0
    transitions_cnt = 0

    invalid_p_cnt, invalid_nz_cnt, invalid_pla_cnt = 0, 0, 0
    invalid_ps, invalid_nzs, invalid_plas = [], [], []
    
    for index1, row in tqdm(env.available_targets.iterrows(), total=env.available_targets.shape[0]):
        target_v, target_mu, target_chi, cur_length = row
        if cur_length > 0:
            # 能够生成轨迹的目标速度矢量
            cur_filename = f"my_f16trace_{target_v}_{target_mu}_{target_chi}.csv"
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

                for index2, row in cur_traj.iterrows():
                    if not (P_MIN <= row[9] <= P_MAX):
                        invalid_ps.append(row[9])
                        invalid_p_cnt += 1
                        print(f"invalid p: {row[9]}, in {cur_filename}!")
                    if not (NZ_MIN <= row[10] <= NZ_MAX):
                        invalid_nzs.append(row[10])
                        invalid_nz_cnt += 1
                        print(f"invalid nz: {row[10]}, in {cur_filename}!")
                    if not (PLA_MIN <= row[11] <= PLA_MAX):
                        invalid_plas.append(row[11])
                        invalid_pla_cnt += 1
                        print(f"invalid pla: {row[11]}, in {cur_filename}!")

    # 输出统计量
    print(f"traj cnt: {traj_cnt}, transition(from res.csv) cnt: {transitions_cnt}, average traj length: {transitions_cnt / traj_cnt}")
    print(f"invalid p, nz, pla: {invalid_p_cnt}, {invalid_nz_cnt}, {invalid_pla_cnt}!!!")
    invalid_nzs = np.array(invalid_nzs)
    invalid_plas = np.array(invalid_plas)
    print(np.sum(invalid_nzs < -3.), np.sum(invalid_plas > 1.))  # nz都是小于-3，p都是大于1
    print(f"process time: {time() - start_time}(s).")


if __name__ =="__main__":
    # check_data(data_dir="10hz_10_5_5_v1")  # invalid (p, nz, pla) 0, 0, 0
    check_data(data_dir="10hz_10_5_5")  # invalid (p, nz, pla) 0, 1048, 370068