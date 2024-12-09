from imitation.data.types import TransitionsMinimal, Transitions
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
from time import time
import logging
from sklearn.model_selection import train_test_split
from typing import Tuple
import sys

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from runF16.env.guidenv import GuideEnv
from runF16.env.utils.my_wrappers import ScaledObservationWrapper, ScaledActionWrapper
from runF16.configs.load_config import load_config



def load_data_from_cache(
        data_dir: Path = Path(__file__).parent / "cache" / "10hz_10_5_5",
        train_size: float=0.9, 
        validation_size: float=0.05, 
        test_size: float=0.05,
        shuffle: bool=True,
    ) -> Tuple[TransitionsMinimal, TransitionsMinimal, TransitionsMinimal]:

    scaled_obs = np.load(str(data_dir / "normalized_obs.npy"))
    scaled_acts = np.load(str(data_dir / "normalized_acts.npy"))
    infos = np.load(str(data_dir / "infos.npy"), allow_pickle=True)

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

    return (
        TransitionsMinimal(obs=train_data, acts=train_labels, infos=train_infos),
        TransitionsMinimal(obs=validation_data, acts=validation_labels, infos=validation_infos),
        TransitionsMinimal(obs=test_data, acts=test_labels, infos=test_infos)
    )