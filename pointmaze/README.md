# code for GCPO on PointMaze

## prepare python environment

```bash
conda create --name GCPO_PointMaze python=3.8
conda activate GCPO_PointMaze
conda install --file requirements.txt
```

## prepare demonstrations

Download demonstrations of PointMaze from [Minari](https://github.com/Farama-Foundation/Minari).

```bash
pip install minari
minari download pointmaze-large-v2
```

## Train policies

### Train policy with GCPO

1. Pre-train policy with Behavioral Cloning

```bash
# in the pointmaze direction, cd GCPO/pointmaze

# train on Markovian reward
python train_scripts/sb3_bc_train.py --config-file-name configs/pointmaze-diverse-g/seed1/sparse_mr_256_256_lambda_1e-1.json

# train on Non-Markovian reward
python train_scripts/sb3_bc_train.py --config-file-name configs/pointmaze-diverse-g/seed1/sparse_nmr_jump_2_256_256_lambda_1e-1.json
```

2. Fine-tune policy with PPO and self-curriculum

```bash
# in the pointmaze direction, cd GCPO/pointmaze

# train on Markovian reward
python train_scripts/sb3_rl_train_after_bc.py --config-file-name configs/pointmaze-diverse-g/seed1/sparse_mr_256_256_lambda_1e-1.json

# train on Non-Markovian reward
python train_scripts/sb3_rl_train_after_bc.py --config-file-name configs/pointmaze-diverse-g/seed1/sparse_nmr_jump_2_256_256_lambda_1e-1.json
```

### Train policy with SAC+HER+MEGA

```bash
# in the pointmaze direction, cd GCPO/pointmaze

# train on Markovian reward
python train_scripts/train_with_rl_sac_her.py --config-file-name configs/pointmaze-diverse-g/seed1/sac_mr_256_256.json

# train on Non-Markovian reward
python train_scripts/train_with_rl_sac_her.py --config-file-name configs/pointmaze-diverse-g/seed1/sac_nmr_jump_2_256_256.json
```
