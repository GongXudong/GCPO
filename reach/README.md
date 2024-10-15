# code for GCPO on Reach

## prepare python environment

```bash
conda create --name GCPO_Reach python=3.8
conda activate GCPO_Reach
conda install --file requirements.txt
```

## prepare demonstrations

Rollout demonstrations with the script ```rollout/rollout_reach_by_pid.ipynb``` for Markovian reward problem and ```rollout/rollout_my_reach_waypoint_by_pid.ipynb``` for Non-Markovian reward problem.

## Train policies

### Train policy with GCPO

1. Pre-train policy with Behavioral Cloning

```bash
# in the reach direction, cd GCPO/reach

# train on Markovian reward
python train_scripts/sb3_bc_train.py --config-file-name configs/reach/seed1/reacher_mr_256_256_kl_1e-1.json

# train on Non-Markovian reward
python train_scripts/sb3_bc_train.py --config-file-name configs/reach/seed1/reacher_nmr_waypoint_256_256_kl_1e-2.json
```

2. Fine-tune policy with PPO and self-curriculum

```bash
# in the reach direction, cd GCPO/reach

# train on Markovian reward
python train_scripts/sb3_rl_train_after_bc.py --config-file-name configs/reach/seed1/reacher_mr_256_256_kl_1e-1.json

# train on Non-Markovian reward
python train_scripts/sb3_rl_train_after_bc.py --config-file-name configs/reach/seed1/reacher_nmr_waypoint_256_256_kl_1e-2.json
```

### Train policy with SAC+HER+MEGA

```bash
# in the reach direction, cd GCPO/reach

# train on Markovian reward
python train_scripts/train_with_rl_sac_her.py --config-file-name configs/reach/seed1/reacher_mr_sac_256_256.json

# train on Non-Markovian reward
python train_scripts/train_with_rl_sac_her.py --config-file-name configs/reach/seed1/reacher_nmr_waypoint_sac_her_256_256.json
```
