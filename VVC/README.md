# code for GCPO on VVC

## prepare python environment

```bash
conda create --name GCPO_VVC python=3.8
conda activate GCPO_VVC
conda install --file requirements.txt
```

## prepare demonstrations

1. collect $\mathcal{D}_E^0$:

```bash
python runF16/rollout/rollout_guide_parallel.py
```

2. please refer to [IRPO](https://github.com/GongXudong/IRPO) for collecting $\mathcal{D}_E^1,\mathcal{D}_E^2,\mathcal{D}_E^3$.

## Train policies

### Train policy with GCPO

1. Pre-train policy with Behavioral Cloning

```bash
# in the vvc direction, cd GCPO/vvc

# train on Markovian reward
python runF16/algorithm/sb3_bc_train.py --config_file_name no_framestack_configs/seed2/config_10hz_loss_128_128_MR_sample_goal_mega_3_099.json

# train on Non-Markovian reward
python runF16/algorithm/sb3_bc_train.py --config_file_name no_framestack_configs/seed2/config_10hz_loss_128_128_sample_goal_mega_3_099.json
```

2. Fine-tune policy with PPO and self-curriculum

```bash
# in the vvc direction, cd GCPO/vvc

# train on Markovian reward with MEGA
python runF16/algorithm/sb3_rl_train_after_bc_MR_sample_goal_MEGA.py --config_file_name no_framestack_configs/seed2/config_10hz_loss_128_128_MR_sample_goal_mega_3_099.json

# train on Non-Markovian reward with MEGA
python runF16/algorithm/sb3_rl_train_after_bc_sample_goal_MEGA.py --config_file_name no_framestack_configs/seed2/config_10hz_loss_128_128_sample_goal_mega_3_099.json

# train on Non-Markovian reward with DISCERN
python runF16/algorithm/sb3_rl_train_after_bc_sample_goal_DISCERN.py --config_file_name no_framestack_configs/seed2/config_10hz_loss_128_128_sample_goal_discern_3_099.json

# train on Non-Markovian reward with RIG
python runF16/algorithm/sb3_rl_train_after_bc_sample_goal_RIG.py --config_file_name no_framestack_configs/seed2/config_10hz_loss_128_128_sample_goal_rig_3_099.json
```
