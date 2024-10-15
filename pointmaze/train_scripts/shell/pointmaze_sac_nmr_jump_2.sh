#!/bin/bash

python train_scripts/train_with_rl_sac_her.py --config-file-name configs/pointmaze-diverse-g/seed1/sac_nmr_jump_2_256_256.json
python train_scripts/train_with_rl_sac_her.py --config-file-name configs/pointmaze-diverse-g/seed2/sac_nmr_jump_2_256_256.json
python train_scripts/train_with_rl_sac_her.py --config-file-name configs/pointmaze-diverse-g/seed3/sac_nmr_jump_2_256_256.json
python train_scripts/train_with_rl_sac_her.py --config-file-name configs/pointmaze-diverse-g/seed4/sac_nmr_jump_2_256_256.json
python train_scripts/train_with_rl_sac_her.py --config-file-name configs/pointmaze-diverse-g/seed5/sac_nmr_jump_2_256_256.json