#!/bin/bash

python train_scripts/sb3_rl_train_after_bc.py --config-file-name configs/pointmaze-diverse-g/seed1/sparse_nmr_20_256_256_lambda_1e-1.json
python train_scripts/sb3_rl_train_after_bc.py --config-file-name configs/pointmaze-diverse-g/seed2/sparse_nmr_20_256_256_lambda_1e-1.json
python train_scripts/sb3_rl_train_after_bc.py --config-file-name configs/pointmaze-diverse-g/seed3/sparse_nmr_20_256_256_lambda_1e-1.json
python train_scripts/sb3_rl_train_after_bc.py --config-file-name configs/pointmaze-diverse-g/seed4/sparse_nmr_20_256_256_lambda_1e-1.json
python train_scripts/sb3_rl_train_after_bc.py --config-file-name configs/pointmaze-diverse-g/seed5/sparse_nmr_20_256_256_lambda_1e-1.json