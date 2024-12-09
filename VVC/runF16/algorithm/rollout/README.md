# 采样

## sb3_rollout.py

采样飞行轨迹

    # 随机一个目标速度矢量rollout，并保存飞行轨迹
    python runF16/algorithm/rollout/sb3_rollout.py --reward_type dense --algo rl_bc --save_acmi --config_file_name no_framestack_configs/seed2/config_10hz_loss_128_128.json

    # 使用某一特定的目标速度矢量rollout
    python runF16/algorithm/rollout/sb3_rollout.py --algo rl --reward_type dense --use_fixed_target --target_v 280 --target_mu 5 --target_chi 170 --config_file_name no_framestack_single_target_280_5_170_configs/seed1/config_10hz_loss_128_128.json

    python runF16/algorithm/rollout/sb3_rollout.py --algo rl_bc --reward_type dense --use_fixed_target --target_v 140 --target_mu -40 --target_chi -165 --save_acmi --config_file_name no_framestack_iter_3_configs/seed4/config_10hz_loss_256_256_128_128_64_b_1.json

## sb3_rollout_traj_length.py

采样飞行轨迹的长度：

（1）采样单条轨迹的长度；

（2）采样多条轨迹的长度：指定任务难度，从任务集上随机采样一部分任务进行测试。

    # 在固定任务上测试
    python runF16/algorithm/rollout/sb3_rollout_traj_length.py --reward_type dense --algo rl_bc --use_fixed_target --target_v 130 --target_mu 20 --target_chi 85 --config_file_name no_framestack_configs/seed2/config_10hz_loss_128_128.json

    # 选择任务难度，选择测试数量
    python runF16/algorithm/rollout/sb3_rollout_traj_length.py --reward_type dense --task_difficulty 1 --task_num 10 --res_save_filename res.csv --config_file_name no_framestack_configs/seed2/config_10hz_loss_128_128.json

    python runF16/algorithm/rollout/sb3_rollout_traj_length.py --reward_type dense --task_difficulty -1 --task_num 2000 --res_save_filename res_iter2_bc_seed1.csv --config_file_name no_framestack_iter_2_configs/seed1/config_10hz_loss_128_128.json

## sb3_rollout_for_checking_precision.py

指定任务集，采样轨迹完成时的v、 mu、 chi、 length, check策略的飞行精度

    # 在10184集合上测试
    python runF16/algorithm/rollout/sb3_rollout_for_checking_precision.py --reward_type dense --algo rl_bc --env_type small_difficulty --difficulty 0 --task_set_capacity 10184 --config_file_name no_framestack_configs/seed2/config_10hz_loss_128_128.json --save_file_name res_for_checking_precision_rl_bc_2.csv

    # 在所有任务上测试
    python runF16/algorithm/rollout/sb3_rollout_for_checking_precision.py --reward_type dense --algo rl_bc --env_type origin --task_difficulty -2 --config_file_name no_framestack_configs/seed2/config_10hz_loss_128_128.json --save_file_name test_on_all_goals/res_for_checking_precision_iter_1_rl_bc_2.csv

## sb3_rollout_all_trajs.py

使用rl_bc训出来的策略更新专家轨迹。在整个任务空间上（可枚举的50715个任务）生成轨迹。

更新是指: (1)生成专家无法完成的轨迹；（2）专家能完成的，生成更短的轨迹。

    # 使用根据{config_file_name}寻出来的rl_bc策略生成轨迹，更新{cur_expert_data_dir}中的专家轨迹
    python runF16/algorithm/rollout/sb3_rollout_all_trajs.py --cur_expert_data_dir 10hz_10_5_5_iter_2 --config_file_name no_framestack_configs/seed2/config_10hz_loss_128_128.json

## sb3_rollout_all_trajs_parallel.py

使用rl_bc训出来的策略更新专家轨迹【并行采样】。在整个任务空间上（可枚举的50715个任务）生成轨迹。
注意：该函数未更新res.csv，因此需要手动更新res.csv!!!

    # 使用根据{config_file_name}寻出来的rl_bc策略生成轨迹，更新{cur_expert_data_dir}中的专家轨迹，使用{process-num}个进程
    python runF16/algorithm/rollout/sb3_rollout_all_trajs_parallel.py --config-file-name no_framestack_iter_4_configs/seed1/config_10hz_loss_256_256_128_128_64_lambda_1e-1_evaluate_on_all_tasks.json --cur-expert-data-dir 10hz_10_5_5_iter_5 --process-num 64

    

## sb3_rollout_all_trajs_update_by_smooth.py

使用rl_bc训出来的策略更新专家轨迹。在整个任务空间上（可枚举的50715个任务）生成轨迹。

更新是指: (1)生成专家无法完成的轨迹；（2）专家能完成的，但a_p平滑度更低的轨迹。

    # 使用根据{config_file_name}寻出来的rl_bc策略生成轨迹，更新{cur_expert_data_dir}中的专家轨迹
    python runF16/algorithm/rollout/sb3_rollout_all_trajs_update_by_smooth.py --cur_expert_data_dir 10hz_10_5_5_iter_2_rebuttal_smooth --config_file_name no_framestack_configs/seed2/config_10hz_loss_128_128.json


## sb3_rollout_parallel_for_checking_precision_termination_smooth.py

在cur-expert-data-dir内的res.csv文件指定的目标集合上，使用process-num个进程，采样轨迹完成时的v、 mu、 chi、 length, check策略的飞行精度，触发的终止条件，控制量与状态量的平滑度，将文件保存在save-file-name

    python runF16/algorithm/rollout/sb3_rollout_parallel_for_checking_precision_termination_smooth.py --algo rl_bc --cur-expert-data-dir test --process-num 4 --config-file-name no_framestack_iter_2_configs/seed2/config_10hz_loss_128_128.json --save-file-name test.csv

    python runF16/algorithm/rollout/sb3_rollout_parallel_for_checking_precision_termination_smooth.py --algo rl_bc --cur-expert-data-dir 10hz_10_5_5_iter_4 --process-num 32 --config-file-name no_framestack_iter_4_configs/seed1/config_10hz_loss_256_256_128_128_64_lambda_1e-1_evaluate_on_all_tasks.json --save-file-name test_on_all_goals/res_for_checking_precision_iter_4_256_256_128_128_64_lambda_1e-1_rl_bc_1.csv