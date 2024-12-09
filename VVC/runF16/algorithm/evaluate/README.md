
# sb3_evaluate_policy.py

随机1000个任务测试

    # 选择难度测试，
    # “env_type为origin且difficulty为-1”表示所有专家能完成的任务（10184）
    # “env_type为origin且difficulty为-2”表示所有任务（50175）
    # “env_type为origin且difficulty为-3”表示所有专家不能完成的任务（50175-10184）
    python runF16/algorithm/evaluate/sb3_evaluate_policy.py --reward_type dense --env_type origin --task_difficulty -1 --algo bc --config_file_name no_framestack_10_fraction_configs/seed1/config_10hz_loss_128_128.json

# sb3_evaluate_policy_MR.py

在马尔科夫奖励的环境上测试，随机1000个任务测试

    python runF16/algorithm/evaluate/sb3_evaluate_policy_MR.py --algo bc --config_file_name no_framestack_configs/seed1/config_10hz_loss_128_128.json

    python runF16/algorithm/evaluate/sb3_evaluate_policy_MR.py --algo rl_bc --config_file_name no_framestack_configs/seed1/config_10hz_loss_128_128_MR_sample_goal_mega_3_099.json

# sb3_evaluate_policy_on_fixed_target_set.py

在指定的任务集上测试，可以通过任务难度指定集合或者通过chi指定集合

    # 根据难度测试
    python runF16/algorithm/evaluate/sb3_evaluate_policy_on_fixed_target_set.py --algo rl_bc --reward_type dense --env_type small_difficulty --difficulty 0 --task_set_capacity 100 --config_file_name no_framestack_configs/seed2/config_10hz_loss_128_128.json

    # 在整个10184个任务的集合上测试
    python runF16/algorithm/evaluate/sb3_evaluate_policy_on_fixed_target_set.py --algo rl_bc --reward_type dense --env_type small_difficulty --difficulty 0 --task_set_capacity 10184 --config_file_name no_framestack_configs/seed2/config_10hz_loss_128_128.json

    # 根据chi测试
    python runF16/algorithm/evaluate/sb3_evaluate_policy_on_fixed_target_set.py --algo rl_bc --reward_type dense --env_type small_chi --target_chi 10 --config_file_name no_framestack_configs/seed2/config_10hz_loss_128_128.json

