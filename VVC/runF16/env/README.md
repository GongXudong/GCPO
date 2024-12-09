|name|reward|remark|
|--|--|--|
|guidenv.py|||
|guide_ponential_angle.py|根据速度方向计算shaping reward||
|guide_scaled_ponential.py|根据速度方向计算shaping reward，放大了phi函数||
|guide_ponential_angle_and_velocity.py|根据速度方向和大小计算shaping reward||
|guide_dense_reward.py|根据速度方向和大小计算过程奖励||
|guide_dense_reward_small.py|根据速度方向和大小计算过程奖励|根据chi划分任务，每个任务包括的目标速度矢量少（训练难度小）|
|guidenv_dense_reward_sample_goal_from_dist.py|dense reward|训练过程中，根据测试中目标完成情况的统计信息采样目标，完成率越低的目标被采样到的概率越高|
