# 功能解释

## augment_trajs.py

对专家轨迹做数据增强（反转目标速度矢量中的chi）

    # 对10hz_10_5_5_iter_1_aug目录中的专家轨迹做数据增强
    python augment_trajs.py --expert_data_dir 10hz_10_5_5_iter_1_aug

## analysis_expert_trajectory.py

读一条专家轨迹，分析速度矢量的变化（飞行过程中，速度矢量误差是变大还是变小）

## check_data.py

检查专家轨迹中的p, nz, pla是否都在指定的范围内（p, nz, pla的范围在文件开头处指定）

## load_data_by_chi.py

加载指定chi对应的专家轨迹

## load_data_by_difficulty.py

加载指定难度的专家轨迹

## load_data.py

从指定的缓存目录中加载专家数据，划分训练集、验证集、测试集，并且随机打乱训练数据数序

## process.py

读所有专家轨迹，标准化，并缓存到指定目录

## process_a_fraction_of_data.py

读取一定比例p的数据(从5个难度的任务中各抽取p比例的数据)，标准化，并缓存到指定目录