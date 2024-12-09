# Guide模型
    
    输入：目标速度矢量、飞机状态
    输出：(p, nz, pla, rud)序列
    注：rud可以设置为0，神经网路的输出至少包含p和nz，pla可由神经网络输出，也可以由简单的规则给出

# ControlLow模型


# 采样数据

## 并行采样

在项目根目录执行（保证环境中安装了ray）：

    python runF16/rollout_parallel.py
