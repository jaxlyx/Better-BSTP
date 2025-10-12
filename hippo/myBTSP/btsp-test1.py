import numpy as np
import torch
import plotly.graph_objects as go
from binary_btsp_network_torch import BinaryBTSPNetworkTorch  # 你定义的类

net = BinaryBTSPNetworkTorch(
    n_input=2000,
    n_memory=3000,
    fw=0.01,   # 连接密度
    fq=0.005,  # 学习翻转概率
    fp=0.0025  # 输入模式稀疏度
)

# 生成训练模式
patterns = net.make_sparse_patterns(n_patterns=100)

# 训练
net.train(patterns)