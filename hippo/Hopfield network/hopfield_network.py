import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# -----------------------------
# 设置中文字体
# -----------------------------
# Windows 系统可用 "SimHei"（黑体）
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  
# 避免负号显示为方块
matplotlib.rcParams['axes.unicode_minus'] = False

class HopfieldNetwork:
    def __init__(self, n_neurons: int, device="cpu"):
        self.n = n_neurons
        self.W = torch.zeros((n_neurons, n_neurons), dtype=torch.float32, device=device)
        self.device = device

    def store_patterns(self, patterns: torch.Tensor):
        """Hebbian rule"""
        patterns = patterns.float()
        self.W = patterns.T @ patterns / self.n
        self.W.fill_diagonal_(0)

    def recall(self, state: torch.Tensor, steps: int = 5):
        x = state.clone().float()
        for _ in range(steps):
            x = torch.sign(self.W @ x)
            x[x == 0] = 1
        return x

def compute_relative_dissimilarity(net, patterns, mask_ratio: float, steps=5, eps=1e-8):
    n_patterns, n_neurons = patterns.shape
    rel_diss_list = []
    for i, p in enumerate(patterns):
        p = p.float()
        # 原始回忆
        z_orig = net.recall(p, steps)

        # 构造掩码输入
        mask = (torch.rand(n_neurons, device=net.device) > mask_ratio).float()
        masked_input = p * mask
        z_masked = net.recall(masked_input, steps)

        # 随机选一个不同的 pattern
        if n_patterns > 1:
            j = torch.randint(0, n_patterns - 1, (1,), device=net.device).item()
            if j >= i:
                j += 1
            p_rand = patterns[j].float()
        else:
            p_rand = torch.randint_like(p, low=0, high=2) * 2 - 1

        z_rand = net.recall(p_rand, steps)

        # 汉明距离
        hd_same = (z_orig != z_masked).float().mean().item()
        hd_rand = (z_orig != z_rand).float().mean().item()
        rel = hd_same / (hd_rand + eps)
        rel_diss_list.append(rel)

    return np.mean(rel_diss_list)