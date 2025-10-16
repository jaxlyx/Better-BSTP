import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import numpy as np
import torch
from typing import Optional, Sequence

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  
matplotlib.rcParams['axes.unicode_minus'] = False

import torch.nn as nn
from tqdm import tqdm

class BinaryBTSPNetworkTorch:

    def __init__(
        self,
        n_input: int,
        n_memory: int,
        fq: float,       # BTSP 更新概率 (突触翻转)
        fp: float,  
        fw: float,      # 输入模式稀疏度 (active neuron ratio)
        threshold: int = None,
        device: Optional[str] = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.n_input = n_input
        self.n_memory = n_memory
        self.fw = fw
        self.fq = fq
        self.fp = fp

        # 固定的连接矩阵：表示哪些CA1神经元连接到哪些CA3神经元
        self.connections = (torch.rand(n_memory, n_input, device=self.device) < fw).to(torch.uint8)
        
        # 突触强度初始化为0
        self.synaptic_strengths = torch.zeros(n_memory, n_input, device=self.device)
        # 只在实际存在的连接上才有突触强度
        self.synaptic_strengths = self.synaptic_strengths * self.connections.float()

        # 阈值设置
        if threshold is None:
            # 基于连接数的阈值
            incoming = self.connections.sum(dim=1)
            thr = (incoming.float()/2).ceil().to(torch.int)
            thr[thr < 1] = 1
            self.threshold = thr
        else:
            thr = torch.tensor(threshold, device=self.device, dtype=torch.int)
            if thr.shape == ():
                thr = thr.repeat(n_memory)
            self.threshold = thr

    def forward(self, input_pattern: Sequence[int]) -> torch.Tensor:
        """前向传播"""
        x = torch.as_tensor(input_pattern, dtype=torch.float32, device=self.device)
        if x.shape != (self.n_input,):
            raise ValueError(f"input_pattern must have shape ({self.n_input},), got {x.shape}")
        
        # 使用突触强度进行计算
        weighted_input = self.synaptic_strengths * x.unsqueeze(0)  # 广播输入到所有记忆神经元
        summed = weighted_input.sum(dim=1)  # shape: (n_memory,)
        out = (summed >= self.threshold.float()).to(torch.uint8)
        return out

    def btsp_update(self, input_pattern: Sequence[int]) -> None:
        """
        BTSP 更新机制（按 CA1 神经元整体更新）：
        - 对每个 CA1 神经元，以 fq 的概率决定是否进行更新；
        - 若该神经元被选中更新，则对所有与活跃 CA3 相连的突触施加 50% 概率翻转。
        """
        x = torch.as_tensor(input_pattern, dtype=torch.uint8, device=self.device)
        active_idx = torch.nonzero(x, as_tuple=True)[0]
        if active_idx.numel() == 0:
            return

        # 当前模式下活跃输入连接情况
        active_connections = self.connections[:, active_idx].bool()  # shape: (n_memory, n_active)

        # 哪些CA1神经元至少有一个连接到活跃CA3
        has_active_connection = active_connections.any(dim=1)  # shape: (n_memory,)

        # 对这些有连接的CA1细胞，以 fq 概率决定是否更新
        rand_neuron_update = torch.rand(self.n_memory, device=self.device)
        neuron_update_mask = (rand_neuron_update < self.fq) & has_active_connection  # 哪些CA1要更新

        if not neuron_update_mask.any():
            return

        # 对每个被选中更新的CA1细胞，执行突触翻转（附加50%概率）
        neuron_indices = torch.nonzero(neuron_update_mask, as_tuple=True)[0]

        for neuron_idx in neuron_indices:
            # 找出该神经元与活跃CA3相连的突触
            connected_active = active_connections[neuron_idx]  # bool mask over active_idx
            if connected_active.any():
                input_indices = active_idx[connected_active]

                # 对这些突触以50%概率翻转
                flip_mask = (torch.rand_like(input_indices.float()) < 0.5)
                if flip_mask.any():
                    flip_indices = input_indices[flip_mask]
                    cur_strengths = self.synaptic_strengths[neuron_idx, flip_indices]
                    new_strengths = 1.0 - cur_strengths
                    self.synaptic_strengths[neuron_idx, flip_indices] = new_strengths

    def btsp_update_fast(self, input_pattern: Sequence[int]) -> None:
        x = torch.as_tensor(input_pattern, dtype=torch.uint8, device=self.device)
        active_idx = torch.nonzero(x, as_tuple=True)[0]
        if active_idx.numel() == 0:
            return

        # 活跃输入与连接
        active_connections = self.connections[:, active_idx].bool()  # (n_memory, n_active)
        has_active_connection = active_connections.any(dim=1)       # 哪些CA1神经元有连接

        # 随机选中 CA1 神经元更新
        neuron_update_mask = (torch.rand(self.n_memory, device=self.device) < self.fq) & has_active_connection
        if not neuron_update_mask.any():
            return

        # 获取被选中的 neuron 索引
        neuron_indices = torch.nonzero(neuron_update_mask, as_tuple=True)[0]

        # 选中 neuron 的所有活跃突触
        rows, cols = torch.nonzero(active_connections[neuron_indices], as_tuple=True)
        # 对这些突触以50%概率翻转
        flip_mask = torch.rand(rows.shape[0], device=self.device) < 0.5
        rows = rows[flip_mask]
        cols = cols[flip_mask]

        # 映射到全局 input 索引
        input_indices = active_idx[cols]
        memory_indices = neuron_indices[rows]

        # 批量翻转
        cur = self.synaptic_strengths[memory_indices, input_indices]
        self.synaptic_strengths[memory_indices, input_indices] = 1.0 - cur

    def train(self, patterns: torch.Tensor, batch_size: int = 64) -> None:
        n_patterns = patterns.shape[0]

        for i in tqdm(range(0, n_patterns, batch_size), desc="Training patterns"):
            batch_patterns = patterns[i:i+batch_size]  # (B, n_input)
            
            # 遍历 batch 中每个模式
            for pat in batch_patterns:
                # 使用向量化 btsp_update_fast 替代原来的 btsp_update
                self.btsp_update_fast(pat)

    def train_patterns(self, patterns: torch.Tensor, batch_size: int = 64) -> None:
        """
        批量训练多个模式。
        patterns: shape (n_patterns, n_input)
        """
        n_patterns = patterns.shape[0]

        for i in tqdm(range(0, n_patterns, batch_size), desc="Training Patterns"):
            batch_patterns = patterns[i:i+batch_size]  # (B, n_input)
            
            # 遍历 batch 中每个模式，使用向量化更新
            for pat in batch_patterns:
                self.btsp_update_fast(pat)

    def get_effective_weights(self) -> torch.Tensor:
        """获取有效的权重矩阵（连接 + 突触强度）"""
        return self.connections.float() * self.synaptic_strengths

    def make_sparse_patterns(self, n_patterns: int) -> torch.Tensor:
        ##生成稀疏二值输入模式矩阵
        if not (0.0 < self.fp <= 1.0):
            raise ValueError("fp must be in (0, 1]")
        return (torch.rand(n_patterns, self.n_input, device=self.device) < self.fp).to(torch.uint8)

    def compute_relative_dissimilarity_vectorized(
        self,
        patterns: torch.Tensor,
        mask_ratios: Sequence[float],
        n_repeats: int = 10,
        batch_size: int = 128,
        eps: float = 1e-8,
    ) -> np.ndarray:

        device = self.device
        patterns = patterns.to(device).float()  # (n_patterns, n_input)
        n_patterns, n_input = patterns.shape

        # ---------------- 预计算：无掩码下所有 pattern 的 CA1 输出与平均活跃神经元数 ----------------
        all_active_counts = []
        with torch.no_grad():
            for i in range(0, n_patterns, batch_size):
                batch_patterns = patterns[i:i+batch_size]  # (B, n_input)
                # 矩阵乘法，得到 (n_CA1, B) 的激活 bool 矩阵
                z_all_batch = (self.synaptic_strengths @ batch_patterns.T) >= self.threshold[:, None]
                # 统计每个 pattern 的活跃 CA1 数
                active_counts = z_all_batch.sum(dim=0).cpu().numpy()  # shape (B,)
                all_active_counts.append(active_counts)
        all_active_counts = np.concatenate(all_active_counts)
        avg_active = float(all_active_counts.mean())
        # 避免除零
        avg_active = max(avg_active, eps)
        # print(f"Baseline average active CA1 per pattern (no mask): {avg_active:.4f}")

        # 计算不同掩码下的记忆保留
        mean_rel_diss = []

        for mask_ratio in tqdm(mask_ratios, desc="Mask Ratios"):
            rel_diss_list = []

            for _ in range(n_repeats):
                # 随机打乱 pattern
                perm = torch.randperm(n_patterns, device=device)
                patterns_shuffled = patterns[perm]

                for i in range(0, n_patterns, batch_size):
                    batch_patterns = patterns_shuffled[i:i+batch_size]  # (B, n_input)
                    B = batch_patterns.shape[0]

                    # 原输出（CA1），形状 (n_CA1, B)
                    with torch.no_grad():
                        z_orig = (self.synaptic_strengths @ batch_patterns.T) >= self.threshold[:, None]
                        # 转 float 用于后续比较，直接保持在 device 上
                        z_orig = z_orig.to(torch.uint8)

                    # 构造掩码：更快的随机掩码，按比例置0（每个样本独立）
                    if mask_ratio > 0:
                        # mask True 表示保留，False 表示被遮挡
                        # 使用 rand > mask_ratio 保证每个位以 mask_ratio 概率被遮挡
                        mask = (torch.rand((B, n_input), device=device) > mask_ratio).to(batch_patterns.dtype)
                        batch_masked = batch_patterns * mask
                    else:
                        batch_masked = batch_patterns

                    # 掩码后输出
                    with torch.no_grad():
                        z_masked = (self.synaptic_strengths @ batch_masked.T) >= self.threshold[:, None]
                        z_masked = z_masked.to(torch.uint8)

                    # 计算掩码前后海明距离（不同位数），按 pattern 列统计
                    # z_orig, z_masked: (n_CA1, B) -> 对列求和
                    hd_same = (z_orig != z_masked).float().sum(dim=0)  # shape (B,)
                    rel = (hd_same / (avg_active + eps)).cpu().numpy()  # 归一化并搬到 CPU
                    rel_diss_list.append(rel)

            # 汇总平均
            rel_diss_list_flat = np.concatenate(rel_diss_list)  # 所有 repeats 与 batch 的 pattern-level 值
            mean_rel_diss.append(rel_diss_list_flat.mean())

        return np.array(mean_rel_diss)

