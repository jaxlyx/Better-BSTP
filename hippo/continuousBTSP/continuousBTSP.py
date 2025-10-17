import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import numpy as np
from typing import Optional, Sequence

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  
matplotlib.rcParams['axes.unicode_minus'] = False

import torch
import torch.nn as nn
from tqdm import tqdm




class continuousBTSPNetwork:

    def __init__(
        self,
        n_input: int,
        n_memory: int,
        fq: float,       # BTSP 更新概率 
        fp: float,  
        fw: float,      # 输入模式稀疏度 
        threshold: int = None,
        device: Optional[str] = None,
        n_subsynapses: int = 8,

    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.n_input = n_input
        self.n_memory = n_memory
        self.n_subsynapses = n_subsynapses
        self.fw = fw
        self.fq = fq
        self.fp = fp

        # 固定的连接矩阵,连续化的关键位置
        self.connections: torch.Tensor = torch.zeros(
            (n_memory, n_input, n_subsynapses), dtype=torch.bool, device=self.device
        )
        self.synaptic_strengths = torch.zeros(
            (n_memory, n_input, n_subsynapses), dtype=torch.float32, device=self.device
        )
        self.synaptic_strengths *= self.connections.float()

         
        pair_connected = torch.rand((n_memory, n_input), device=self.device) < self.fw  # (n_memory, n_input)
        # 若连接存在，则拥有所有子突触
        self.connections = pair_connected.unsqueeze(-1).expand(-1, -1, n_subsynapses).clone()  # (n_memory, n_input, n_subsynapses)
 
        # 只保留实际存在的突触
        self.synaptic_strengths *= self.connections.float()

        # 阈值计算
        if threshold is None:
            # 平均每个 CA1 接收到的有效突触数（乘以 n_subsynapses）
            incoming = self.connections.sum(dim=(1, 2))
            thr = (incoming.float() / 2).ceil().to(torch.int)
            thr[thr < 1] = 1
            self.threshold = thr
        else:
            thr = torch.tensor(threshold, device=self.device, dtype=torch.int)
            if thr.shape == ():
                thr = thr.repeat(n_memory)
            self.threshold = thr


    def forward(self, input_pattern: Sequence[int]) -> torch.Tensor:
        x = torch.as_tensor(input_pattern, dtype=torch.float32, device=self.device)
        if x.shape != (self.n_input,):
            raise ValueError(f"input_pattern must have shape ({self.n_input},), got {x.shape}")

        # 广播输入 (1, n_input, 1)
        x_expanded = x.unsqueeze(0).unsqueeze(-1)  # (1, n_input, 1)
        
        # 计算有效输入：加权平均时只考虑存在的子突触
        weighted = self.synaptic_strengths * self.connections.float() * x_expanded
        effective_input = weighted.mean(dim=2).sum(dim=1)  # (n_memory,)

        out = (effective_input >= self.threshold.float()).to(torch.uint8)
        return out


    def btsp_update_fast(self, input_pattern: Sequence[int]) -> None:
        x = torch.as_tensor(input_pattern, dtype=torch.uint8, device=self.device)
        active_idx = torch.nonzero(x, as_tuple=True)[0]
        if active_idx.numel() == 0:
            return

        active_connections = self.connections[:, active_idx]  # (n_memory, n_active, n_subsynapses)
        has_active_connection = active_connections.any(dim=(1, 2))

        neuron_update_mask = (torch.rand(self.n_memory, device=self.device) < self.fq) & has_active_connection
        if not neuron_update_mask.any():
            return

        neuron_indices = torch.nonzero(neuron_update_mask, as_tuple=True)[0]

        # 向量化更新
        subconn = self.synaptic_strengths[neuron_indices][:, active_idx, :]  # (N_upd, n_active, n_subsynapses)
        flip_mask = torch.rand_like(subconn) < 0.5

        self.synaptic_strengths[neuron_indices[:, None], active_idx[None, :], :] = torch.where(
            flip_mask, 1.0 - subconn, subconn
        )



    def train(self, patterns: torch.Tensor, batch_size: int = 64) -> None:
        """
        批量训练多个模式，使用向量化更新加速。
        
        patterns: shape (n_patterns, n_input)
        batch_size: 每次处理的模式数量，可根据显存调节
        """
        n_patterns = patterns.shape[0]

        for i in tqdm(range(0, n_patterns, batch_size), desc="Training patterns"):
            batch_patterns = patterns[i:i+batch_size]  # (B, n_input)
            
            # 遍历 batch 中每个模式
            for pat in batch_patterns:
                # 使用向量化 btsp_update_fast 替代原来的 btsp_update
                self.btsp_update_fast(pat)

    def train_patterns(self, patterns: torch.Tensor, batch_size: int = 64) -> None:
        """
        批量训练多个模式（向量化 + 进度条）。
        
        patterns: shape (n_patterns, n_input)
        batch_size: 每次处理的模式数量，可根据显存调节
        """
        n_patterns = patterns.shape[0]

        for i in tqdm(range(0, n_patterns, batch_size), desc="Training Patterns"):
            batch_patterns = patterns[i:i+batch_size]  # (B, n_input)
            
            # 遍历 batch 中每个模式，使用向量化更新
            for pat in batch_patterns:
                self.btsp_update_fast(pat)

    def get_effective_weights(self) -> torch.Tensor:
        """返回平均后的有效突触权重矩阵"""
        return (self.synaptic_strengths * self.connections.float()).mean(dim=2)


    def make_sparse_patterns(self, n_patterns: int) -> torch.Tensor:
        """
        生成稀疏二值输入模式矩阵
        n_patterns: 模式数量
        稀疏度由 self.fp 控制
        """
        if not (0.0 < self.fp <= 1.0):
            raise ValueError("fp must be in (0, 1]")
        return (torch.rand(n_patterns, self.n_input, device=self.device) < self.fp).to(torch.uint8)

    def compute_relative_dissimilarity_vectorized_fast(
        self,
        patterns: torch.Tensor,
        mask_ratios: Sequence[float],
        n_repeats: int = 10,
        batch_size: int = 128,
        eps: float = 1e-8,
    ) -> np.ndarray:
        """
        利用稀疏输入 + GPU 矩阵运算，快速计算连续 BTSP 网络
        在不同掩码比例下的平均归一化海明距离。
        """
        device = self.device
        patterns = patterns.to(device).float()  # (n_patterns, n_input)
        n_patterns, n_input = patterns.shape

        # ---------------- 预计算有效权重 ----------------
        effective_weights = (self.synaptic_strengths * self.connections.float()).mean(dim=2)  # (n_memory, n_input)

        # ---------------- 计算无掩码下平均活跃 CA1 ----------------
        with torch.no_grad():
            z_orig_full = (effective_weights @ patterns.T) >= self.threshold[:, None]  # (n_memory, n_patterns)
            z_orig_full = z_orig_full.to(torch.uint8)
            avg_active = max(float(z_orig_full.sum(dim=0).float().mean()), eps)

        # ---------------- 初始化结果 ----------------
        mean_rel_diss = []

        for mask_ratio in tqdm(mask_ratios, desc="Mask Ratios"):
            # 累加所有 repeats 的相对不相似度
            rel_diss_accum = []

            for _ in range(n_repeats):
                # 构造掩码：True 保留，False 遮挡
                if mask_ratio > 0:
                    mask = (torch.rand((n_patterns, n_input), device=device) > mask_ratio).float()
                    patterns_masked = patterns * mask
                else:
                    patterns_masked = patterns

                # 批量矩阵乘法
                for i in range(0, n_patterns, batch_size):
                    batch_patterns = patterns_masked[i:i+batch_size]  # (B, n_input)
                    B = batch_patterns.shape[0]

                    with torch.no_grad():
                        z_masked = (effective_weights @ batch_patterns.T) >= self.threshold[:, None]  # (n_memory, B)
                        z_masked = z_masked.to(torch.uint8)

                    # 对应模式在 z_orig_full 中的列索引
                    idx_range = torch.arange(i, i+B, device=device)
                    z_orig_batch = z_orig_full[:, idx_range]  # (n_memory, B)

                    # 海明距离并归一化
                    hd = (z_orig_batch != z_masked).float().sum(dim=0)  # 每个模式
                    rel = (hd / avg_active).cpu().numpy()
                    rel_diss_accum.append(rel)

            # 汇总平均
            rel_diss_all = np.concatenate(rel_diss_accum)
            mean_rel_diss.append(rel_diss_all.mean())

        return np.array(mean_rel_diss)
