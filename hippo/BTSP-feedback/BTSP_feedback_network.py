import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import numpy as np
import torch
from typing import Optional, Sequence

# -----------------------------
# 设置中文字体
# -----------------------------
# Windows 系统可用 "SimHei"（黑体）
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  
# 避免负号显示为方块
matplotlib.rcParams['axes.unicode_minus'] = False

import torch
import torch.nn as nn
from typing import Sequence, Optional
import numpy as np
from tqdm import tqdm

class BinaryBTSPfeedbackNetwork:
    """带反馈连接的二值 BTSP 网络（CA3 <-> CA1）"""

    def __init__(
        self,
        n_input: int,
        n_memory: int,
        fq: float,
        fp: float,
        fw: float,
        threshold: Optional[int] = None,
        fb_threshold: Optional[int] = None,
        device: Optional[str] = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.n_input = n_input
        self.n_memory = n_memory
        self.fw = fw
        self.fq = fq
        self.fp = fp

        # ------------------- 前馈 -------------------
        self.connections_ff = (torch.rand(n_memory, n_input, device=self.device) < fw).to(torch.uint8)
        self.synaptic_ff = torch.zeros(n_memory, n_input, device=self.device, dtype=torch.float32)

        # ------------------- 反馈 -------------------
        self.connections_fb = (torch.rand(n_input, n_memory, device=self.device) < fw).to(torch.uint8)
        self.synaptic_fb = torch.zeros(n_input, n_memory, device=self.device, dtype=torch.float32)

        # ------------------- CA1 阈值 -------------------
        if threshold is None:
            incoming = self.connections_ff.sum(dim=1)
            thr = (incoming.float() / 2).ceil().to(torch.int)
            thr[thr < 1] = 1
            self.threshold = thr
        else:
            thr = torch.tensor(threshold, device=self.device, dtype=torch.int)
            if thr.shape == (): thr = thr.repeat(n_memory)
            self.threshold = thr

        # ------------------- CA3 阈值（反馈激活） -------------------
        if fb_threshold is None:
            incoming_fb = self.connections_fb.sum(dim=1)
            thr_fb = (incoming_fb.float() / 2).ceil().to(torch.int)
            thr_fb[thr_fb < 1] = 1
            self.fb_threshold = thr_fb
        else:
            thr_fb = torch.tensor(fb_threshold, device=self.device, dtype=torch.int)
            if thr_fb.shape == (): thr_fb = thr_fb.repeat(n_input)
            self.fb_threshold = thr_fb

    # ------------------- 前馈 -------------------
    def forward(self, input_pattern):
        """
        前馈传播，支持批量输入
        input_pattern: (n_input,) 或 (batch, n_input)
        返回: (n_memory,) 或 (batch, n_memory)
        """
        x = torch.as_tensor(input_pattern, dtype=torch.float32, device=self.device)

        w_eff = self.synaptic_ff * self.connections_ff.float()
        y_input = x @ w_eff.t()  # 支持 (batch, n_input) @ (n_input, n_memory)

        thr = self.threshold.float()
        if thr.dim() == 1 and y_input.dim() == 2:
            thr = thr.unsqueeze(0)  # (1, n_memory)
        y = (y_input >= thr).float()
        return y


    # ------------------- 反馈 -------------------
    def feedback(self, ca1_pattern):
        """
        CA1 -> CA3，通过阈值生成 CA3 激活
        支持批量输入
        """
        y = torch.as_tensor(ca1_pattern, dtype=torch.float32, device=self.device)

        w_eff = self.synaptic_fb * self.connections_fb.float()
        ca3_input = y @ w_eff.t()  # (batch, n_memory) @ (n_memory, n_input)

        thr = self.fb_threshold.float()
        if thr.dim() == 1 and ca3_input.dim() == 2:
            thr = thr.unsqueeze(0)
        ca3_act = (ca3_input >= thr).float()
        return ca3_act


    # ------------------- BTSP 更新 -------------------
    def btsp_update_fast(self, input_pattern: torch.Tensor):
        x = input_pattern.to(torch.uint8)
        active_idx = torch.nonzero(x, as_tuple=True)[0]
        if active_idx.numel() == 0: return

        active_connections = self.connections_ff[:, active_idx].bool()
        has_active_connection = active_connections.any(dim=1)

        neuron_update_mask = (torch.rand(self.n_memory, device=self.device) < self.fq) & has_active_connection
        if not neuron_update_mask.any(): return

        neuron_indices = torch.nonzero(neuron_update_mask, as_tuple=True)[0]
        rows, cols = torch.nonzero(active_connections[neuron_indices], as_tuple=True)
        flip_mask = torch.rand(rows.shape[0], device=self.device) < 0.5
        rows = rows[flip_mask]
        cols = cols[flip_mask]
        input_indices = active_idx[cols]
        memory_indices = neuron_indices[rows]
        cur = self.synaptic_ff[memory_indices, input_indices]
        self.synaptic_ff[memory_indices, input_indices] = 1.0 - cur

    # ------------------- Hebbian 更新 -------------------
    def hebbian_update(self, ca3_pattern: torch.Tensor, ca1_pattern: torch.Tensor):
        """二值 Hebbian 更新反馈突触"""
        pre = ca1_pattern.to(torch.uint8)    # CA1
        post = ca3_pattern.to(torch.uint8)   # CA3

        # 同时活跃且存在连接的突触置为 1
        coactive = ((post.unsqueeze(1) & pre.unsqueeze(0)) & self.connections_fb.bool()).bool()  # 转 bool
        self.synaptic_fb[coactive] = 1




    # ------------------- 训练 -------------------
    @torch.no_grad()
    def train(self, patterns: torch.Tensor, batch_size: int = 32):
        """
        批量训练，支持 GPU 加速，兼容二值 BTSP + Hebbian 机制
        """
        patterns = patterns.to(self.device).float()
        n_patterns = patterns.shape[0]

        # 为避免频繁打印，建议少量输出
        for start in tqdm(range(0, n_patterns, batch_size), desc="Training (batched)"):
            batch = patterns[start:start + batch_size]  # shape: (B, n_input)

            # ------------------- BTSP 更新 -------------------
            for pat in batch:  # BTSP 机制仍需单 pattern 处理，因为有随机突触翻转
                self.btsp_update_fast(pat)

            # ------------------- 前馈 (CA3 -> CA1) -------------------
            w_eff_ff = self.synaptic_ff * self.connections_ff.float()
            y_input = torch.matmul(batch, w_eff_ff.T)
            ca1_batch = (y_input >= self.threshold.float()).float()  # (B, n_memory)

            # ------------------- Hebbian 更新 (CA1 -> CA3) -------------------
            # 对每个样本逐一更新二值 Hebbian 突触
            for ca3_pat, ca1_pat in zip(batch, ca1_batch):
                self.hebbian_update(ca3_pat, ca1_pat)

            # ------------------- 可选：反馈查看 -------------------
            if start == 0:  # 仅首批打印一次以调试
                sample_ca1 = ca1_batch[0]
                ca3_feedback = self.feedback(sample_ca1)
                print(f"反馈激活 CA3 数量: {ca3_feedback.sum().item()}, "
                    f"反馈权重总和: {self.synaptic_fb.sum().item()}")



    # ===============================================================
    # Pattern generation
    # ===============================================================
    def make_sparse_patterns(self, n_patterns: int) -> torch.Tensor:
        """生成稀疏二值输入模式"""
        if not (0.0 < self.fp <= 1.0):
            raise ValueError("fp must be in (0, 1]")
        return (torch.rand(n_patterns, self.n_input, device=self.device) < self.fp).to(torch.uint8)

    # ===============================================================
    # Utilities
    # ===============================================================
    def get_effective_weights_ff(self) -> torch.Tensor:
        """获取前馈有效权重矩阵"""
        return self.connections_ff.float() * self.synaptic_ff

    def get_effective_weights_fb(self) -> torch.Tensor:
        """获取反馈有效权重矩阵"""
        return self.connections_fb.float() * self.synaptic_fb

    def compute_relative_dissimilarity_vectorized(
        self,
        patterns: torch.Tensor,
        mask_ratios: Sequence[float],
        n_repeats: int = 10,
        batch_size: int = 128,
        eps: float = 1e-8,
    ) -> np.ndarray:
        """
        改进版：
        - 预计算 CA1 在无掩码下对所有 pattern 的输出，得到 avg_active（CA1 平均活跃数）
        - 使用快速随机掩码： mask = (torch.rand(B, n_input) > mask_ratio)
        - 完全在 device 上运算，减少 CPU-GPU 拷贝
        返回：每个 mask_ratio 对应的平均归一化海明距离（np.ndarray）
        """
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
        # 可打印或记录
        # print(f"Baseline average active CA1 per pattern (no mask): {avg_active:.4f}")

        # ---------------- 计算不同掩码下的记忆保留 ----------------
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
    
    @torch.no_grad()
    def test_reconstruction_under_mask(self, patterns, mask_ratios):
        """
        批量版本：利用 GPU 并行计算
        """
        errors = []
        patterns = patterns.float().to(self.device)

        for mask_ratio in tqdm(mask_ratios, desc="Testing different mask ratios"):
            mask = (torch.rand_like(patterns) > mask_ratio).float()
            masked_inputs = patterns * mask

            ca1 = self.forward(masked_inputs)    # (batch, n_memory)
            recon = self.feedback(ca1)           # (batch, n_input)

            diff = (patterns != recon).float()
            active_count = patterns.sum(dim=1) + 1e-9
            hamming_error = diff.sum(dim=1) / 50
            avg_error = hamming_error.mean().item()
            errors.append(avg_error)

        return errors
