"""SIGReg —— Sketch Isotropic Gaussian Regularizer.

逐字对齐 LeWM (lucas-maes/le-wm) module.py 的 SIGReg 实现。
Epps-Pulley 高斯性检验 + Cramér-Wold 定理：若所有一维随机投影都是高斯，
则原分布是各向同性高斯。常数/坍塌 latent 不满足 → 被强惩罚。

关键实现细节（对齐原版，之前偏差导致防坍塌失效）：
  1. t ∈ [0, 3] 上 17 个 knots（不是 [0,π]）；phi = exp(-t²/2) 在此区间有效
  2. 投影矩阵 A 每次 forward 重新采样（多组随机投影覆盖各方向，Epps-Pulley 关键）
  3. 梯形积分权重 × 窗口，最后 × batch size 缩放
"""
from __future__ import annotations

import torch
from torch import nn


class SIGReg(nn.Module):
    """SIGReg: Sketch Isotropic Gaussian Regularizer（对齐 LeWM module.py）。

    Args:
        dim_latent: latent 维度 D（未直接用，投影矩阵按输入动态算）
        knots: Epps-Pulley 检验的频率采样点数（默认 17，LeWM 值）
        num_proj: 随机投影方向数（默认 1024，LeWM 值）
    """

    def __init__(self, dim_latent: int = 192, knots: int = 17, num_proj: int = 1024):
        super().__init__()
        self.num_proj = num_proj
        # 频率采样 t ∈ [0, 3]（LeWM：linspace(0, 3, knots)）
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3.0 / (knots - 1)
        # 梯形积分权重（端点 dt，内部 2*dt）
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        # 标准高斯特征函数实部 φ(t) = exp(-t²/2)（理论目标）
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)                       # [knots]
        self.register_buffer("phi", window)                # [knots]
        self.register_buffer("weights", weights * window)  # [knots] 梯形权重 × 窗口

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """计算 SIGReg loss（对齐 LeWM module.py SIGReg.forward）。

        proj: (..., D) latent，最后一维是 latent 维。
        内部展平成 (T, D) 处理（T = 所有 token 数）。
        return: 标量 loss。
        """
        d = proj.size(-1)
        flat = proj.reshape(-1, d)   # (T, D)

        # 随机投影矩阵 A ∈ R^{D × num_proj}，每次 forward 重新采样（列单位化）
        A = torch.randn(d, self.num_proj, device=flat.device, dtype=flat.dtype)
        A = A.div_(A.norm(p=2, dim=0))

        # 投影到一维：(T, D) @ (D, num_proj) = (T, num_proj)
        # 再 × t（频率）：(T, num_proj, knots)
        x_t = (flat @ A).unsqueeze(-1) * self.t

        # Epps-Pulley 统计量：
        #   实部 cos 的经验均值 vs 理论 φ（高斯特征函数实部）
        #   虚部 sin 的经验均值 vs 0
        #   mean(-3) = mean over T（token 维）
        err = (x_t.cos().mean(dim=-3) - self.phi).square() + x_t.sin().mean(dim=-3).square()
        # err: (num_proj, knots)

        # 梯形积分（× weights）+ batch 缩放（× T）
        statistic = (err @ self.weights) * flat.size(-2)   # (num_proj,)
        return statistic.mean()
