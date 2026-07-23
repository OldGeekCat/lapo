"""Schrödinger Bridge —— 二阶加速度桥动作生成器（移植 Chronos）。

源码 ground-truth（yulinzhouZYL/Chronos mamba_policy_par_3D_IMLE.py）：
  - 插值: 三次样条  q_t = q0 + 3δt² − 2δt³（双零点边界 q0/v0=0 → q1/v1=0）
                 v_t = 6δt − 6δt²
                 a_t = 6δ − 12δt
  - 噪声包络: σ_t = 16·sigma_peak·(t(1−t))² / 16 = sigma_peak·(t(1−t))²
              （Chronos 源码 sigma_t = 16 * 0.03 * (t*(1-t))**2，峰值在 t=0.5）
  - 积分: 简单 Euler（非 leapfrog），v += a·dt; q += v·dt，N=3-5 步
  - IMLE: 每 batch 抽 K 个候选，torch.min 取每个 expert 的最近候选，只惩罚该对

v2 改造（相对 Chronos）：
  - 去掉 Mamba 全历史主干（用单帧 z_t + 任务意图 h）
  - condition 从 Mamba cond 换成 [z_t, z_goal]（薛定谔桥两端物理锚点）
  - 去掉 DINO/3D RoPE（我们是 2D RGB + latent）

condition 注入方式（对齐 Chronos CNNSymplecticHead 的 visual_adapter）：
  [z_t, z_goal] 拼接 → adapter → 广播到 chunk 时序 → 作为通道拼进 [q, v, t_emb]
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def cubic_spline_targets(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor):
    """三次样条插值目标（双零点边界）。

    q0, q1: [B, H, D]（起点/终点 chunk；v0=v1=0 假设零速度）
    t: [B] 或标量，桥时间 ∈ [0,1]
    return: (q_target, v_target, a_target) 各 [B, H, D]

    源码: q_t = q0 + (q1−q0)·(3δt² − 2δt³)，δt = t
    """
    delta = q1 - q0
    # 广播 t 到 chunk 维度
    t_b = t.view(-1, 1, 1) if t.dim() == 1 else t
    q_t = q0 + delta * (3 * t_b ** 2 - 2 * t_b ** 3)
    v_t = delta * (6 * t_b - 6 * t_b ** 2)
    a_t = delta * (6 - 12 * t_b)
    return q_t, v_t, a_t


def noise_envelope(t: torch.Tensor, sigma_peak: float) -> torch.Tensor:
    """quartic 噪声包络 σ_t = 16·sigma_peak·(t(1−t))²（双零点）。

    t: [B]
    双零点（σ(0)=σ(1)=0），峰值在 t=0.5 = sigma_peak。
    Chronos 源码 sigma_t = 16 * sigma_peak * (t*(1-t))**2。
    """
    return 16 * sigma_peak * (t * (1 - t)).pow(2)


def noise_envelope_dot(t: torch.Tensor, sigma_peak: float) -> torch.Tensor:
    """σ_t 的时间导数 dσ/dt（用于噪声速度 v_noisy 的注入）。

    σ_t = 16·s·(t(1−t))² = 16·s·(t² − t³)
    dσ/dt = 16·s·(2t(1−t)·(1−2t)) = 16·s·(2t − 6t² + 4t³)
    对齐 Chronos 源码的 sigma_dot_t。
    """
    return 16 * sigma_peak * (2 * t * (1 - t) * (1 - 2 * t))


class AccField(nn.Module):
    """加速度场网络（移植 Chronos CNNSymplecticHead，去 conv1d 改 MLP）。

    输入通道: [position q, velocity v, time_emb, cond(z_t+goal 广播)]
    输出: acceleration a
    Chronos 用 conv1d（处理序列），这里用 per-token MLP（chunk 已是离散动作序列，
    不需要卷积的局部性）。
    """

    def __init__(self, dim_action: int, dim_cond: int, dim_time: int = 32, hidden: int = 128):
        super().__init__()
        self.dim_time = dim_time
        # time embedding（正弦）
        self.time_mlp = nn.Sequential(
            nn.Linear(dim_time, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        # cond adapter（[z_t, goal] → hidden，广播到时序）
        self.cond_adapter = nn.Sequential(
            nn.Linear(dim_cond, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        # 主网络: [q, v, time, cond] → a
        in_dim = dim_action * 2 + hidden + hidden
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim_action),
        )
        # 零初始化最后一层（加速度初始=0 → 积分初期 = 直线，稳定）
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def time_embed(self, t: torch.Tensor) -> torch.Tensor:
        """t [B] → [B, dim_time] 正弦编码。"""
        half = self.dim_time // 2
        freqs = torch.exp(
            -torch.arange(half, device=t.device).float() / half * torch.log(torch.tensor(10000.0))
        )
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)  # [B, half]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, dim_time]

    def forward(
        self,
        q: torch.Tensor,        # [B, H, D_act] 位置
        v: torch.Tensor,        # [B, H, D_act] 速度
        t: torch.Tensor,        # [B] 桥时间
        cond: torch.Tensor,     # [B, D_cond] 条件（z_t + goal）
    ) -> torch.Tensor:
        """返回加速度 a [B, H, D_act]."""
        b, h, d = q.shape
        # time embedding [B, hidden] → 广播到 [B, H, hidden]
        t_emb = self.time_mlp(self.time_embed(t)).unsqueeze(1).expand(b, h, -1)
        # cond [B, hidden] → 广播
        c_emb = self.cond_adapter(cond).unsqueeze(1).expand(b, h, -1)
        # 拼通道
        x = torch.cat([q, v, t_emb, c_emb], dim=-1)  # [B, H, in_dim]
        return self.net(x)  # [B, H, D_act]


class SchrodingerBridge(nn.Module):
    """薛定谔桥动作生成器 + IMLE 多模态训练。

    condition = [z_t, z_goal]（latent 空间物理锚点，文档 §2.1）
    路径在 action 空间：噪声候选 chunk → 专家 chunk（桥钉死两端）
    """

    def __init__(self, dim_action: int, dim_latent: int, cfg):
        super().__init__()
        self.cfg = cfg
        self.dim_action = dim_action
        self.dim_latent = dim_latent
        # condition = [z_t, z_goal] 拼接
        dim_cond = dim_latent * 2
        self.acc_field = AccField(dim_action, dim_cond, hidden=cfg.bridge_hidden)
        # IMLE 候选生成器：z_noise → q（从噪声生成候选起点，对齐 Chronos IMLE_Generator）
        self.imle_gen = _IMLEGenerator(dim_action, dim_latent, dim_cond, hidden=cfg.bridge_hidden)

    def sample(
        self,
        z_t: torch.Tensor,       # [B, D_lat]
        z_goal: torch.Tensor,    # [B, D_lat]
        chunk_size: int,
        steps: int,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """推理: Euler 积分从噪声 → clean chunk。

        return: chunk [B, chunk_size, D_act]
        """
        b = z_t.shape[0]
        device = z_t.device
        cond = torch.cat([z_t, z_goal], dim=-1)  # [B, 2*D_lat]

        # IMLE 生成器出候选起点 q_curr（从噪声，conditioned）
        if noise is None:
            noise = torch.randn(b, chunk_size, self.dim_action, device=device)
        q = self.imle_gen(noise, cond)  # [B, H, D_act]
        v = torch.zeros_like(q)

        dt = 1.0 / steps
        # 从 t=1（噪声端）积分到 t=0（clean 端），对齐 Chronos sample_actions
        for i in range(steps):
            t_curr = torch.full((b,), 1.0 - i * dt, device=device)
            a = self.acc_field(q, v, t_curr, cond)
            v = v + a * dt
            q = q + v * dt
        return q  # clean chunk 估计 x̂_0

    def loss(
        self,
        z_t: torch.Tensor,         # [B, D_lat]
        z_goal: torch.Tensor,      # [B, D_lat]
        expert_chunk: torch.Tensor,  # [B, H, D_act] 专家动作块
        k_p: float = 4.0,          # PD 位置反馈（Chronos 默认 4.0）
        k_d: float = 4.0,          # PD 速度反馈（Chronos 默认 4.0）
        sigma_peak: float | None = None,
    ) -> dict[str, torch.Tensor]:
        """训练 loss（IMLE + 薛定谔力），对齐 Chronos compute_loss。

        return: {"imle": L_imle, "force": L_force}

        薛定谔力的 PD 反馈（平滑的关键，Chronos 源码精确形式）：
            force_target = a_target + k_p·(q_target − q_noisy) + k_d·(v_target − v_noisy)
        其中:
            q_noisy = q_target + σ_t·eps        （位置噪声）
            v_noisy = v_target + σ_dot_t·eps    （速度噪声，用包络导数）
            q_target/v_target/a_target = best_q0→expert 的三次样条（双零点边界）
        PD 项把积分轨迹拉回样条路径 → 平滑。

        边界一致性 (文档 L_BC): 三次样条自带双零点边界（q0/v0=0 → q1/v1=0），
        无独立 loss 项（对齐 Chronos，它也没有独立 L_BC）。
        """
        b, h, d = expert_chunk.shape
        device = expert_chunk.device
        if sigma_peak is None:
            sigma_peak = self.cfg.sigma_peak
        cond = torch.cat([z_t, z_goal], dim=-1)

        # ---- IMLE：抽 K 候选起点，取每个 expert 的最近候选 ----
        noise = torch.randn(b, self.cfg.K_imle, h, d, device=device)  # [B, K, H, D]
        cond_K = cond.unsqueeze(1).expand(b, self.cfg.K_imle, -1)     # [B, K, D_cond]
        q0_K = self.imle_gen(
            noise.reshape(b * self.cfg.K_imle, h, d),
            cond_K.reshape(b * self.cfg.K_imle, -1),
        ).reshape(b, self.cfg.K_imle, h, d)  # [B, K, H, D]

        # winner-take-all: 每个 expert 选最近候选起点（torch.cdist 风格，对齐 Chronos）
        q0_flat = q0_K.reshape(b * self.cfg.K_imle, h * d)            # [B*K, H*D]
        expert_flat = expert_chunk.reshape(b, h * d)                  # [B, H*D]
        dist = torch.cdist(q0_flat, expert_flat, p=2).reshape(b, self.cfg.K_imle, b)
        # 每个 expert 取最近候选（对角线 = 该 expert 的候选集合）
        best_k = dist.diagonal(dim1=1, dim2=2).argmin(dim=1)          # [B]
        best_q0 = q0_K[torch.arange(b), best_k]                       # [B, H, D]

        # L_IMLE: 最近候选起点回归到 expert（桥的 q0→q1，candidate 认领 mode）
        L_imle = (best_q0 - expert_chunk).pow(2).mean()

        # ---- 薛定谔力：加速度场回归到三次样条目标 + PD 反馈 ----
        # 采桥时间 t ∈ [0,1]
        t = torch.rand(b, device=device)                              # [B]
        t_b = t.view(-1, 1, 1).expand(-1, h, d)                       # [B, H, D]

        # 三次样条目标（best_q0 → expert，双零点边界）
        q_target, v_target, a_target = cubic_spline_targets(best_q0, expert_chunk, t_b)

        # 噪声注入（位置用 σ_t，速度用 σ_dot_t）
        sigma = noise_envelope(t, sigma_peak).view(-1, 1, 1)         # [B,1,1]
        sigma_dot = noise_envelope_dot(t, sigma_peak).view(-1, 1, 1) # [B,1,1]
        eps = torch.randn(b, h, d, device=device)                    # 共享噪声（位置+速度同源）
        q_noisy = q_target + sigma * eps
        v_noisy = v_target + sigma_dot * eps

        # PD 反馈项：把积分轨迹从 noisy 状态拉回样条路径（平滑的核心约束）
        force_target = a_target + k_p * (q_target - q_noisy) + k_d * (v_target - v_noisy)

        # 加速度场预测（输入是 noisy 状态，对齐 Chronos CNNSymplecticHead）
        a_pred = self.acc_field(q_noisy, v_noisy, t, cond)
        L_force = (a_pred - force_target.detach()).pow(2).mean()

        return {"imle": L_imle, "force": L_force}


class _IMLEGenerator(nn.Module):
    """IMLE 候选生成器：z_noise + cond → q（候选起点）。

    对齐 Chronos IMLE_Generator（1D U-Net），简化为 MLP（chunk 时序独立处理）。
    """

    def __init__(self, dim_action: int, dim_latent: int, dim_cond: int, hidden: int = 128):
        super().__init__()
        self.cond_adapter = nn.Sequential(
            nn.Linear(dim_cond, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.net = nn.Sequential(
            nn.Linear(dim_action + hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim_action),
        )

    def forward(self, noise: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """noise [B, H, D_act], cond [B, D_cond] → q [B, H, D_act]."""
        b, h, d = noise.shape
        c = self.cond_adapter(cond).unsqueeze(1).expand(b, h, -1)
        x = torch.cat([noise, c], dim=-1)
        return self.net(x)
