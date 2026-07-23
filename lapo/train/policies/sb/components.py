"""SB-VLA 组件：Encoder / GoalProposer(g)。

架构基于 LeWM (lucas-maes/le-wm) 的 ARPredictor + ConditionalBlock（AdaLN-zero）。
关键：单 encoder（无 EMA），靠 SIGReg 防坍塌（已与用户确认，忠于 LeWM 源码）。

角色分工（docs/next-gen-architecture-v3.md §2.1）：
  Encoder(DaViT 特征) → z_t              物理 latent（SIGReg 守护）
  GoalProposer g(z_t, h) → z_goal        act-free 世界模型（预测一秒后 latent）

f（JumpDynamics）已在 v3 删除：约束来自 g 的预测一致性，无需额外裁判。
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


# ============================================================
# AdaLN-zero 调制块（逐字移植 LeWM module.py ConditionalBlock + Attention）
# ============================================================

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN-zero 调制: x * (1 + scale) + shift。

    shift/scale: [B, dim] → unsqueeze 到 [B, 1, dim] 与 x [B, n, dim] 广播。
    """
    shift = shift.unsqueeze(1)
    scale = scale.unsqueeze(1)
    return x * (1 + scale) + shift


class Attention(nn.Module):
    """标准多头自注意力（对齐 LeWM module.py Attention）。"""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.dim_head = dim // heads
        self.scale = self.dim_head ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, self.dim_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, b, heads, n, dh]
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [b, heads, n, n]
        attn = attn.softmax(dim=-1)
        out = attn @ v  # [b, heads, n, dh]
        out = out.transpose(1, 2).reshape(b, n, d)
        return self.out(out)


class ConditionalBlock(nn.Module):
    """AdaLN-zero 条件 transformer block（移植 LeWM module.py）。

    用 cond embedding 通过 6 层 siLU MLP 产生 (shift, scale, gate)，
    分别调制 attention 和 FFN 的 LayerNorm 输出。gate 初始化为 0（零初始化，
    训练初期 block = 恒等，稳定训练）。
    """

    def __init__(self, dim: int, heads: int, mlp_ratio: float, cond_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        # AdaLN-zero: cond → 6*dim (shift1, scale1, gate1, shift2, scale2, gate2)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * dim),
        )
        # 零初始化（gate 项初始为 0 → block 初始为恒等）
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """x: [b, n, dim], cond: [b, cond_dim]（条件已池化成向量）."""
        shift_m, scale_m, gate_m, shift_a, scale_a, gate_a = self.adaLN(cond).chunk(6, dim=-1)
        # attention 分支
        h = modulate(self.norm1(x), shift_m, scale_m)
        h = self.attn(h)
        x = x + gate_m.unsqueeze(1) * h
        # FFN 分支
        h = modulate(self.norm2(x), shift_a, scale_a)
        h = self.ffn(h)
        x = x + gate_a.unsqueeze(1) * h
        return x


class CondTransformer(nn.Module):
    """堆叠 ConditionalBlock 的条件 transformer（encoder/g/f 共用骨架）。"""

    def __init__(self, dim: int, depth: int, heads: int, mlp_ratio: float, cond_dim: int,
                 final_norm: bool = True):
        super().__init__()
        self.blocks = nn.ModuleList([
            ConditionalBlock(dim, heads, mlp_ratio, cond_dim)
            for _ in range(depth)
        ])
        # final_norm=False 时不在末尾做 LayerNorm（Encoder 用：避免把 z 钉死在单位球面，
        # 改由 Encoder 自己的可学标准化接管，让 z 天然趋近 N(0,1)，VICReg 起步即达标）。
        self.norm = nn.LayerNorm(dim, elementwise_affine=False) if final_norm else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x, cond)
        return self.norm(x)


# ============================================================
# Encoder: DaViT(冻结) 视觉特征 → 物理 latent z（轻量投影头，可训练）
# ============================================================

class Encoder(nn.Module):
    """DaViT 预训练视觉特征 → z（物理 latent，SIGReg 守护）。

    输入: davit_feat [B, D_davit]（Florence2 DaViT 视觉塔的池化特征，冻结预训练）
    输出: z [B, D_latent]（物理 latent）

    设计：
      - DaViT 已预训练（Florence2 的视觉塔），物理细节保留好（std=2.45，区分度 220）
      - 冻结 DaViT，只训投影头（轻量，277 条够）
      - 不从零训 ViT（277 条学不动，顿悟慢且平台）
      - SIGReg 守护投影头输出防坍塌

    DaViT 特征来源：vlm.vision_tower.forward_features_unpool(img) → 全局池化
    """

    def __init__(self, dim_latent: int, dim_davit: int = 1024,
                 depth: int = 4, heads: int = 4, mlp_ratio: float = 4.0):
        super().__init__()
        self.dim_latent = dim_latent
        # 投影头：DaViT 特征 → latent（可训练，2 层 MLP）
        self.proj = nn.Sequential(
            nn.Linear(dim_davit, dim_latent * 2),
            nn.GELU(),
            nn.Linear(dim_latent * 2, dim_latent),
        )
        # 无条件 transformer（进一步加工，cond 用 dummy）。
        # final_norm=False：不在末尾硬 LayerNorm。硬 LN 把每个样本钉在单位球面(RMS≡1)，
        # 既让 mag_z 监控失真，又破坏时序区分度(不同帧的 z 被归一化后方向趋同)。
        # 改由下方可学的 z_scale 接管：初始化让 z ≈ N(0,1)（VICReg γ=1 起步即达标，
        # 不用一开始就猛猛拉），训练中靠 VICReg 维持分布、模型自由优化方向。
        self.transformer = CondTransformer(
            dim=dim_latent, depth=depth, heads=heads, mlp_ratio=mlp_ratio, cond_dim=dim_latent,
            final_norm=False,
        )
        self.dummy_cond = nn.Parameter(torch.zeros(1, dim_latent))
        nn.init.normal_(self.dummy_cond, std=0.02)
        # 可学逐维仿射：替代硬 LayerNorm。
        # 初始化时 z_scale/z_bias 设为 1/0（恒等），首次 forward 时用前几个 batch 的
        # 统计量校准（running mean/std → 把 raw 输出仿射成 ≈ N(0,1)），之后冻成固定仿射。
        # 这样起步 z 天然各维 std≈1（VICReg γ=1 达标，不猛拉），训练中靠 VICReg 维持，
        # 模型自由优化方向（不像硬 LN 强制每样本 var=1 → 破坏时序区分度）。
        self.z_scale = nn.Parameter(torch.ones(dim_latent))
        self.z_bias = nn.Parameter(torch.zeros(dim_latent))
        # 累计统计量（前 calib_batches 个 batch 累积，校准后冻结）
        self.register_buffer("run_sum", torch.zeros(dim_latent))
        self.register_buffer("run_sumsq", torch.zeros(dim_latent))
        self.register_buffer("run_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("calibrated", torch.tensor(False))
        self.calib_batches = 10  # 前 10 个 batch 累积样本，第 10 个一次性校准

    def forward(self, davit_feat: torch.Tensor) -> torch.Tensor:
        """davit_feat: [B, D_davit] → z: [B, D_latent].

        输出 z = raw · z_scale + z_bias。
        前 calib_batches 个 batch（training 模式）：累积样本到 run_sum/run_sumsq，
        满 calib_batches 后用累计统计一次性把 z_scale/z_bias 设成标准化系数
        （raw → N(0,1)），置 calibrated=True。之后仿射固定，靠 VICReg 维持分布。

        DDP 同步：各 rank 看到的数据不同，累积的统计量也不同。若各 rank 独立校准，
        两卡会有两套不同的 z 空间，VICReg 要额外花力气拉齐 → 校准的收益被抵消。
        因此满 calib_batches 后，对 run_sum/run_sumsq/run_count 做 all-reduce(SUM)，
        得到全局统计，两卡用同一套系数校准 → z 空间完全一致。
        """
        b = davit_feat.shape[0]
        x = self.proj(davit_feat)        # [B, D_lat]
        x = x.unsqueeze(1)               # [B, 1, D_lat] 当单 token 序列
        cond = self.dummy_cond.expand(b, -1)
        x = self.transformer(x, cond)
        z_raw = x[:, 0]                  # [B, D_lat] 仿射前的 raw 输出

        # 校准阶段：累积统计，满 calib_batches 后一次性校准
        if self.training and not bool(self.calibrated):
            with torch.no_grad():
                self.run_sum.add_(z_raw.detach().sum(dim=0))
                self.run_sumsq.add_(z_raw.detach().pow(2).sum(dim=0))
                self.run_count.add_(b)
                if int(self.run_count) >= self.calib_batches * b or \
                   int(self.run_count) >= 64:  # 至少累积 ~10 batch 或 64 样本
                    # DDP 同步：all-reduce 各 rank 的累计统计 → 全局一致
                    self._sync_calib_stats()
                    n = int(self.run_count)
                    mean = self.run_sum / n
                    var = (self.run_sumsq / n - mean.pow(2)).clamp(min=1e-6)
                    std = var.sqrt()
                    self.z_scale.data.copy_(1.0 / std)
                    self.z_bias.data.copy_(-mean / std)
                    self.calibrated.fill_(True)

        return z_raw * self.z_scale + self.z_bias

    def _sync_calib_stats(self):
        """DDP 下把各 rank 的累积统计 all-reduce(SUM) 成全局值，保证两卡校准一致。

        单进程（无 dist 或未 init）时 no-op，退回本 rank 统计。
        同步后 run_count 是全局总样本数，run_sum/run_sumsq 是全局总和。
        """
        import torch.distributed as dist
        if not dist.is_available() or not dist.is_initialized():
            return
        world = dist.get_world_size()
        if world == 1:
            return
        # 把 run_count 转成 float 做 all-reduce（NCCL long 支持不佳，统一用 float32）
        rc = self.run_count.float()
        dist.all_reduce(self.run_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.run_sumsq, op=dist.ReduceOp.SUM)
        dist.all_reduce(rc, op=dist.ReduceOp.SUM)
        self.run_count.fill_(int(rc.item()))


# ============================================================
# GoalProposer g: (z_t, h) → z_goal（act-free）
# ============================================================

class GoalProposer(nn.Module):
    """g(z_t, h) → z_goal_seq（act-free 多 horizon 目标提议器）。

    输入: z_t [B, D_lat], h [B, D_h]（任务意图，Florence 语言特征池化）
    输出: z_goal_seq [B, num_horizon, D_lat] —— num_horizon 个未来时刻的 latent

    多 horizon 监督（破恒等映射）：
      act-free g 单点监督下，抄当前 z_t 是最优解（未来和现在很像）。
      改成同时预测多个 horizon [h=15,30,45,60]，抄当前 = 输出常数序列，
      但真实轨迹随 h 漂移 → 常数序列在远点 loss 累积爆炸 → g 被逼学 dynamics。
      h=30 是主角（推理 z_goal 取它，SB 到达目标，=1秒）。

    实现: 增量形式 z_goal_seq = z_t + Δg（各 horizon 独立加到 z_t 上）。
      horizon embedding 让 net 区分要预测哪个时刻；Δg 一次出 num_horizon 个。
    """

    def __init__(self, dim_latent: int, dim_h: int, depth: int, heads: int,
                 mlp_ratio: float, num_horizon: int = 4):
        super().__init__()
        self.dim_latent = dim_latent
        self.dim_h = dim_h
        self.num_horizon = num_horizon
        # horizon embedding：让 net 知道在预测哪个时刻
        self.horizon_emb = nn.Embedding(num_horizon, dim_latent)
        nn.init.normal_(self.horizon_emb.weight, std=0.02)
        # condition = [z_t, h] 拼接（horizon 走单独 embedding，不进 cond）
        cond_dim = dim_latent + dim_h
        hidden = int(dim_latent * mlp_ratio)
        # 输出层出 dim_latent * num_horizon，一次出所有 horizon 的增量
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim_latent * num_horizon),
        )
        # 注册 horizon id buffer（用于查 embedding）
        self.register_buffer("horizon_ids", torch.arange(num_horizon), persistent=False)

    def forward(self, z_t: torch.Tensor, h: torch.Tensor,
                return_delta: bool = False) -> torch.Tensor:
        """z_t [B, D_lat], h [B, D_h] → z_goal_seq [B, num_horizon, D_lat].

        各 horizon 独立加到 z_t 上（增量形式）。
        return_delta=True 时额外返回 raw delta_g_seq [B, num_h, D_lat]（loss 用, 避免反推）。
        """
        b = z_t.shape[0]
        cond = torch.cat([z_t, h], dim=-1)         # [B, cond_dim]
        delta = self.net(cond)                      # [B, D_lat * num_h]
        delta = delta.view(b, self.num_horizon, self.dim_latent)  # [B, num_h, D_lat]
        # horizon embedding 作为每 horizon 的偏置（让 net 区分时刻）
        h_emb = self.horizon_emb.weight              # [num_h, D_lat]
        delta_g_seq = delta + h_emb.unsqueeze(0)     # [B, num_h, D_lat]
        z_goal_seq = z_t.unsqueeze(1) + delta_g_seq  # [B, num_h, D_lat]
        if return_delta:
            return z_goal_seq, delta_g_seq
        return z_goal_seq
