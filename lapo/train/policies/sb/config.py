"""SBVLAConfig —— SB-VLA policy 的配置 dataclass。

所有超参对齐参考实现的 ground-truth 值（LeWM / Chronos 源码）：
  - latent dim 192  : LeWM encoder CLS token 维度
  - chunk_size 32   : X-VLA 默认动作窗口（30fps ≈ 1 秒）
  - K_imle 4        : IMLE 每帧抽 4 个候选，各认领一个 mode
  - N_steps 5       : Chronos 推理 Euler 积分步数（3-5）
  - sigma_peak 0.48 : Chronos 噪声包络峰值 16*0.03=0.48（源码 sigma_t=16*0.03*(t(1-t))²）
  - gamma_sigreg    : SIGReg 权重，LeWM 默认 1.0（防坍塌主项）
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SBVLAConfig:
    """SB-VLA policy 配置。

    phase (1/2) 控制两阶段训练，通过 policy_overrides.phase 传入：
      phase=1: 训 encoder + g + SIGReg（冻结 Florence2）
      phase=2: 冻结 encoder/g/Florence2，只训 SB 生成器
    """

    # ---- 训练阶段 ----
    phase: int = 1

    # ---- 维度 ----
    chunk_size: int = 32              # 动作窗口 H
    dim_latent: int = 192             # 物理 latent z 维度（LeWM CLS）
    dim_action: int = 10              # ee6d 真实动作维（xyz3+rot6d+grip1）
    max_action_dim: int = 32          # model-facing 动作维（pad 兼容预训练，对齐 XVLA）
    dim_proprio: int = 20             # 本体 proprio（pad 到 max_state_dim）
    florence_hidden: int = 1024       # Florence2 projection_dim（视觉特征维）

    # ---- Encoder（DaViT 冻结 → 投影头）----
    enc_depth: int = 4                # encoder transformer 层数
    enc_heads: int = 4
    enc_mlp_ratio: float = 4.0
    num_sigreg_proj: int = 1024       # SIGReg 随机投影数（Cramér-Wold）

    # ---- g（act-free 世界模型）----
    gf_depth: int = 4
    gf_heads: int = 4
    gf_mlp_ratio: float = 4.0

    # ---- g 多 horizon 监督（破恒等映射）----
    # g 同时预测多个未来时刻的 latent。单点监督下 act-free g 的最优解是抄当前 z_t
    # （未来和现在很像）；多点轨迹监督下，抄当前 = 输出常数序列，在远点 loss 累积爆炸，
    # g 被逼输出随 h 变化的轨迹 = 学 dynamics。
    # horizons: g 监督的未来帧索引（帧，非秒）。h=30=1秒=chunk 末点 = SB 到达目标。
    # horizon_weights: 各 horizon loss 权重，h=30 最重（主角）。
    # z_goal_horizon: 推理时取哪个 horizon 当 SB 的 z_goal（=1秒，语义不变）。
    horizons: tuple = (15, 30, 45, 60)              # 帧（≈0.5/1/1.5/2 秒 @30fps）
    horizon_weights: tuple = (0.25, 1.0, 0.25, 0.25)  # h=30 主角
    z_goal_horizon: int = 30                          # 推理 z_goal 取此（=1秒）

    # ---- SB 桥（Chronos）----
    bridge_hidden: int = 128          # AccField 隐维
    K_imle: int = 4                   # IMLE 候选数（多模态 mode 数）
    N_steps: int = 5                  # 推理 Euler 积分步数
    sigma_peak: float = 0.48          # 噪声包络峰值 = 16 * 0.03
    k_p: float = 4.0                  # 桥力 PD 控制（位置项，Chronos 默认 4.0）
    k_d: float = 4.0                  # 桥力 PD 控制（速度项，Chronos 默认 4.0）

    # ---- 损失权重 ----
    gamma_sigreg: float = 1.0         # SIGReg 权重（阶段1）
    lambda_acc: float = 0.1           # L_force 权重（阶段2，加速度场回归）

    # ---- loss_g 三项：方向 + 距离(残差gate) + 差异化 ----
    # 原 loss_g 是单一 MSE，方向/幅度两个目标在 192 维里打架 → 震荡不收敛。
    # 拆成三项协同：
    #   ① 方向 L_dir = (1-cos)：保护绝对方向，cos 高时贡献趋零，退化时立刻拉住。
    #   ② 距离 L_fit = cos.detach()·MSE：残差 gate 调制的分量逼近。方向好(cos高)时
    #      权重拉满去打残差精度，方向差时让位给方向项。detach 防 gate 作弊。
    #   ③ 差异化 L_div = (1-cos(Δg,Δtar))：增量方向对齐。抄当前→Δg=0→方向乱→罚。
    #      不强制 z_goal 偏离 z_t（那会和逼近 z_tar 冲突，因 z_tar≈z_t），只约束增量方向。
    w_loss_dir: float = 1.0           # ① 方向项 (1-cos) 权重（方向是地基，权重要够）
    w_loss_fit: float = 1.0           # ② 距离项 MSE 权重（静态，不用 gate）
    w_loss_div: float = 0.3           # ③ 差异化项（增量方向对齐）权重（降权，防压过方向项）

    # ---- VICReg（破 g 恒等映射坍缩）----
    # act-free g 的最优解是抄当前 z_t（输出常数序列），mag_g 跨样本趋零。
    # VICReg 方差项直接罚：g 输出每维度跨 batch 标准差 < gamma_vic → 惩罚。
    # 逼 g 输出在 batch 内 spread 开，不能所有样本抄同样的"几乎零"。
    # 协方差项去相关维度（锦上添花）。施加在 g 输出 z_goal_seq 上。
    gamma_vic: float = 1.0            # VICReg 方差目标 γ（std 阈值，VICReg 默认 1）
    lambda_vic_var: float = 1.0       # 方差项权重（主力）
    lambda_vic_cov: float = 0.02      # 协方差项权重（VICReg 默认 1/4d≈0.0013，这里微调）
    # cov 项尺度归一化模式（破缩尺度捷径）。
    # raw:       cov(z)（原版, scale-variant → encoder 缩 z 降 cov, 是塌缩推手）
    # sample_l2: cov(z / ‖z‖_sample)（按样本 L2 归一, 去整体尺度）
    # corr:      先 per-dim standardize 再算 cov（=相关矩阵 off-diag, 真正惩罚相关性,
    #            既不奖励缩尺度也不受单维尺度差异影响。推荐。）
    # 注: var 项恒用 raw z（守 absolute std, 职责与 cov 分离）。
    vic_cov_mode: str = "raw"

    # ---- 时序增量正则（补 VICReg 管不到的维度）----
    # VICReg 守「跨样本每维 std」，管不到「同轨迹 z_t 与 z_tar 帧间趋同」——后者是
    # dynamics 信号本身。encoder 可被 loss_g.l_fit 驱动走捷径（把 z_tar 映近 z_t 降 MSE），
    # VICReg 满足仍可发生。本项 hinge 罚 mag_tar=‖z_tar-z_t‖ 过小，堵此漏洞。
    #   mag_tar ≥ τ_temporal 时梯度为 0（达标放手，不与 loss_g 打架）。
    # τ 取在健康值（~1.1）与塌缩值（~0.8）之间 → 0.9；λ=1.0（实测 0.3 挡不住 encoder 压尺度）。
    tau_temporal: float = 0.9         # 时序增量阈值 τ（mag_tar 的 hinge 目标）
    lambda_temporal: float = 1.0      # 时序增量正则权重

    # ---- P0/P1: delta-space loss（替代 full-space, 修「g 抄当前」结构性作弊）----
    # 背景见 plan v2：z_t 与 z_tar 同源 → full cos/MSE 被 z_t 主体淹没 →
    # g 输出反向 Δg 也能让 L_dir=0.10（假象）。改在 delta 空间监督, 避开主体淹没。
    use_delta_loss: bool = True            # True=delta-space loss; False=回退到旧 full-space（ablation）
    w_dir_delta: float = 1.0               # ① 增量方向（masked cos）权重
    w_mag_ratio: float = 0.5               # ② log-ratio 幅值权重（双边对称, 尺度自适应）
    w_delta_mse: float = 0.05              # ③ delta smooth_l1 辅助权重
    w_loss_full: float = 0.0               # full-space MSE 残留（P0=0 干净诊断; ablation 可设 0.01）
    delta_mask_tau: float = 0.05           # 方向 loss mask 阈值（‖Δtar‖<此 则方向不可信, 不计）
    delta_mag_floor: float = 0.03          # log-ratio 下限（防小 target log 爆炸）
    mag_loss_clip: float = 9.0             # soft clamp 过渡起点 C（log1p: C·log1p(diff²/C)）
                                           # diff²<C 时 ≈ 精确; diff²>C 时梯度衰减不归零
                                           # C=9: ratio 0.05~0.1 区间在过渡区（非死区）

    # ---- P0: 纯 freeze encoder 诊断实验 ----
    # 单独 run: encoder.requires_grad=False, 只训 g, 看 g 能否打过 zero baseline。
    # 判据: cos_delta>0 且 improve_over_zero>0 → 问题在 encoder target-moving + full 盲区, 进 P1。
    #       否则 → 问题在 h/多模态/capacity, 先查 P1.5。
    p0_freeze_encoder: bool = False        # True=本 run 冻结 encoder（P0 诊断）
    encoder_lr_mult: float = 1.0           # P1 解冻时 encoder lr 倍率（master 建议 0.01~0.1）

    # ---- P1: delta-VICReg（对 Δz 做每维 std + 去相关, 替代标量 L_temporal hinge）----
    # 标量 hinge 可被无意义统一方向糊弄; delta-VICReg 逼 Δz 多维有真实分布。
    use_delta_vicreg: bool = False         # P1 启用（P0 不用, encoder 已 freeze）
    gamma_delta: float = 0.05              # Δz 每维 std 阈值（按 delta_std_mean 设, 非 norm）
    lambda_delta_var: float = 1.0          # delta-VICReg 方差项权重
    lambda_delta_cov: float = 0.02         # delta-VICReg 协方差项权重

    # ---- 诊断 A: g(z_t) 输入梯度通路 ----
    # master 诊断: delta_tar.detach 只切了 target 端, 但 z_t 作为 g 输入仍反传到 encoder。
    # encoder 沿这条路径的最优策略是缩小 z_t 尺度（z_t 小→delta 小→g 易拟合）→ mag_z 塌。
    # 开 = g(z_t.detach(), h), 切断 delta loss 经 g.net → z_t → encoder 的塌缩通路。
    delta_loss_freeze_zt: bool = False
    # 记录 encoder 参数梯度范数（监控用, 定位塌缩梯度来源）
    log_encoder_grad_norm: bool = True

    # ---- 杂项 ----
    action_mode: str = "ee6d"
    dtype: str = "float32"
