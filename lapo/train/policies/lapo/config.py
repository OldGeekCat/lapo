"""LAPoConfig —— LAPo (Local Action Bridge with Endpoint Prior) 配置。

Stage 1: Oracle Endpoint + Direct Decoder
  obs_t → encoder(DaViT冻结) → z_t
  obs_{t+H} → encoder(no_grad) → e_t (oracle, detached)
  cond = ConditionEncoder([z_t, e_t, e_t-z_t])
  pred = DirectDecoder(cond) → action chunk [B, H, 10]
  loss = action-space loss（xyz + rot6d + gripper BCE + smooth + endpoint_disp）

不再有 latent prediction / delta loss / SIGReg / VICReg。
encoder 无膨胀捷径（action loss 与 z 尺度无关）。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LapoConfig:
    """LAPo policy 配置。"""

    # ---- 维度 ----
    chunk_size: int = 30              # 动作窗口 H（30fps ≈ 1秒）
    dim_latent: int = 192             # latent z 维度
    dim_action: int = 10              # ee6d 真实维（xyz3 + rot6d6 + grip1）
    max_action_dim: int = 32          # model-facing pad（兼容预训练，推理截断回 10）
    dim_proprio: int = 20             # 本体 proprio
    florence_hidden: int = 1024       # Florence2 projection_dim

    # ---- Endpoint horizon ----
    horizon: int = 30                 # H=30 帧 = 1秒（oracle endpoint 来自 obs_{t+H}）

    # ---- Encoder（复用 SB 的 Encoder 结构）----
    enc_depth: int = 4
    enc_heads: int = 4
    enc_mlp_ratio: float = 4.0

    # ---- Condition encoder ----
    # 输入 = [z_t, e_t, e_t-z_t] concat = 3 × dim_latent = 576
    cond_dim: int = 512               # condition 输出维度

    # ---- Direct decoder ----
    decoder_hidden: int = 512         # decoder MLP 隐维

    # ---- Stage 2: SB bridge ----
    # 用 Stage 1 的 ConditionEncoder 输出当 cond，接 SchrodingerBridge 生成动作
    decoder: str = "direct"           # "direct" (Stage1) | "sb" (Stage2)
    sb_hidden: int = 128              # AccField / IMLE gen 隐维
    K_imle: int = 4                   # IMLE 候选数
    N_steps: int = 5                  # 推理 Euler 积分步数
    sigma_peak: float = 0.48          # quartic 噪声包络峰值
    k_p: float = 4.0                  # PD 位置反馈
    k_d: float = 4.0                  # PD 速度反馈
    lambda_force: float = 0.1         # L_force 权重（相对 L_imle=1.0）

    # ---- Action loss 权重 ----
    w_xyz: float = 1.0                # xyz SmoothL1
    w_rot: float = 1.0                # rot6d SmoothL1
    w_grip: float = 2.0               # gripper BCE（稀疏，加权）
    w_smooth: float = 0.05            # 时序平滑（相邻帧差 MSE）
    w_endpoint_disp: float = 0.5      # xyz 累计位移 SmoothL1

    # ---- ee6d 动作布局 ----
    xyz_slice: tuple = (0, 3)         # [0:3]
    rot_slice: tuple = (3, 9)         # [3:9] rot6d
    grip_idx: int = 9                 # dim 9
    gripper_binary: bool = True       # True=BCE, False=SmoothL1

    # ---- 诊断 ----
    log_endpoint_usage: bool = True   # 每 val 步算 loss_shuffled_e / gain_endpoint

    # ---- Stage 3: Endpoint predictor ----
    # q(z_t, language, progress) → e_pred = z_t + delta
    # progress feature = proprio（关节状态，天然编码轨迹阶段）
    endpoint_source: str = "oracle"  # "oracle" (Stage1/2) | "predictor" (Stage3) | "joint" (Stage4)
    pred_hidden: int = 1024          # predictor MLP 隐维（加深加宽）
    pred_depth: int = 6              # predictor MLP 层数
    lang_dim: int = 1024             # Florence 语言特征维（h_proj 输入）
    use_progress: bool = True        # 接入 proprio 作 progress feature
    w_ep_mse: float = 1.0            # endpoint MSE
    w_ep_dir: float = 0.2            # 方向 cos
    w_ep_scale: float = 0.1          # 幅值 L1

    # ---- Stage 4: Joint fine-tune (predictor + bridge) ----
    # teacher forcing: 每步以概率 p_oracle 用 oracle endpoint, 否则用 predictor
    # p_oracle 从高到低渐退, 让 bridge 逐步适应 predictor 的噪声
    # schedule: 线性衰减, p_oracle(start) → p_oracle(end) over num_steps
    p_oracle_start: float = 0.5      # Stage 2 bridge 已训好, 不用从 1.0 开始
    p_oracle_end: float = 0.0        # 最终全用 predictor
    joint_lr_mult: float = 0.1       # 微调 lr 倍率 (相对 base lr, 防振荡)

    # ---- 杂项 ----
    action_mode: str = "ee6d"
    dtype: str = "float32"
