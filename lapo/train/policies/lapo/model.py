"""LAPoBridge —— Local Action Bridge with Endpoint Prior (Stage 1: Oracle Direct)。

数据流（训练）：
  ① z_t = encoder(DaViT(冻结)(obs_t))          ← 可训投影头
  ② e_t = sg(encoder(DaViT(obs_{t+H})))        ← oracle endpoint, no_grad
  ③ cond = ConditionEncoder([z_t, e_t, e_t-z_t])
  ④ pred = DirectDecoder(cond)                   ← [B, H, dim_action]
  ⑤ loss = action_loss(pred, expert_chunk)       ← action-space, 无膨胀捷径

设计要点：
  - e_t 是 detached 的，encoder 不能通过放大 z 来降 loss
  - action loss 与 z 的尺度无关 → encoder 无膨胀捷径
  - 不需要 SIGReg/VICReg/freeze_zt
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from lapo.train.policies.lapo.config import LapoConfig
from lapo.train.policies.sb.components import Encoder
from lapo.train.policies.sb.bridge import (
    AccField, _IMLEGenerator,
    cubic_spline_targets, noise_envelope, noise_envelope_dot,
)


class LAPoConditionEncoder(nn.Module):
    """[z_t, e_t, e_t-z_t] concat → cond_vec。

    rel = e_t - z_t 是关键：显式编码"到终点的位移方向"，decoder 不用自己算。
    """

    def __init__(self, z_dim: int = 192, cond_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim * 3, cond_dim),
            nn.SiLU(),
            nn.LayerNorm(cond_dim),
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.LayerNorm(cond_dim),
        )

    def forward(self, z_t: torch.Tensor, e_t: torch.Tensor) -> torch.Tensor:
        rel = e_t - z_t
        x = torch.cat([z_t, e_t, rel], dim=-1)  # [B, 3*z_dim]
        return self.net(x)                       # [B, cond_dim]


class LAPoDirectDecoder(nn.Module):
    """condition + learnable time embedding → action chunk。

    cond_vec [B, hidden] 广播到 H 个时间步，各加 time_emb。
    MLP blocks 逐时间步处理，head 输出每步 action。
    """

    def __init__(self, cond_dim: int = 512, hidden: int = 512,
                 chunk_size: int = 30, action_dim: int = 10):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.time_emb = nn.Parameter(torch.randn(chunk_size, hidden) * 0.02)
        self.cond_proj = nn.Linear(cond_dim, hidden)
        self.blocks = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
        )
        self.head = nn.Linear(hidden, action_dim)
        # 最后一层零初始化：初期输出≈0（接近 action 均值），稳定起步
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, cond_vec: torch.Tensor) -> torch.Tensor:
        b = cond_vec.shape[0]
        c = self.cond_proj(cond_vec)[:, None, :]        # [B, 1, hidden]
        x = c + self.time_emb[None, :, :]               # [B, H, hidden]
        x = self.blocks(x)
        return self.head(x)                             # [B, H, action_dim]


def lapo_action_loss(pred: torch.Tensor, target: torch.Tensor,
                     cfg: LapoConfig) -> tuple[torch.Tensor, dict]:
    """分通道 action loss。

    pred:   [B, H, dim_action] 模型输出（logits for gripper）
    target: [B, H, dim_action] 专家动作（gripper 已二值化到 0/1）
    """
    xyz_s, xyz_e = cfg.xyz_slice
    rot_s, rot_e = cfg.rot_slice
    grip_i = cfg.grip_idx

    xyz_loss = F.smooth_l1_loss(pred[..., xyz_s:xyz_e], target[..., xyz_s:xyz_e])
    rot_loss = F.smooth_l1_loss(pred[..., rot_s:rot_e], target[..., rot_s:rot_e])

    if cfg.gripper_binary:
        grip_logit = pred[..., grip_i]
        grip_target = target[..., grip_i].clamp(0, 1)
        grip_loss = F.binary_cross_entropy_with_logits(grip_logit, grip_target)
        grip_acc = ((torch.sigmoid(grip_logit) > 0.5).float() == grip_target).float().mean()
    else:
        grip_loss = F.smooth_l1_loss(pred[..., grip_i], target[..., grip_i])
        grip_acc = torch.tensor(0.0, device=pred.device)

    # 时序平滑（相邻帧差）
    smooth_loss = ((pred[:, 1:] - pred[:, :-1]) ** 2).mean()

    # xyz 累计位移（chunk 内 xyz delta 求和 = 总位移）
    pred_disp = pred[..., xyz_s:xyz_e].sum(dim=1)
    tgt_disp = target[..., xyz_s:xyz_e].sum(dim=1)
    endpoint_disp_loss = F.smooth_l1_loss(pred_disp, tgt_disp)

    total = (
        cfg.w_xyz * xyz_loss
        + cfg.w_rot * rot_loss
        + cfg.w_grip * grip_loss
        + cfg.w_smooth * smooth_loss
        + cfg.w_endpoint_disp * endpoint_disp_loss
    )

    metrics = {
        "loss_xyz": xyz_loss.detach(),
        "loss_rot": rot_loss.detach(),
        "loss_grip": grip_loss.detach(),
        "grip_acc": grip_acc.detach(),
        "loss_smooth": smooth_loss.detach(),
        "loss_endpoint_disp": endpoint_disp_loss.detach(),
        "loss_action": total.detach(),
    }
    return total, metrics


class LAPoSchrodingerBridge(nn.Module):
    """LAPo Stage 2 的动作生成器：用 ConditionEncoder 输出当 cond 的 Schrödinger Bridge。

    和 sb/bridge.py 的 SchrodingerBridge 区别：
      - cond 不再是 cat([z_t, z_goal])，而是 LAPoConditionEncoder 输出（含 e_t - z_t）
      - 用 LAPo 的 cond_dim（默认 512），不是 2*dim_latent
    底层组件（AccField / IMLE / 三次样条 / 噪声包络）完全复用 sb/bridge.py。
    """

    def __init__(self, dim_action: int, cond_dim: int, cfg: LapoConfig):
        super().__init__()
        self.cfg = cfg
        self.dim_action = dim_action
        self.acc_field = AccField(dim_action, cond_dim, hidden=cfg.sb_hidden)
        self.imle_gen = _IMLEGenerator(dim_action, cfg.dim_latent, cond_dim, hidden=cfg.sb_hidden)

    def sample(self, cond: torch.Tensor, chunk_size: int, steps: int,
               noise: torch.Tensor | None = None) -> torch.Tensor:
        """推理: Euler 积分从噪声 → clean chunk。"""
        b = cond.shape[0]
        device = cond.device
        if noise is None:
            noise = torch.randn(b, chunk_size, self.dim_action, device=device)
        q = self.imle_gen(noise, cond)
        v = torch.zeros_like(q)
        dt = 1.0 / steps
        for i in range(steps):
            t_curr = torch.full((b,), 1.0 - i * dt, device=device)
            a = self.acc_field(q, v, t_curr, cond)
            v = v + a * dt
            q = q + v * dt
        return q

    def loss(self, cond: torch.Tensor, expert_chunk: torch.Tensor) -> dict[str, torch.Tensor]:
        """IMLE + 薛定谔力（对齐 Chronos）。cond 来自 LAPoConditionEncoder。"""
        b, h, d = expert_chunk.shape
        device = expert_chunk.device
        sigma_peak = self.cfg.sigma_peak
        K = self.cfg.K_imle

        # IMLE: 抽 K 候选起点，winner-take-all
        noise = torch.randn(b, K, h, d, device=device)
        cond_K = cond.unsqueeze(1).expand(b, K, -1)
        q0_K = self.imle_gen(
            noise.reshape(b * K, h, d),
            cond_K.reshape(b * K, -1),
        ).reshape(b, K, h, d)

        q0_flat = q0_K.reshape(b * K, h * d)
        expert_flat = expert_chunk.reshape(b, h * d)
        dist = torch.cdist(q0_flat, expert_flat, p=2).reshape(b, K, b)
        best_k = dist.diagonal(dim1=1, dim2=2).argmin(dim=1)
        best_q0 = q0_K[torch.arange(b, device=device), best_k]
        L_imle = (best_q0 - expert_chunk).pow(2).mean()

        # 薛定谔力：三次样条 + PD 反馈
        t = torch.rand(b, device=device)
        t_b = t.view(-1, 1, 1).expand(-1, h, d)
        q_target, v_target, a_target = cubic_spline_targets(best_q0, expert_chunk, t_b)
        sigma = noise_envelope(t, sigma_peak).view(-1, 1, 1)
        sigma_dot = noise_envelope_dot(t, sigma_peak).view(-1, 1, 1)
        eps = torch.randn(b, h, d, device=device)
        q_noisy = q_target + sigma * eps
        v_noisy = v_target + sigma_dot * eps
        force_target = a_target + self.cfg.k_p * (q_target - q_noisy) + self.cfg.k_d * (v_target - v_noisy)
        a_pred = self.acc_field(q_noisy, v_noisy, t, cond)
        L_force = (a_pred - force_target.detach()).pow(2).mean()

        return {"imle": L_imle, "force": L_force}


class LAPoEndpointPredictor(nn.Module):
    """Stage 3: q(z_t, language, progress) → e_pred = z_t + delta。

    线性捷径 + 非线性残差:
      delta = Linear(z_t) + MLP(z_t, lang, progress)

    分析发现 z_t→delta 的线性 R²=0.87, 线性部分直接抓住大部分信号,
    MLP 只需学剩余 ~13% 的非线性修正。
    """

    def __init__(self, z_dim: int = 192, lang_dim: int = 1024,
                 progress_dim: int = 0, hidden: int = 1024, depth: int = 6):
        super().__init__()
        self.progress_dim = progress_dim
        # 线性捷径（可选，由 LAPO_LSTSQ_FIT 环境变量控制）
        # 注意: 实测 1280 样本时线性 R² 仅 0.53, 过拟合严重, 默认不用
        self.linear_shortcut = nn.Linear(z_dim, z_dim)
        nn.init.zeros_(self.linear_shortcut.weight)
        nn.init.zeros_(self.linear_shortcut.bias)
        # 语言投影
        self.lang_proj = nn.Sequential(
            nn.Linear(lang_dim, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
        )
        # progress 投影（如果有）
        if progress_dim > 0:
            self.progress_proj = nn.Sequential(
                nn.Linear(progress_dim, hidden),
                nn.SiLU(),
                nn.LayerNorm(hidden),
            )
        # 输入维: z_t + lang_hidden + (progress_hidden)
        in_dim = z_dim + hidden + (hidden if progress_dim > 0 else 0)
        # 残差块堆叠（学非线性修正）
        self.blocks = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.proj_in = nn.Linear(in_dim, hidden)
        for _ in range(depth):
            self.blocks.append(nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.SiLU(),
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
            ))
            self.norms.append(nn.LayerNorm(hidden))
        self.head = nn.Linear(hidden, z_dim)  # MLP 输出（非线性残差）
        # 零初始化 MLP head：初始非线性修正=0 → delta = linear_shortcut(z_t)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _try_load_lstsq_init(self):
        """从预计算文件加载最小二乘解到 linear_shortcut。

        文件格式 (torch.save): {'weight': [z_dim, z_dim], 'bias': [z_dim]}
        路径: 由环境变量 LAPO_LSTSQ_FIT 指定，或默认 /tmp/lapo_lstsq_fit.pt
        """
        import os
        path = os.environ.get('LAPO_LSTSQ_FIT', '/tmp/lapo_lstsq_fit.pt')
        if not os.path.exists(path):
            return
        try:
            fit = torch.load(path, map_location='cpu', weights_only=True)
            with torch.no_grad():
                self.linear_shortcut.weight.copy_(fit['weight'])
                self.linear_shortcut.bias.copy_(fit['bias'])
            print(f"[lapo] linear_shortcut 已加载最小二乘初始化: {path}", flush=True)
        except Exception as e:
            print(f"[lapo] 最小二乘初始化加载失败: {e}", flush=True)

    def forward(self, z_t: torch.Tensor, lang_emb: torch.Tensor,
                progress: torch.Tensor | None = None) -> torch.Tensor:
        # 线性部分（抓住主体信号）
        delta = self.linear_shortcut(z_t)
        # 非线性残差（MLP 学剩余修正）
        l = self.lang_proj(lang_emb)
        parts = [z_t, l]
        if self.progress_dim > 0 and progress is not None:
            parts.append(self.progress_proj(progress))
        x = self.proj_in(torch.cat(parts, dim=-1))
        for blk, norm in zip(self.blocks, self.norms):
            x = norm(x + blk(x))
        delta = delta + self.head(x)  # 线性 + 非线性残差
        return z_t + delta


def endpoint_predictor_loss(e_pred: torch.Tensor, e_oracle: torch.Tensor,
                            z_t: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """MSE + 方向 cos + 幅值 L1。"""
    mse = F.mse_loss(e_pred, e_oracle)
    pred_delta = e_pred - z_t
    true_delta = e_oracle - z_t
    dir_loss = 1.0 - F.cosine_similarity(pred_delta, true_delta, dim=-1).mean()
    scale_loss = (pred_delta.norm(dim=-1) - true_delta.norm(dim=-1)).abs().mean()
    total = mse + 0.2 * dir_loss + 0.1 * scale_loss
    return total, {
        "ep_mse": mse.detach(),
        "ep_dir_loss": dir_loss.detach(),
        "ep_scale_loss": scale_loss.detach(),
        "ep_loss": total.detach(),
    }


class LapoBridge(nn.Module):
    """LAPo Stage 1: Oracle Endpoint + Direct Decoder。

    Args:
        cfg: LapoConfig
        vlm: 已加载的 Florence2（冻结，从 lerobot/xvla-base 注入）
        action_space: lerobot action space（接口兼容，LAPo Stage1 不用它的 loss）
    """

    def __init__(self, cfg: LapoConfig, vlm: nn.Module, action_space=None):
        super().__init__()
        self.cfg = cfg
        self.chunk_size = cfg.chunk_size
        self.dim_action = cfg.dim_action

        # ---- 冻结 Florence2（复用 DaViT 视觉塔）----
        self.vlm = vlm
        self.vlm.eval()
        for p in self.vlm.parameters():
            p.requires_grad_(False)

        # ---- encoder：DaViT(冻结) 特征 → z_t（投影头可训，复用 SB Encoder）----
        self.encoder = Encoder(
            dim_latent=cfg.dim_latent, dim_davit=4096,
            depth=cfg.enc_depth, heads=cfg.enc_heads, mlp_ratio=cfg.enc_mlp_ratio,
        )

        # ---- h_proj: Florence 语言特征 → dim_latent（Stage 3 predictor 用）----
        self.h_proj = nn.Linear(cfg.florence_hidden, cfg.dim_latent)

        # ---- LAPo 专有模块 ----
        self.condition_encoder = LAPoConditionEncoder(
            z_dim=cfg.dim_latent, cond_dim=cfg.cond_dim)
        # decoder: Stage 1 = DirectDecoder, Stage 2 = SB bridge
        if cfg.decoder == "sb":
            self.decoder = LAPoSchrodingerBridge(cfg.dim_action, cfg.cond_dim, cfg)
        else:
            self.decoder = LAPoDirectDecoder(
                cond_dim=cfg.cond_dim, hidden=cfg.decoder_hidden,
                chunk_size=cfg.chunk_size, action_dim=cfg.dim_action)

        # ---- Stage 3: endpoint predictor（endpoint_source="predictor" 时启用）----
        progress_dim = cfg.dim_proprio if cfg.use_progress else 0
        self.endpoint_predictor = LAPoEndpointPredictor(
            z_dim=cfg.dim_latent, lang_dim=cfg.dim_latent,
            progress_dim=progress_dim, hidden=cfg.pred_hidden, depth=cfg.pred_depth)

        # ---- Stage 4: teacher forcing 状态 ----
        self._current_p_oracle = cfg.p_oracle_start
        self._train_step = 0
        self._total_steps = 1

    # ---- DaViT 特征提取（逐字复用 SB 的实现）----
    def _davit_features(self, image_input, image_mask):
        """两视角 DaViT 池化特征 → [B, 4096]。"""
        b = image_input.shape[0]
        feats = []
        for v in range(image_input.shape[1]):
            if image_mask[:, v].all():
                img = image_input[:, v]
                raw = self.vlm.vision_tower.forward_features_unpool(img)
                pooled = raw.mean(dim=1)
                feats.append(pooled)
            if len(feats) == 2:
                break
        if len(feats) < 2:
            dim = feats[0].shape[-1] if feats else 2048
            while len(feats) < 2:
                feats.append(torch.zeros(b, dim, device=image_input.device,
                                         dtype=image_input.dtype))
        return torch.cat(feats, dim=-1)

    def _florence_lang(self, input_ids, image_input, image_mask):
        """Florence2 语言特征池化 → h_proj → [B, dim_latent]。（Stage 3 predictor 用）

        复用 SB 的 _florence_semantic 逻辑：语言 token + 视觉对齐 → encoder → 池化。
        """
        batch_size, num_views = image_input.shape[:2]
        flat_mask = image_mask.view(-1).to(dtype=torch.bool)
        flat_images = image_input.flatten(0, 1)
        valid_images = flat_images[flat_mask]
        valid_feats = self.vlm._encode_image(valid_images)
        tokens_per_view, hidden_dim = valid_feats.shape[1:]
        image_features = valid_feats.new_zeros((batch_size * num_views, tokens_per_view, hidden_dim))
        image_features[flat_mask] = valid_feats
        image_features = image_features.view(batch_size, num_views, tokens_per_view, hidden_dim)
        inputs_embeds = self.vlm.get_input_embeddings()(input_ids)
        merged_embeds, attention_mask = self.vlm._merge_input_ids_with_image_features(
            image_features[:, 0], inputs_embeds,
        )
        enc_out = self.vlm.language_model.model.encoder(
            attention_mask=attention_mask, inputs_embeds=merged_embeds,
        )[0]
        h = enc_out.mean(dim=1)
        return self.h_proj(h)

    def _update_p_oracle(self):
        """根据当前 step 线性衰减 p_oracle (teacher forcing 渐退)。"""
        cfg = self.cfg
        if cfg.endpoint_source != "joint":
            return
        progress = self._train_step / max(1, self._total_steps)
        progress = min(1.0, progress)
        self._current_p_oracle = cfg.p_oracle_start + \
            (cfg.p_oracle_end - cfg.p_oracle_start) * progress

    # ============================================================
    # 训练 forward
    # ============================================================
    def forward(
        self,
        input_ids,
        image_input,
        image_mask,
        domain_id,          # noqa: ARG002
        proprio,            # noqa: ARG002
        action,
        image_input_tar=None,
        image_mask_tar=None,
    ) -> dict[str, torch.Tensor]:
        """训练 loss。按 cfg.endpoint_source 和 cfg.decoder 分支：

        Stage 1/2 (endpoint_source="oracle"):
          e_t = sg(encoder(obs_{t+H}))  ← oracle, detached
          decoder="direct" → action loss (Stage 1)
          decoder="sb"     → IMLE + force loss (Stage 2)

        Stage 3 (endpoint_source="predictor"):
          h = florence_lang(input_ids, obs_t)
          e_pred = predictor(z_t, h)
          e_oracle = sg(encoder(obs_{t+H}))  ← target
          loss = endpoint_predictor_loss(e_pred, e_oracle, z_t)
        """
        target_dtype = next(self.encoder.parameters()).dtype
        image_input = image_input.to(dtype=target_dtype)

        # ① z_t = encoder(DaViT(obs_t))
        davit_t = self._davit_features(image_input, image_mask)
        z_t = self.encoder(davit_t)                              # [B, dim_lat]

        # ② e_t 来源
        oracle_e = None
        if image_input_tar is not None:
            image_input_tar = image_input_tar.to(dtype=target_dtype)
            mask_tar = image_mask_tar if image_mask_tar is not None else image_mask
            with torch.no_grad():
                davit_tar = self._davit_features(image_input_tar, mask_tar)
                oracle_e = self.encoder(davit_tar)             # [B, dim_lat] detached

        # ---- Stage 3: 训 endpoint predictor ----
        if self.cfg.endpoint_source == "predictor":
            assert oracle_e is not None, "Stage 3 需要 oracle endpoint 作 target"
            h = self._florence_lang(input_ids, image_input, image_mask)
            # progress feature = proprio（关节状态，天然编码轨迹阶段）
            progress = proprio if (self.cfg.use_progress and proprio is not None) else None
            e_pred = self.endpoint_predictor(z_t, h, progress)
            ep_loss, ep_metrics = endpoint_predictor_loss(e_pred, oracle_e, z_t)
            with torch.no_grad():
                out = {**ep_metrics}
                out["loss"] = ep_loss.detach()
                out["mag_z_t"] = z_t.pow(2).mean().sqrt()
                out["mag_e"] = oracle_e.pow(2).mean().sqrt()
            return {"loss_ep": ep_loss, **out}

        # ---- Stage 4: joint fine-tune (predictor + bridge, teacher forcing) ----
        if self.cfg.endpoint_source == "joint":
            assert oracle_e is not None, "Stage 4 需要 oracle endpoint 作 target/teacher"
            h = self._florence_lang(input_ids, image_input, image_mask)
            progress = proprio if (self.cfg.use_progress and proprio is not None) else None
            e_pred = self.endpoint_predictor(z_t, h, progress)

            # teacher forcing: 每样本独立掷骰子, 决定用 oracle 还是 predictor
            p_oracle = self._current_p_oracle  # 由 strategy.on_step_end 每步更新
            use_oracle = torch.rand(z_t.shape[0], device=z_t.device) < p_oracle
            # 混合 endpoint: oracle 样本用 oracle_e(detach), 其余用 e_pred(带梯度)
            e_t = torch.where(use_oracle.unsqueeze(-1), oracle_e, e_pred)

            cond = self.condition_encoder(z_t, e_t)

            # action loss (通过 bridge, 梯度回 predictor + bridge)
            if self.cfg.decoder == "sb":
                sb_out = self.decoder.loss(cond, action)
                action_loss = sb_out["imle"] + self.cfg.lambda_force * sb_out["force"]
            else:
                pred = self.decoder(cond)
                action_loss, _ = lapo_action_loss(pred, action, self.cfg)

            # 辅助: predictor 仍要接近 oracle (防止 predictor 漂太远)
            ep_loss, ep_metrics = endpoint_predictor_loss(e_pred, oracle_e, z_t)

            total_loss = action_loss + 0.1 * ep_loss  # 小权重辅助正则

            with torch.no_grad():
                oracle_ratio = use_oracle.float().mean()
                out = {**ep_metrics}
                out["loss_action"] = action_loss.detach()
                out["loss"] = total_loss.detach()
                out["p_oracle"] = torch.tensor(p_oracle)
                out["oracle_ratio"] = oracle_ratio
                out["mag_z_t"] = z_t.pow(2).mean().sqrt()
                out["mag_e"] = oracle_e.pow(2).mean().sqrt()
            return {"loss_joint": total_loss, **out}

        # ---- Stage 1/2: oracle endpoint + decoder ----
        e_t = oracle_e if oracle_e is not None else z_t.detach()

        # ③ cond = ConditionEncoder([z_t, e_t, e_t-z_t])
        cond = self.condition_encoder(z_t, e_t)                 # [B, cond_dim]

        # ④ decoder 出动作 + loss
        if self.cfg.decoder == "sb":
            # Stage 2: SB bridge (IMLE + force)
            sb_out = self.decoder.loss(cond, action)
            total_loss = sb_out["imle"] + self.cfg.lambda_force * sb_out["force"]
            with torch.no_grad():
                out = {
                    "loss_imle": sb_out["imle"].detach(),
                    "loss_force": (self.cfg.lambda_force * sb_out["force"]).detach(),
                    "loss": total_loss.detach(),
                    "mag_z_t": z_t.pow(2).mean().sqrt(),
                    "mag_e": e_t.pow(2).mean().sqrt(),
                }
            return {"loss_sb_total": total_loss, **out}
        else:
            # Stage 1: Direct decoder + action loss
            pred = self.decoder(cond)
            total_loss, metrics = lapo_action_loss(pred, action, self.cfg)
            with torch.no_grad():
                out = {**metrics}
                out["loss"] = total_loss.detach()
                out["mag_z_t"] = z_t.pow(2).mean().sqrt()
                out["mag_e"] = e_t.pow(2).mean().sqrt()
                out["rel_norm"] = (e_t - z_t).pow(2).mean().sqrt()
            return {"loss_action_total": total_loss, **out}

    # ============================================================
    # 推理
    # ============================================================
    @torch.no_grad()
    def generate_actions(
        self,
        input_ids,
        image_input,
        image_mask,
        domain_id,          # noqa: ARG002
        proprio,            # noqa: ARG002
        steps: int = 5,
        image_input_tar=None,
        image_mask_tar=None,
    ) -> torch.Tensor:
        """推理: 生成动作块。

        endpoint 来源按 cfg.endpoint_source:
          "oracle": 有 obs_{t+H} 就用，否则 e_t = z_t
          "predictor": e_pred = predictor(z_t, florence_lang(obs_t))
        decoder 按 cfg.decoder:
          "direct": DirectDecoder 一步出
          "sb": SB bridge Euler 积分 (steps 步)
        """
        self.eval()
        target_dtype = next(self.encoder.parameters()).dtype
        image_input = image_input.to(dtype=target_dtype)

        davit_t = self._davit_features(image_input, image_mask)
        z_t = self.encoder(davit_t)

        # endpoint 来源（推理时 joint 也走 predictor — 真机没有 oracle）
        if self.cfg.endpoint_source in ("predictor", "joint"):
            h = self._florence_lang(input_ids, image_input, image_mask)
            progress = proprio if (self.cfg.use_progress and proprio is not None) else None
            e_t = self.endpoint_predictor(z_t, h, progress)
        elif image_input_tar is not None:
            image_input_tar = image_input_tar.to(dtype=target_dtype)
            mask_tar = image_mask_tar if image_mask_tar is not None else image_mask
            davit_tar = self._davit_features(image_input_tar, mask_tar)
            e_t = self.encoder(davit_tar)
        else:
            e_t = z_t

        cond = self.condition_encoder(z_t, e_t)

        # decoder 出动作
        if self.cfg.decoder == "sb":
            pred = self.decoder.sample(cond, self.cfg.chunk_size, steps)
        else:
            pred = self.decoder(cond)
        return pred  # [B, H, dim_action]

    @torch.no_grad()
    def eval_endpoint_usage(self, image_input, image_mask, action,
                            image_input_tar, image_mask_tar):
        """计算 loss_with_e / loss_shuffled_e / gain_endpoint。

        每 val 步调用一次。batch 内打乱 e_t 测试 endpoint 是否被利用。
        """
        target_dtype = next(self.encoder.parameters()).dtype
        image_input = image_input.to(dtype=target_dtype)
        image_input_tar = image_input_tar.to(dtype=target_dtype)

        davit_t = self._davit_features(image_input, image_mask)
        z_t = self.encoder(davit_t)
        davit_tar = self._davit_features(image_input_tar, image_mask_tar)
        e_t = self.encoder(davit_tar)

        # 正常 endpoint
        cond = self.condition_encoder(z_t, e_t)
        pred = self.decoder(cond)
        _, m = lapo_action_loss(pred, action, self.cfg)
        loss_with_e = m["loss_action"]

        # 打乱 endpoint（batch 内 shuffle）
        perm = torch.randperm(e_t.shape[0], device=e_t.device)
        e_shuffled = e_t[perm]
        cond_s = self.condition_encoder(z_t, e_shuffled)
        pred_s = self.decoder(cond_s)
        _, m_s = lapo_action_loss(pred_s, action, self.cfg)
        loss_shuffled_e = m_s["loss_action"]

        # gain_endpoint: 换 endpoint 对输出的影响量
        # 固定 z_t，用两个不同 e_t 看输出差异
        gain = (pred - pred_s).pow(2).mean().sqrt() / \
               ((e_t - e_shuffled).pow(2).mean().sqrt() + 1e-8)

        return {
            "loss_with_e": loss_with_e,
            "loss_shuffled_e": loss_shuffled_e,
            "endpoint_gain": gain,
            "endpoint_delta": loss_shuffled_e - loss_with_e,  # >0 = endpoint 有用
        }
