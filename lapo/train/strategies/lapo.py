"""LAPoStrategy —— LAPo 训练策略。

复用 SBVLAStrategy 的 vlm 加载 + image_projection 修复 + preprocess 逻辑。
简化 delta_timestamps 为 [0, H=30]（只拉当前帧 + endpoint 帧）。
"""
from __future__ import annotations

import logging
import math

import torch

from lapo.train.strategy import TrainStrategy

logger = logging.getLogger(__name__)


class LapoStrategy(TrainStrategy):
    """LAPo Stage 1 训练策略。"""

    # ============================================================
    # delta_timestamps: 只拉 [0, H] 共 2 帧
    # ============================================================
    def build_delta_timestamps(self, ds_meta):
        """拉 2 帧：当前帧 (idx=0) + endpoint 帧 (idx=H)。

        action 拉 chunk_size 个连续帧。
        """
        chunk = self.cfg.policy_overrides.get("chunk_size", 30)
        fps = ds_meta.fps
        H = self.cfg.policy_overrides.get("horizon", 30)
        obs_keys = [k for k in ds_meta.features if k.startswith("observation.")]
        dt = {}
        frame_indices = [0, H]  # [当前, endpoint]
        for k in obs_keys:
            dt[k] = [i / fps for i in frame_indices]
        dt["action"] = [i / fps for i in range(chunk)]
        return dt

    # ============================================================
    # build_policy: 加载 vlm + 构 LapoBridge + LapoPolicy
    # ============================================================
    def build_policy(self, cfg, ds_meta):
        import torch
        from lapo.train.compat import build_policy_for
        from lapo.train.policies.lapo.config import LapoConfig
        from lapo.train.policies.lapo.model import LapoBridge
        from lapo.train.policies.lapo.policy import LapoPolicy

        overrides = dict(cfg.policy_overrides)
        base_model = overrides.get("base_model", "lerobot/xvla-base")

        # 1. 加载标准 xvla policy（拿 Florence2 + action_space）
        xvla_overrides = {
            "base_model": base_model,
            "action_mode": overrides.get("action_mode", "ee6d"),
            "max_action_dim": overrides.get("max_action_dim", 32),
            "empty_cameras": overrides.get("empty_cameras", 1),
            "dtype": overrides.get("dtype", "float32"),
        }
        xvla_policy = build_policy_for(
            "xvla", self.registry, ds_meta,
            overrides=xvla_overrides,
            rename_map=cfg.dataset.rename_map,
        )
        vlm = xvla_policy.model.vlm
        action_space = xvla_policy.model.action_space

        # 1.5 修复 image_projection（复用 SBVLAStrategy 的逻辑）
        self._fix_image_projection(vlm, base_model)

        # 2. 构 LapoConfig
        lapo_cfg = LapoConfig(
            chunk_size=xvla_policy.config.chunk_size,
            dim_action=action_space.dim_action,
            max_action_dim=overrides.get("max_action_dim", 32),
            dim_proprio=xvla_policy.config.max_state_dim,
            florence_hidden=getattr(vlm.config, "projection_dim", 1024),
            action_mode=overrides.get("action_mode", "ee6d"),
            dtype=overrides.get("dtype", "float32"),
        )
        # 通用透传
        for k, v in overrides.items():
            if hasattr(lapo_cfg, k) and not k.startswith("_"):
                cur = getattr(lapo_cfg, k)
                try:
                    if isinstance(cur, bool):
                        setattr(lapo_cfg, k, bool(v))
                    elif isinstance(cur, int):
                        setattr(lapo_cfg, k, int(v))
                    elif isinstance(cur, float):
                        setattr(lapo_cfg, k, float(v))
                    else:
                        setattr(lapo_cfg, k, v)
                except (ValueError, TypeError):
                    setattr(lapo_cfg, k, v)

        logger.info(f"[lapo] decoder={lapo_cfg.decoder} endpoint_source={lapo_cfg.endpoint_source} "
                    f"horizon={lapo_cfg.horizon} cond_dim={lapo_cfg.cond_dim} "
                    f"dim_action={lapo_cfg.dim_action}")

        # 3. 构 LapoBridge（注入冻结 Florence2）
        head = LapoBridge(lapo_cfg, vlm, action_space)
        # 阶段化冻结
        self._apply_stage_freezing(head, lapo_cfg)

        # 4. 包成 LapoPolicy
        policy = LapoPolicy(lapo_cfg, head, xvla_config=xvla_policy.config)
        dtype = torch.float32 if lapo_cfg.dtype == "float32" else getattr(torch, lapo_cfg.dtype)
        policy = policy.to(dtype=dtype)
        return policy

    @staticmethod
    def _apply_stage_freezing(head, cfg):
        """阶段化冻结参数。

        Florence2 恒冻（已在 LapoBridge.__init__ 处理）。
        Stage 2 (decoder=sb): 冻 encoder + condition_encoder（只训 SB）
        Stage 3 (endpoint_source=predictor): 冻 encoder + condition_encoder + decoder（只训 predictor）
        Stage 4 (endpoint_source=joint): 冻 encoder（微调 predictor + bridge + condition_encoder）
        """
        # Florence2 已在 head.__init__ 冻结
        if cfg.endpoint_source == "predictor":
            # Stage 3: 只训 endpoint_predictor + h_proj
            for name in ["encoder", "condition_encoder", "decoder"]:
                mod = getattr(head, name, None)
                if mod is not None:
                    for p in mod.parameters():
                        p.requires_grad_(False)
        elif cfg.endpoint_source == "joint":
            # Stage 4: 只冻 encoder, 微调 predictor + bridge + condition_encoder
            for name in ["encoder"]:
                mod = getattr(head, name, None)
                if mod is not None:
                    for p in mod.parameters():
                        p.requires_grad_(False)
        elif cfg.decoder == "sb":
            # Stage 2: 冻 encoder + condition_encoder（只训 SB bridge）
            for name in ["encoder", "condition_encoder"]:
                mod = getattr(head, name, None)
                if mod is not None:
                    for p in mod.parameters():
                        p.requires_grad_(False)

    @staticmethod
    def _fix_image_projection(vlm, base_model: str):
        """复用 SBVLAStrategy._fix_image_projection 的逻辑。"""
        from pathlib import Path
        from safetensors.torch import load_file
        from lapo.train.compat import _resolve_base_model

        resolved = _resolve_base_model(base_model)
        ckpt_dir = Path(resolved)
        model_file = ckpt_dir / "model.safetensors" if ckpt_dir.is_dir() else None
        if model_file is None or not model_file.exists():
            return

        sd = load_file(str(model_file))
        fixed = []
        if "model.vlm.image_projection" in sd and hasattr(vlm, "image_projection"):
            w = sd["model.vlm.image_projection"]
            if vlm.image_projection.shape == w.shape:
                vlm.image_projection.data.copy_(w.to(vlm.image_projection.dtype))
                fixed.append(f"image_projection(norm={w.norm():.2f})")
        for suffix in ["weight", "bias"]:
            k = f"model.vlm.image_proj_norm.{suffix}"
            mod_norm = getattr(vlm, "image_proj_norm", None)
            if k in sd and mod_norm is not None:
                w = sd[k]
                attr = getattr(mod_norm, suffix)
                if attr.shape == w.shape:
                    attr.data.copy_(w.to(attr.dtype))
                    fixed.append(f"image_proj_norm.{suffix}")
        if fixed:
            import sys
            print(f"[lapo] 视觉投影已修复（注入预训练权重）: {', '.join(fixed)}", file=sys.stderr)

    # ============================================================
    # preprocess / compute_loss / on_step_end
    # ============================================================
    def preprocess(self, batch):
        from lapo.train.policies.xvla_tokenizer import tokenize_language_xvla
        rename_map = getattr(self.cfg.dataset, "rename_map", None)
        batch = tokenize_language_xvla(batch, rename_map=rename_map)
        return batch

    def compute_loss(self, policy, batch):
        batch = self.preprocess(batch)
        out = policy.forward(batch)
        if isinstance(out, (tuple, list)) and len(out) >= 2:
            self._last_log_dict = out[1]
            return out[0]
        self._last_log_dict = {}
        return out

    def on_step_end(self, step, ctx):
        # Stage 4: 更新 teacher forcing p_oracle schedule
        # 注意: ctx.policy 可能是 DDP wrapper, 必须解包到 LapoBridge 才能拿到
        # _current_p_oracle 属性 (普通 Python attribute, DDP 不会转发)
        policy = ctx.policy
        while hasattr(policy, "module"):   # 剥 DDP / FSDP wrapper
            policy = policy.module
        head = policy.model if hasattr(policy, "model") else policy
        if hasattr(head, "_update_p_oracle"):
            head._train_step = step
            head._total_steps = self.cfg.training.num_steps
            head._update_p_oracle()

        log_dict = getattr(self, "_last_log_dict", None) or {}
        if isinstance(ctx.metrics, dict):
            for k, v in log_dict.items():
                if k == "loss":
                    continue
                ctx.metrics[k] = v.detach().item() if torch.is_tensor(v) else float(v)

    # ============================================================
    # optimizer / scheduler（复用 SB 的模式）
    # ============================================================
    def build_optimizer(self, policy):
        """按阶段分配不同 lr:
        Stage 1: encoder(高) + condition_encoder(高) + decoder(高) → 统一 lr
        Stage 2: 只 SB bridge → 统一 lr (encoder/condition 冻了)
        Stage 3: 只 predictor → 统一 lr
        Stage 4: predictor(高) + bridge(低) + condition_encoder(低) → 分组
        """
        t = self.cfg.training
        raw = policy.model if hasattr(policy, "model") else policy
        ep = getattr(raw, "cfg", None)

        # Stage 4: 分参数组 (predictor 要多学, bridge 只微调)
        if ep is not None and getattr(ep, "endpoint_source", "") == "joint":
            groups = []
            # predictor + h_proj: 全 lr (它们要从 Stage 3 继续学)
            pred_params = []
            for name in ["endpoint_predictor", "h_proj"]:
                mod = getattr(raw, name, None)
                if mod is not None:
                    pred_params.extend([p for p in mod.parameters() if p.requires_grad])
            # bridge + condition_encoder: 低 lr (已训好, 只微调适配)
            bridge_params = []
            for name in ["decoder", "condition_encoder"]:
                mod = getattr(raw, name, None)
                if mod is not None:
                    bridge_params.extend([p for p in mod.parameters() if p.requires_grad])
            mult = getattr(ep, "joint_lr_mult", 0.1)
            groups = [
                {"name": "predictor", "params": pred_params, "lr": t.lr},
                {"name": "bridge", "params": bridge_params, "lr": t.lr * mult},
            ]
            logger.info(f"[lapo] Stage 4 optimizer: predictor lr={t.lr}, bridge lr={t.lr*mult}")
            return torch.optim.AdamW(
                groups, lr=t.lr, weight_decay=t.weight_decay,
                betas=(t.adam_beta1, t.adam_beta2), eps=1e-8)

        # 其他 Stage: 统一 lr
        params = [p for p in policy.parameters() if p.requires_grad]
        return torch.optim.AdamW(
            params, lr=t.lr, weight_decay=t.weight_decay,
            betas=(t.adam_beta1, t.adam_beta2), eps=1e-8)

    def build_scheduler(self, optimizer):
        t = self.cfg.training
        warmup = t.scheduler_warmup_steps
        decay = t.scheduler_decay_steps
        if warmup == 0 and decay == 0:
            return None
        decay_lr = t.scheduler_decay_lr
        peak_lr = t.lr

        num_steps = t.num_steps
        if num_steps < decay:
            warmup = max(1, int(warmup * num_steps / decay))
            decay = num_steps

        def lr_lambda(step):
            if step < warmup:
                return (step + 1) / (warmup + 1)
            progress = (step - warmup) / max(1, decay - warmup)
            progress = min(1.0, progress)
            cos_val = 0.5 * (1 + math.cos(math.pi * progress))
            return decay_lr / peak_lr + (1 - decay_lr / peak_lr) * cos_val

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
