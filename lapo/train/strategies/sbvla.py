"""SBVLAStrategy —— 把 SB-VLA head 接入训练/推理引擎。

核心职责：
  build_policy: 加载 lerobot/xvla-base（提取冻结 Florence2 + EE6DActionSpace）→
                包进 SBVLAHead → 包进 SBPolicy。
  preprocess:   语言注入（复用 tokenize_language_xvla）+ 图像 key rename。
  compute_loss: 按 phase 调 head.forward（阶段1 训 enc/g/f，阶段2 训 SB）。
  build_optimizer: 阶段化参数组（Florence 恒冻；阶段2 再冻 enc/g/f）。
  build_scheduler: 复用 xvla cosine warmup+decay。
  denormalize_action: 对接 infer_server。

设计：复用 xvla_ee6d 的预处理/调度器逻辑（同数据路径），只换 head。
"""
from __future__ import annotations

import logging

from lapo.train.strategy import TrainStrategy, StepContext

logger = logging.getLogger(__name__)


class SBVLAStrategy(TrainStrategy):
    """SB-VLA 训练策略：两阶段（enc/g/f → SB）。"""

    def required_traits(self) -> set[str]:
        return set()  # 独立 policy，不做 trait 校验

    def build_delta_timestamps(self, ds_meta):
        """拉当前帧 + g 多 horizon 监督的未来帧。

        g 预测 horizons=[15,30,45,60] 帧的未来 latent（破恒等映射）。
        数据层拉的帧序 = [0] + horizons = [0,15,30,45,60]（秒：[0,0.5,1,1.5,2]）。
        frame_idx 0 = 当前帧（z_t/h 用），frame_idx 1..4 = 4 个 horizon（z_tar 用）。
        h=30（frame_idx 2）= 1秒 = chunk 末点 = SB 到达目标（主角）。

        action 拉 [0..chunk-1]（整个动作块）。
        state 多取几帧无害（policy 取 [:, -1] = 当前帧）。
        return: dict[feature_key, list[秒]]
        """
        chunk = self.cfg.policy_overrides.get("chunk_size", 30)
        fps = ds_meta.fps
        horizons = self.cfg.policy_overrides.get(
            "horizons", [15, 30, 45, 60])  # g 监督的未来帧索引
        # observation.* 拉 [0] + horizons
        obs_keys = [k for k in ds_meta.features if k.startswith("observation.")]
        action_key = "action"
        dt = {}
        frame_indices = [0] + list(horizons)
        for k in obs_keys:
            dt[k] = [i / fps for i in frame_indices]
        dt[action_key] = [i / fps for i in range(chunk)]
        return dt

    # ============================================================
    # build_policy: 加载 Florence2 + 构 SBVLAHead
    # ============================================================
    def build_policy(self, cfg, ds_meta):
        """构建 SBPolicy。

        流程：
          1. 用 lerobot 的标准链路加载一个 xvla-base policy（只为拿冻结 Florence2）
          2. 提取 policy.model.vlm + policy.model.action_space
          3. 构 SBVLAConfig（从 policy_overrides）+ SBVLAHead(vlm, action_space)
          4. 包成 SBPolicy
        """
        import torch
        from lapo.train.compat import build_policy_for
        from lapo.train.policies.sb.config import SBVLAConfig
        from lapo.train.policies.sb.model import SBVLAHead
        from lapo.train.policies.sb.policy import SBPolicy

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

        # 1.5 强制注入 image_projection / image_proj_norm 预训练权重。
        # lerobot fork 的 modeling_florence2 用 `self.x = nn.Parameter(torch.empty(...))`
        # 游离注册 image_projection，导致 from_pretrained 加载时 key 命中却不覆盖
        # （实测加载后 norm=0，视觉全断）。这里绕过坏链路，直接从 checkpoint 文件注入。
        # Florence 视觉先验完整保留，参数保持冻结。
        self._fix_image_projection(vlm, base_model)

        # 2. 构 SBVLAConfig
        sb_cfg = SBVLAConfig(
            phase=int(overrides.get("phase", 1)),
            chunk_size=xvla_policy.config.chunk_size,
            dim_action=action_space.dim_action,
            max_action_dim=overrides.get("max_action_dim", 32),
            dim_proprio=xvla_policy.config.max_state_dim,
            florence_hidden=getattr(vlm.config, "projection_dim", 1024),
            action_mode=overrides.get("action_mode", "ee6d"),
            dtype=overrides.get("dtype", "float32"),
            # 超参透传（用户可在 policy_overrides 调）
            K_imle=int(overrides.get("K_imle", 4)),
            N_steps=int(overrides.get("N_steps", 5)),
            sigma_peak=float(overrides.get("sigma_peak", 0.48)),
            gamma_sigreg=float(overrides.get("gamma_sigreg", 1.0)),
            lambda_acc=float(overrides.get("lambda_acc", 0.1)),
        )
        # 通用透传：把 policy_overrides 里所有能匹配 SBVLAConfig 字段的项覆盖上去。
        # 上面显式构造只覆盖了少数字段，P0/P1 的 use_delta_loss/use_delta_vicreg/
        # p0_freeze_encoder/encoder_lr_mult/gamma_delta 等新字段靠这里透传。
        for k, v in overrides.items():
            if hasattr(sb_cfg, k) and not k.startswith("_"):
                # 类型推断：按现有字段的类型转换（bool 要 int→bool 特殊处理）
                cur = getattr(sb_cfg, k)
                try:
                    if isinstance(cur, bool):
                        setattr(sb_cfg, k, bool(v))
                    elif isinstance(cur, int):
                        setattr(sb_cfg, k, int(v))
                    elif isinstance(cur, float):
                        setattr(sb_cfg, k, float(v))
                    else:
                        setattr(sb_cfg, k, v)
                except (ValueError, TypeError):
                    setattr(sb_cfg, k, v)
        logger.info(f"[sbvla] delta config: use_delta_loss={sb_cfg.use_delta_loss} "
                    f"use_delta_vicreg={sb_cfg.use_delta_vicreg} "
                    f"p0_freeze={sb_cfg.p0_freeze_encoder} "
                    f"enc_lr_mult={sb_cfg.encoder_lr_mult} "
                    f"gamma_delta={sb_cfg.gamma_delta}")

        # 3. 构 head（注入冻结 Florence2）
        head = SBVLAHead(sb_cfg, vlm, action_space)
        # 应用阶段化冻结
        self._apply_phase_freezing(head, sb_cfg.phase)

        # 4. 包成 SBPolicy（传 xvla_config 以取 resize_imgs_with_padding）
        policy = SBPolicy(sb_cfg, head, xvla_config=xvla_policy.config)
        dtype = torch.float32 if sb_cfg.dtype == "float32" else getattr(torch, sb_cfg.dtype)
        policy = policy.to(dtype=dtype)
        return policy

    def _apply_phase_freezing(self, head, phase: int):
        """阶段化冻结参数。

        Florence2 恒冻（已在 SBVLAHead.__init__ 处理）。
        阶段2: 额外冻 encoder/g/SIGReg（只训 SB）。
        """
        # Florence2 已在 head.__init__ 冻结
        if phase == 2:
            for name in ["encoder", "g", "sigreg", "h_proj"]:
                mod = getattr(head, name, None)
                if mod is not None:
                    for p in mod.parameters():
                        p.requires_grad_(False)

    @staticmethod
    def _fix_image_projection(vlm, base_model: str):
        """强制从 xvla-base checkpoint 注入 image_projection / image_proj_norm。

        背景：lerobot fork 的 modeling_florence2 用 `self.image_projection =
        nn.Parameter(torch.empty(...))` 游离注册，from_pretrained 加载时 key 命中
        却不覆盖 → 实测加载后 norm=0（torch.empty 的内存恰好被零填），视觉特征全断。
        我们的 encoder 直接依赖视觉，所以必须修；X-VLA 原版靠 proprio 绕过了它。

        做法：读 checkpoint 文件的 model.vlm.image_projection[+_norm] 直接赋值，
        绕过坏掉的加载链路。Florence 视觉先验完整保留，参数保持冻结（不学）。
        """
        from pathlib import Path
        from safetensors.torch import load_file
        from lapo.train.compat import _resolve_base_model

        resolved = _resolve_base_model(base_model)
        ckpt_dir = Path(resolved)
        model_file = ckpt_dir / "model.safetensors" if ckpt_dir.is_dir() else None
        if model_file is None or not model_file.exists():
            return  # 非本地路径，无法注入（走原加载值，视觉可能断）

        sd = load_file(str(model_file))
        fixed = []
        # image_projection（游离 Parameter，直接 .data 赋值）
        if "model.vlm.image_projection" in sd and hasattr(vlm, "image_projection"):
            w = sd["model.vlm.image_projection"]
            if vlm.image_projection.shape == w.shape:
                vlm.image_projection.data.copy_(w.to(vlm.image_projection.dtype))
                fixed.append(f"image_projection(norm={w.norm():.2f})")
        # image_proj_norm（正常注册，但保险起见也注入）
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
            print(f"[sbvla] 视觉投影已修复（注入预训练权重）: {', '.join(fixed)}", file=sys.stderr)

    # ============================================================
    # preprocess: 语言注入 + 图像 rename（复用 xvla_ee6d 逻辑）
    # ============================================================
    def preprocess(self, batch):
        from lapo.train.policies.xvla_tokenizer import tokenize_language_xvla
        rename_map = getattr(self.cfg.dataset, "rename_map", None)
        batch = tokenize_language_xvla(batch, rename_map=rename_map)
        return batch

    # ============================================================
    # compute_loss: 按 phase
    # ============================================================
    def compute_loss(self, policy, batch):
        batch = self.preprocess(batch)
        out = policy.forward(batch)
        # forward 返回 (total_loss, log_dict)。
        # 暂存 log_dict 到实例，供 on_step_end 落盘（世界模型分项 loss + gain_h 健康监控）。
        if isinstance(out, (tuple, list)) and len(out) >= 2:
            self._last_log_dict = out[1]
            return out[0]
        self._last_log_dict = {}
        return out

    def on_step_end(self, step: int, ctx: StepContext) -> None:
        """注入 train_step（供 SB schedule 用）+ 记录世界模型分项指标。"""
        import torch
        # 更新 head 的训练步数（reach warmup schedule）
        head = ctx.policy.model if hasattr(ctx.policy, "model") else ctx.policy
        if hasattr(head, "_train_step"):
            head._train_step = step
            head._total_steps = self.cfg.training.num_steps
        # 把 forward 返回的 log_dict 分项写进 ctx.metrics（落盘到 metrics.jsonl）。
        # 阶段1: loss_g/loss_sigreg/gain_h（世界模型健康监控，文档 §3）
        # 阶段2: loss_imle/loss_force（SB 桥收敛）
        log_dict = getattr(self, "_last_log_dict", None) or {}
        if isinstance(ctx.metrics, dict):
            for k, v in log_dict.items():
                if k == "loss":
                    continue  # 总 loss 已由 engine 记
                try:
                    ctx.metrics[k] = v.detach().item() if torch.is_tensor(v) else float(v)
                except Exception:
                    pass

            # encoder/g grad norm（诊断: 定位 encoder 塌缩梯度来源）
            # on_step_end 现在在 backward 之后、step/zero_grad 之前 → grad 已累积完
            # DDP 包裹: ctx.policy 可能是 DistributedDataParallel, 用 .module 解包
            raw_policy = ctx.policy
            if hasattr(raw_policy, "module") and hasattr(raw_policy.module, "cfg"):
                raw_policy = raw_policy.module
            sb_cfg = getattr(raw_policy, "cfg", None)
            if getattr(sb_cfg, "log_encoder_grad_norm", False):
                def _grad_norm(mod):
                    sq, n = 0.0, 0
                    for p in mod.parameters():
                        if p.grad is not None:
                            sq += float(p.grad.detach().pow(2).sum())
                            n += 1
                    return (sq ** 0.5) if n > 0 else 0.0
                encoder = getattr(raw_policy, "encoder", None)
                if encoder is not None:
                    ctx.metrics["enc_grad_norm"] = _grad_norm(encoder)
                g_mod = getattr(raw_policy, "g", None)
                if g_mod is not None:
                    ctx.metrics["g_grad_norm"] = _grad_norm(g_mod)

    # ============================================================
    # build_optimizer: 阶段化参数组
    # ============================================================
    def build_optimizer(self, policy):
        import torch
        t = self.cfg.training
        lr, wd = t.lr, t.weight_decay
        sb_cfg = policy.model.cfg   # SBVLAConfig

        # P0: 纯 freeze encoder 诊断 → encoder 永久 requires_grad=False（不进优化器）
        if getattr(sb_cfg, "p0_freeze_encoder", False):
            for p in policy.model.encoder.parameters():
                p.requires_grad_(False)
            logger.info("build_optimizer: P0 诊断 — encoder 已冻结（requires_grad=False）")
            params = [p for p in policy.parameters() if p.requires_grad]
            return torch.optim.AdamW(
                params, lr=lr, weight_decay=wd,
                betas=(t.adam_beta1, t.adam_beta2), eps=1e-8,
            )

        # P1: 解冻 encoder 但低 lr（master 建议 lr_mult 0.01~0.1）
        mult = getattr(sb_cfg, "encoder_lr_mult", 1.0)
        if mult != 1.0:
            enc_params = list(policy.model.encoder.parameters())
            enc_ids = {id(p) for p in enc_params}
            enc_group = [p for p in enc_params if p.requires_grad]
            other_group = [p for p in policy.parameters()
                           if p.requires_grad and id(p) not in enc_ids]
            groups = [
                {"params": other_group, "lr": lr},
                {"params": enc_group, "lr": lr * mult},
            ]
            logger.info(f"build_optimizer: encoder lr_mult={mult} → enc lr={lr*mult:.2e}, other lr={lr:.2e}")
            return torch.optim.AdamW(
                groups, lr=lr, weight_decay=wd,
                betas=(t.adam_beta1, t.adam_beta2), eps=1e-8,
            )

        # 默认：SB 自有参数统一一组；Florence 已冻结不进优化器
        params = [p for p in policy.parameters() if p.requires_grad]
        return torch.optim.AdamW(
            params, lr=lr, weight_decay=wd,
            betas=(t.adam_beta1, t.adam_beta2), eps=1e-8,
        )

    # ============================================================
    # build_scheduler: 复用 xvla cosine warmup + decay
    # ============================================================
    def build_scheduler(self, optimizer):
        import math, torch
        t = self.cfg.training
        if t.scheduler_warmup_steps == 0 and t.scheduler_decay_steps == 0:
            return None
        warmup = t.scheduler_warmup_steps
        decay = t.scheduler_decay_steps
        num_steps = t.num_steps
        if num_steps < decay and decay > 0:
            scale = num_steps / decay
            warmup = int(warmup * scale)
            decay = num_steps
        peak_lr = t.lr
        decay_lr = t.scheduler_decay_lr

        def lr_lambda(current_step):
            if warmup > 0 and current_step < warmup:
                if current_step <= 0:
                    return 1 / (warmup + 1)
                frac = 1 - current_step / warmup
                return (1 / (warmup + 1) - 1) * frac + 1
            if decay > 0:
                step = min(current_step, decay)
                cosine = 0.5 * (1 + math.cos(math.pi * step / decay))
                alpha = decay_lr / peak_lr
                return (1 - alpha) * cosine + alpha
            return 1.0

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, -1)

    # ============================================================
    # denormalize_action: 对接 infer_server 部署
    # ============================================================
    def denormalize_action(self, pred):
        """ee6d 路径：action_space 内置处理，无需额外 denormalize。"""
        return pred
