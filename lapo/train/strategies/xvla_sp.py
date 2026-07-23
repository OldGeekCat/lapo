"""X-VLA Soft Prompt 策略：三档差异学习率 + cosine warmup/decay scheduler。

对齐官方 XVLAAdamWConfig + CosineDecayWithWarmupSchedulerConfig：
  - VLM 参数（name 含 'vlm'）:           lr * lr_vlm_scale（默认 0.1）
  - soft prompts（name 含 'soft_prompt'）: lr * lr_soft_prompt_scale（默认 1.0）
  - transformer / action head（其它）:    lr（全量）
VLM 组的 weight_decay 再 × 0.1（保护预训练对齐）。
betas 默认 (0.9, 0.99) 对齐官方（PyTorch 默认是 0.999）。
若配置了 scheduler_warmup_steps/decay_steps，构建 cosine warmup+decay scheduler。

draccus / strict-load 等 lerobot bug 规避在 compat.py，由 build_policy 默认
路径或 strategy 覆写处理——本策略只关心 optimizer + scheduler。
"""
from __future__ import annotations

from lapo.train.strategy import TrainStrategy


class XVLASoftPromptStrategy(TrainStrategy):
    """X-VLA 的 soft prompt 差异学习率策略。"""

    def required_traits(self) -> set[str]:
        return {"has_soft_prompts"}

    def build_policy(self, cfg, ds_meta):
        """构建 policy 并顺带构造 normalizer（从 ds_meta.stats）。

        normalizer 用于训练时 action 归一化 + 图像 ImageNet 归一化，
        推理时 action 反归一化。这是打破"复制state"捷径的关键。
        """
        # 触发自定义 action space 注册（openarm_gripper: 关节MSE + 夹爪BCE）
        # 必须在 policy 创建前 import，让 @register_action 装饰器执行注册
        import lapo.train.policies.openarm_action_space  # noqa: F401

        policy = super().build_policy(cfg, ds_meta)
        # 构造 normalizer（action mean-std，从数据集 stats）
        from lapo.train.policies.xvla_norm import XVLANormalizer
        try:
            self._normalizer = XVLANormalizer.from_ds_meta(ds_meta, action_key="action")
        except Exception as e:
            import sys
            print(f"[xvla_sp] ⚠️ 无法构造 normalizer（{e}），归一化关闭", file=sys.stderr)
            self._normalizer = None
        return policy

    def preprocess(self, batch):
        """注入 language tokens + rename 图像 key + 归一化（训练/推理共用）。

        归一化两步（打破"复制state"捷径）：
        1. 图像 ImageNet 归一化（Florence2 VLM 骨干要求）
        2. action target mean-std 归一化（各维等权 + 量纲 ≠ state）
        """
        from lapo.train.policies.xvla_tokenizer import tokenize_language_xvla
        rename_map = getattr(self.cfg.dataset, "rename_map", None)
        batch = tokenize_language_xvla(batch, rename_map=rename_map)

        norm = getattr(self, "_normalizer", None)
        if norm is not None:
            import torch
            # 图像 ImageNet 归一化（所有 image key）
            img_keys = [k for k in batch if k.startswith("observation.images.") and torch.is_tensor(batch[k])]
            batch = norm.normalize_images(batch, img_keys)
            # action target 归一化（训练 loss 才需要 action；推理时 batch 可能没 action）
            if "action" in batch and torch.is_tensor(batch["action"]):
                batch["action"] = norm.normalize_action(batch["action"])
        return batch

    def denormalize_action(self, action):
        """推理输出反归一化：归一化 action → 原始度数量纲。

        server / predict_action_chunk 出来的 action 是归一化的，
        下发给电机前必须反归一化。
        """
        norm = getattr(self, "_normalizer", None)
        if norm is None:
            return action
        return norm.denormalize_action(action)

    def compute_loss(self, policy, batch):
        """覆写：forward 前注入 language tokens + 归一化。

        XVLA 的 forward 内部已自动 sum(losses.values())：
          - joints_loss（前 7 维 MSE）
          - gripper_loss（第 8 维 BCE，见 openarm_gripper action_space）
        所以这里只需 preprocess + 透传 forward。

        注意：smooth_loss（二阶差分）已移除——flow-matching 学的是速度场
        v(x_t,t)≈action，不是直接回归 action target，约束 target 平滑度
        对模型输出无约束力，属于无效正则。
        """
        batch = self.preprocess(batch)
        out = policy.forward(batch)
        main_loss = out[0] if isinstance(out, (tuple, list)) else out
        return main_loss

    def build_optimizer(self, policy):
        import torch

        t = self.cfg.training
        lr, wd = t.lr, t.weight_decay

        vlm, sp, other = [], [], []
        for name, p in policy.named_parameters():
            if not p.requires_grad:
                continue
            low = name.lower()
            if "vlm" in low:
                vlm.append(p)
            elif "soft_prompt" in low:
                sp.append(p)
            else:
                other.append(p)

        groups = [
            {"params": vlm,   "lr": lr * t.lr_vlm_scale,
             "weight_decay": wd * 0.1, "name": "vlm"},
            {"params": sp,    "lr": lr * t.lr_soft_prompt_scale,
             "weight_decay": wd, "name": "soft_prompts"},
            {"params": other, "lr": lr,
             "weight_decay": wd, "name": "other"},
        ]
        groups = [g for g in groups if g["params"]]
        return torch.optim.AdamW(
            groups, betas=(t.adam_beta1, t.adam_beta2), eps=1e-8,
        )

    def build_scheduler(self, optimizer):
        """Cosine warmup + decay scheduler（对齐 XVLA 官方）。

        若 scheduler_warmup_steps 和 scheduler_decay_steps 都为 0，返回 None
        （恒定 lr）。否则构建 cosine schedule，并在训练步数 < decay_steps 时
        自动缩放（同 lerobot CosineDecayWithWarmupSchedulerConfig 逻辑）。
        """
        import math
        import torch

        t = self.cfg.training
        if t.scheduler_warmup_steps == 0 and t.scheduler_decay_steps == 0:
            return None

        # auto-scale：训练步数 < decay_steps 时缩放
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
            # warmup 阶段：线性升温
            if warmup > 0 and current_step < warmup:
                if current_step <= 0:
                    return 1 / (warmup + 1)
                frac = 1 - current_step / warmup
                return (1 / (warmup + 1) - 1) * frac + 1
            # decay 阶段：cosine 衰减
            if decay > 0:
                step = min(current_step, decay)
                cosine = 0.5 * (1 + math.cos(math.pi * step / decay))
                alpha = decay_lr / peak_lr
                return (1 - alpha) * cosine + alpha
            return 1.0

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, -1)
