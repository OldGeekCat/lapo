"""XVLA EE6D 策略：只做语言注入，不做 action/state 归一化。

EE6D 数据已经通过 FK 转换好（xyz + rot6d + grip），官方 ee6d action space
内置 XYZ_SCALE/ROT_SCALE/夹爪 BCE，不需要我们手动归一化。

和 xvla_sp 的区别：
  - 不注入 XVLANormalizer（ee6d 数据已在正确空间）
  - 不做夹爪二值化（ee6d action space 自己处理夹爪）
  - 只做 language token 注入 + 图像 key rename
"""
from __future__ import annotations

from lapo.train.strategy import TrainStrategy


class XVLAEE6DStrategy(TrainStrategy):
    """XVLA EE6D：纯语言注入 + image rename，无手动归一化。"""

    def required_traits(self) -> set[str]:
        return {"has_soft_prompts"}

    def build_policy(self, cfg, ds_meta):
        """构建 policy（无 normalizer）。"""
        return super().build_policy(cfg, ds_meta)

    def preprocess(self, batch):
        """注入 language tokens + rename 图像 key（无归一化）。"""
        from lapo.train.policies.xvla_tokenizer import tokenize_language_xvla
        rename_map = getattr(self.cfg.dataset, "rename_map", None)
        batch = tokenize_language_xvla(batch, rename_map=rename_map)
        # ee6d 不需要手动归一化——官方 action space 内置处理
        return batch

    def compute_loss(self, policy, batch):
        """preprocess + forward 透传。"""
        batch = self.preprocess(batch)
        out = policy.forward(batch)
        main_loss = out[0] if isinstance(out, (tuple, list)) else out
        return main_loss

    def build_optimizer(self, policy):
        """和 xvla_sp 一样的三档差异学习率。"""
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
            {"params": vlm, "lr": lr * t.lr_vlm_scale, "weight_decay": wd, "name": "vlm"},
            {"params": sp, "lr": lr * t.lr_soft_prompt_scale, "weight_decay": wd, "name": "soft_prompts"},
            {"params": other, "lr": lr, "weight_decay": wd, "name": "other"},
        ]
        groups = [g for g in groups if g["params"]]
        return torch.optim.AdamW(groups, betas=(t.adam_beta1, t.adam_beta2), eps=1e-8)

    def build_scheduler(self, optimizer):
        """和 xvla_sp 一样的 cosine warmup + decay。"""
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
