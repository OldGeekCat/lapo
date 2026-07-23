"""XVLA 归一化工具：action mean-std 归一化 + 夹爪二值化 + 图像 ImageNet 归一化。

解决的问题：训练时 action 不归一化导致 MSE 被大方差维度主导 + 模型走
"复制 state" 捷径（action≈state 在原始量纲下天然成立）。归一化后：
  - action 关节维 mean=0/std=1，MSE 等权
  - action(归一化) 和 state(原始) 数值量纲不同，打破复制捷径
  - 图像做 ImageNet 归一化（Florence2 VLM 骨干的要求）

夹爪二值化（配合 openarm_gripper action space 的 BCE loss）：
  - 夹爪 Rj8 是双峰（闭合 ~-141 占 84%，张开 ~-188 占 16%）
  - MSE 会回归均值 → 永远不张开
  - 归一化时按阈值二值化成 0/1，BCE 学开合决策
  - 反归一化时 sigmoid 概率 → 阈值 0.5 → 映射回固定角度

推理时 predict_action_chunk 输出需反归一化回原始度数（denormalize_action）。
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ImageNet 标准归一化常数（Florence2 / 所有 ImageNet 预训练 VLM 通用）
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class XVLANormalizer:
    """持有 action 的 mean/std，提供正/反向归一化。

    action 归一化用 mean-std（对齐 lerobot NormalizeProcessor + xvla preprocessor.json
    的 ACTION: MEAN_STD）。state 保持原始（IDENTITY，不归一化）——这是官方约定，
    且保持 state/action 量纲差异正是打破"复制state"捷径的关键。

    夹爪维（gripper_idx）特殊处理：
      normalize:   原始角度 → 二值 0/1（按阈值）
      denormalize: sigmoid 概率 → 固定角度（闭合/张开两态）
    """

    def __init__(self, action_mean, action_std,
                 gripper_idx: int = 7,
                 gripper_threshold: float = -155.0,
                 gripper_closed: float = -141.0,
                 gripper_open: float = -188.0):
        import torch
        self.action_mean = torch.as_tensor(action_mean, dtype=torch.float32)
        self.action_std = torch.as_tensor(action_std, dtype=torch.float32)
        self.gripper_idx = gripper_idx
        self.gripper_threshold = gripper_threshold   # 二值化阈值（角度）
        self.gripper_closed = gripper_closed         # 闭合态角度（≈-141）
        self.gripper_open = gripper_open             # 张开态角度（≈-188）
        logger.info(
            "XVLANormalizer: action_mean=%s action_std=%s gripper(idx=%d thr=%.1f closed=%.1f open=%.1f)",
            self.action_mean.tolist(), self.action_std.tolist(),
            gripper_idx, gripper_threshold, gripper_closed, gripper_open)

    @classmethod
    def from_ds_meta(cls, ds_meta: Any, action_key: str = "action", **kwargs) -> "XVLANormalizer":
        """从 dataset.meta 的 stats 构造（转换 v3.0 时已计算并存好）。"""
        import numpy as np
        stats = None
        if hasattr(ds_meta, "stats"):
            stats = ds_meta.stats
        if stats is None or action_key not in stats:
            raise ValueError(
                f"ds_meta.stats 里没有 '{action_key}'，无法构造 normalizer。"
                f"可用 keys: {list(stats.keys()) if stats else 'None'}")
        s = stats[action_key]
        mean = np.asarray(s["mean"], dtype=np.float32)
        std = np.asarray(s["std"], dtype=np.float32)
        std = np.maximum(std, 1e-6)  # 防 0 除
        return cls(mean, std, **kwargs)

    def normalize_action(self, action):
        """action: (..., D) tensor → (..., D) 归一化。

        关节维：mean-std 归一化
        第 gripper_idx 维：按阈值二值化（< threshold → 1.0 张开，≥ threshold → 0.0 闭合）
          注意：gripper 维是二值化，不参与 mean-std，单独处理。
        """
        import torch
        mean = self.action_mean.to(action.device, dtype=action.dtype)
        std = self.action_std.to(action.device, dtype=action.dtype)
        out = action.clone()
        if self.gripper_idx < out.shape[-1]:
            # 关节维：mean-std（跳过 gripper）
            joints_idx = [i for i in range(out.shape[-1]) if i != self.gripper_idx]
            out[..., joints_idx] = (action[..., joints_idx] - mean[joints_idx]) / std[joints_idx]
            # 夹爪维：二值化（角度 < threshold（更负=更开）→ 1.0，否则 → 0.0）
            out[..., self.gripper_idx] = (
                action[..., self.gripper_idx] < self.gripper_threshold
            ).to(action.dtype)
        else:
            out = (action - mean) / std
        return out

    def denormalize_action(self, action):
        """action: (..., D) tensor → (..., D) 反归一化回原始量纲。

        前 D-1 维：反 mean-std
        第 gripper_idx 维：raw logits（BCE 训练空间）→ 阈值 0 → 固定角度（closed/open）
          注意：gripper 维是 logits，不参与 mean-std 反归一化，单独处理。
        """
        import torch
        mean = self.action_mean.to(action.device, dtype=action.dtype)
        std = self.action_std.to(action.device, dtype=action.dtype)
        out = action.clone()
        # 关节维：反 mean-std（跳过 gripper 维）
        if self.gripper_idx < out.shape[-1]:
            joints_idx = [i for i in range(out.shape[-1]) if i != self.gripper_idx]
            out[..., joints_idx] = action[..., joints_idx] * std[joints_idx] + mean[joints_idx]
            # 夹爪维：logits ≥ 0 → 张开（-188）；logits < 0 → 闭合（-141）
            g = action[..., self.gripper_idx]
            open_mask = g >= 0.0
            out[..., self.gripper_idx] = torch.where(
                open_mask, g.new_full((), self.gripper_open), g.new_full((), self.gripper_closed))
        else:
            out = action * std + mean
        return out

    def normalize_images(self, batch: dict, image_keys: list[str]) -> dict:
        """对 batch 里的图像做 ImageNet 归一化（原地修改）。

        输入期望 [0,1] float（lerobot dataset 默认输出），输出 ImageNet 归一化。
        Florence2 VLM 骨干要求此归一化。
        """
        import torch
        mean = torch.tensor(IMAGENET_MEAN, device=next(iter(batch.values())).device
                            if batch else "cpu").view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=mean.device).view(1, 3, 1, 1)
        for k in image_keys:
            if k in batch and torch.is_tensor(batch[k]):
                img = batch[k]
                # 支持 (B, C, H, W) 和 (B, T, C, H, W)
                if img.ndim == 4:
                    m, s = mean, std
                elif img.ndim == 5:
                    m, s = mean.unsqueeze(0), std.unsqueeze(0)
                else:
                    continue
                batch[k] = (img - m) / s
        return batch
