"""OpenArm 单臂自定义 action space：7 关节 MSE + 1 夹爪 BCE。

对齐官方 JointActionSpace（lerobot/policies/xvla/action_hub.py）的设计，
适配我们的 8 维单臂布局（7 关节 + 1 夹爪）。

为什么夹爪用 BCE 而非 MSE：
  夹爪 Rj8 是双峰分布（闭合 ~-141 占 84%，张开 ~-188 占 16%）。
  MSE 对双峰分布会回归到均值 → 永远不张开（NRMSE=1.41）。
  BCEWithLogitsLoss 把夹爪当二分类（开/合）学，强制决策。

关键设计（对齐官方 JointActionSpace）：
  - preprocess: pad 8→32（兼容预训练）+ 清零 gripper 维（防抄 proprio）
  - compute_loss: 前 7 维 MSE（关节回归），第 8 维 BCE（夹爪开合决策）
  - postprocess: trim 到 8 维（gripper sigmoid 还原交给 normalizer 统一处理）

前置条件：normalizer.normalize_action 已把 gripper target 二值化成 0/1。
  （BCEWithLogitsLoss 要求 target ∈ [0,1]）
"""
from __future__ import annotations

import torch
import torch.nn as nn

from lerobot.policies.xvla.action_hub import BaseActionSpace, register_action, _ensure_indices_valid


@register_action("openarm_gripper")
class OpenArmGripperActionSpace(BaseActionSpace):
    """OpenArm 单臂：7 关节 + 1 夹爪。

    Model-facing dim: 32（pad 兼容预训练 action proj）
    Real dim: 8（7 关节 idx 0-6 + 1 夹爪 idx 7）
    """

    REAL_DIM = 8
    dim_action = 32  # model-facing，对齐 config 的 max_action_dim
    gripper_idx = (7,)  # 第 8 维是夹爪

    JOINTS_SCALE = 1.0
    GRIPPER_SCALE = 0.1  # 对齐官方 JointActionSpace，避免 BCE 压制关节 MSE

    def __init__(self):
        super().__init__()
        self.real_dim = self.REAL_DIM
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    # ---------- pad/trim ----------

    def _pad_to_model_dim(self, x: torch.Tensor) -> torch.Tensor:
        """8 → 32（末尾补零）。若已是 32 直接返回。"""
        if x is None:
            return None
        if x.size(-1) == self.dim_action:
            return x
        if x.size(-1) != self.real_dim:
            raise ValueError(
                f"Expected last dim {self.real_dim} or {self.dim_action}, got {x.size(-1)}"
            )
        pad_shape = list(x.shape[:-1]) + [self.dim_action - self.real_dim]
        pad = x.new_zeros(pad_shape)
        return torch.cat([x, pad], dim=-1)

    def _trim_to_real_dim(self, x: torch.Tensor) -> torch.Tensor:
        """32 → 8（取前 8 维）。"""
        return x[..., : self.real_dim]

    # ---------- loss ----------

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        """前 7 维关节 MSE + 第 8 维夹爪 BCE。

        pred:   [B, T, 32] 来自 transformer
        target: [B, T, 8]  或 [B, T, 32]
               gripper 维须已是 0/1（由 normalizer 二值化保证）
        """
        pred = self._pad_to_model_dim(pred)
        target = self._pad_to_model_dim(target)
        assert pred.shape == target.shape, f"Shape mismatch: pred {pred.shape} vs target {target.shape}"
        action_dim = pred.shape[-1]
        _ensure_indices_valid(action_dim, self.gripper_idx, "gripper_idx")

        # 关节 MSE（idx 0-6）
        joints_idx = tuple(i for i in range(self.real_dim) if i not in set(self.gripper_idx))
        joints_loss = self.mse(pred[:, :, joints_idx], target[:, :, joints_idx]) * self.JOINTS_SCALE

        # 夹爪 BCE（idx 7）—— target 必须是 0/1
        g_losses = [self.bce(pred[:, :, gi], target[:, :, gi]) for gi in self.gripper_idx]
        gripper_loss = sum(g_losses) / len(self.gripper_idx) * self.GRIPPER_SCALE

        return {
            "joints_loss": joints_loss,
            "gripper_loss": gripper_loss,
        }

    # ---------- preprocess / postprocess ----------

    def preprocess(self, proprio: torch.Tensor, action: torch.Tensor, mode: str = "train"):
        """清零 gripper 维（防抄 proprio，对齐官方 JointActionSpace）。

        注意：调用方（XVLAModel.forward）传入的 proprio/action 已经被各自 pad：
          - proprio: pad 到 dim_proprio（max_state_dim，通常 20）
          - action:  pad 到 dim_action（max_action_dim，通常 32）
        所以这里只做 gripper 清零，不强制 pad（维度由调用方负责）。

        清零 gripper 维后，flow-matching 噪声注入的 action_noisy 在 gripper 维为 0，
        transformer 不从 gripper 噪声学，BCE 单独从图像学开合决策。
        """
        proprio_m = proprio.clone()
        action_m = action.clone() if action is not None else None

        # gripper_idx=(7,)，proprio 和 action 都至少 8 维，安全清零
        proprio_m[..., self.gripper_idx] = 0.0
        if action_m is not None:
            action_m[..., self.gripper_idx] = 0.0
        return proprio_m, action_m

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        """trim 32 → 8（gripper sigmoid 还原交给 normalizer.denormalize_action）。"""
        return self._trim_to_real_dim(action)
