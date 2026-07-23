"""LAPoPolicy —— LAPo 的 lerobot 兼容包装层。

复刻 SBPolicy 的图像/state/language 准备逻辑，但简化为：
  - 只拉 2 帧：当前帧 obs_t (frame_idx=0) + endpoint 帧 obs_{t+H} (frame_idx=1)
  - 不需要多 horizon，不需要 delta_timestamps 拉 5 帧
  - forward 直接返回 action loss

batch 输入契约（与 ee6d 路径一致）：
  observation.language.tokens     [B, 64]   BART token id
  observation.images.image2/image3 [B, 2, C, 480, 640]（2 帧 = 当前 + endpoint）
  observation.state               [B, 2, dim_proprio]
  action                          [B, chunk_size, dim_action]
"""
from __future__ import annotations

from collections import deque

import torch
from torch import Tensor

from lerobot.policies.xvla.modeling_xvla import (
    OBS_LANGUAGE_TOKENS, OBS_STATE, ACTION,
    pad_vector, pad_tensor_along_dim, resize_with_pad, populate_queues,
)


class LapoPolicy(torch.nn.Module):
    """LAPo policy：包装 LapoBridge，暴露 lerobot 兼容接口。"""

    def __init__(self, config, head, xvla_config=None):
        super().__init__()
        self.config = config
        self.model = head
        self.image_features = ["observation.images.image", "observation.images.image2", "observation.images.image3"]
        self.num_image_views = 3
        if xvla_config is not None and getattr(xvla_config, "resize_imgs_with_padding", None):
            self.resize_imgs_with_padding = list(xvla_config.resize_imgs_with_padding)
        else:
            self.resize_imgs_with_padding = [224, 224]
        self.n_action_steps = config.chunk_size
        self.reset()

    def reset(self) -> None:
        self._queues = {ACTION: deque(maxlen=self.n_action_steps)}

    @property
    def dim_action(self):
        return self.model.dim_action

    # ============================================================
    # 输入准备
    # ============================================================
    def _build_model_inputs(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """构建模型输入。当前帧 + endpoint 帧（H=30）。

        数据层 build_delta_timestamps 拉 [0, H] 共 2 帧。
        frame_idx 0 = 当前帧（z_t 用），frame_idx 1 = endpoint 帧（e_t 用）。
        """
        input_ids = batch[OBS_LANGUAGE_TOKENS]
        batch_size = input_ids.shape[0]
        images_t, mask_t = self._prepare_images_frame(batch, frame_idx=0)
        domain_id = self._get_domain_id(batch, batch_size, images_t.device)
        proprio = self._prepare_state(batch, batch_size, images_t.device)

        # endpoint 帧：frame_idx=1（数据层拉的帧序 = [0, H]）
        any_img_key = next(k for k in self.image_features if k in batch)
        if batch[any_img_key].ndim == 5:  # [B, n_frames, C, H, W]
            images_tar, mask_tar = self._prepare_images_frame(batch, frame_idx=1)
        else:
            # 单帧（推理/兼容）：退化为当前帧（e_t = z_t）
            images_tar, mask_tar = images_t, mask_t

        return {
            "input_ids": input_ids,
            "image_input": images_t,
            "image_mask": mask_t,
            "image_input_tar": images_tar,   # endpoint 帧 [B, n_view, C,H,W]
            "image_mask_tar": mask_tar,
            "domain_id": domain_id,
            "proprio": proprio,
        }

    def _prepare_images_frame(self, batch: dict[str, Tensor], frame_idx: int = 0) -> tuple[Tensor, Tensor]:
        """从 batch 取某一帧的图像 → [B, n_view, C, H, W]。"""
        present_img_keys = [key for key in self.image_features if key in batch]
        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features missing. Batch keys: {list(batch.keys())}, "
                f"expected one of {self.image_features}."
            )
        images, masks = [], []
        for key in present_img_keys:
            img = batch[key]
            if img.ndim == 5:
                img = img[:, frame_idx]
            if self.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.resize_imgs_with_padding)
            images.append(img)
            masks.append(torch.ones(img.size(0), dtype=torch.bool, device=img.device))
        stacked_imgs = torch.stack(images, dim=1)
        stacked_masks = torch.stack(masks, dim=1)
        total_views = self.num_image_views or stacked_imgs.size(1)
        total_views = max(total_views, stacked_imgs.size(1))
        num_pad = total_views - stacked_imgs.size(1)
        if num_pad > 0:
            pad_shape = (stacked_imgs.size(0), num_pad, *stacked_imgs.shape[2:])
            pad_imgs = stacked_imgs.new_zeros(pad_shape)
            pad_masks = stacked_masks.new_zeros((stacked_masks.size(0), num_pad))
            stacked_imgs = torch.cat([stacked_imgs, pad_imgs], dim=1)
            stacked_masks = torch.cat([stacked_masks, pad_masks], dim=1)
        return stacked_imgs, stacked_masks

    def _prepare_state(self, batch: dict[str, Tensor], batch_size: int, device: torch.device) -> Tensor:
        if OBS_STATE not in batch:
            return torch.zeros(batch_size, 0, device=device)
        state = batch[OBS_STATE]
        if state.ndim > 2:
            state = state[:, -1, :]
        return pad_vector(state, self.model.cfg.dim_proprio)

    def _get_domain_id(self, batch: dict[str, Tensor], batch_size: int, device: torch.device) -> Tensor:
        candidate = batch.get("domain_id")
        if candidate is None:
            return torch.zeros(batch_size, dtype=torch.long, device=device)
        if not isinstance(candidate, torch.Tensor):
            candidate = torch.as_tensor(candidate, device=device)
        else:
            candidate = candidate.to(device=device)
        if candidate.ndim == 0:
            candidate = candidate.expand(batch_size)
        if candidate.ndim > 1:
            candidate = candidate.view(candidate.shape[0], -1)[:, 0]
        if candidate.shape[0] != batch_size:
            candidate = candidate.expand(batch_size)
        return candidate.to(dtype=torch.long)

    def _prepare_action_targets(self, batch: dict[str, Tensor]) -> Tensor:
        if ACTION not in batch:
            raise ValueError("Batch is missing action targets required for training.")
        actions = batch[ACTION]
        if actions.ndim == 2:
            actions = actions.unsqueeze(1)
        actions = pad_tensor_along_dim(actions, self.config.chunk_size, dim=1)
        if actions.shape[-1] != self.model.dim_action:
            actions = pad_vector(actions, self.model.dim_action)
        return actions

    # ============================================================
    # 训练 / 推理入口
    # ============================================================
    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        inputs = self._build_model_inputs(batch)
        targets = self._prepare_action_targets(batch)
        losses = self.model(action=targets, **inputs)

        # loss 项按 stage 不同:
        #   Stage 1: loss_action_total
        #   Stage 2: loss_sb_total
        #   Stage 3: loss_ep
        loss_keys = {"loss_action_total", "loss_sb_total", "loss_ep", "loss_joint"}
        total_loss = sum(v for k, v in losses.items() if k in loss_keys)

        log_dict = {k: v.detach().item() if torch.is_tensor(v) else float(v)
                    for k, v in losses.items()}
        log_dict["loss"] = total_loss.detach().item()
        return total_loss, log_dict

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:  # noqa: ARG002
        self.eval()
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])
        return self._get_action_chunk(batch)

    def _get_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        inputs = self._build_model_inputs(batch)
        steps = getattr(self.config, "N_steps", 5)
        actions = self.model.generate_actions(steps=steps, **inputs)
        return actions
