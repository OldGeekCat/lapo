"""SBPolicy —— lerobot 兼容的 SB-VLA policy 包装。

复刻 XVLAPolicy 的接口（forward / predict_action_chunk / _build_model_inputs /
_prepare_action_targets），但 model 换成 SBVLAHead。这样不动 engine / infer_server
就能接入（它们只调 policy.forward(batch) 和 policy.predict_action_chunk(batch)）。

batch 输入契约（与 ee6d 路径一致）：
  observation.language.tokens     [B, 64]   BART token id
  observation.images.image2/image3 [B, C, 480, 640]
  observation.state               [B, dim_proprio]
  action                          [B, chunk_size, dim_action]（训练时）
"""
from __future__ import annotations

from collections import deque

import torch
import torch.nn.functional as F
from torch import Tensor

# 复用 xvla 的常量 + 辅助（同 env 内）
from lerobot.policies.xvla.modeling_xvla import (
    OBS_LANGUAGE_TOKENS, OBS_STATE, ACTION,
    pad_vector, pad_tensor_along_dim, resize_with_pad, populate_queues,
)


class SBPolicy(torch.nn.Module):
    """SB-VLA policy：包装 SBVLAHead，暴露 lerobot 兼容接口。

    与 XVLAPolicy 的区别：model = SBVLAHead（encoder/g/f/SB），而非 XVLAModel（FM）。
    其余（图像/state/language 准备）与 XVLAPolicy 一致。
    """

    def __init__(self, config, head, xvla_config=None):
        """config: SBVLAConfig（含 chunk_size/dim 等）; head: SBVLAHead 实例.

        xvla_config: 可选，原 xvla-base 的 XVLAConfig（用于取 resize_imgs_with_padding
                     等 Florence2 要求的图像预处理参数）。Florence2 要求方形输入图
                     （image_pos_embed 只支持 square feature map），所以必须 resize。
        """
        super().__init__()
        self.config = config
        self.model = head
        # image_features：由 dataset.rename_map 决定（ee6d = image2/image3）
        self.image_features = ["observation.images.image", "observation.images.image2", "observation.images.image3"]
        self.num_image_views = 3
        # Florence2 要求方形图（image_pos_embed 只支持 square feature map）。
        # xvla-base 默认 resize_imgs_with_padding=[224, 224]。
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
    # 输入准备（复刻 XVLAPolicy，扩展为支持双帧：当前帧 t + 未来帧 t+H）
    # ============================================================
    def _build_model_inputs(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """构建模型输入。当前帧 + 多个未来 horizon 帧。

        g 多 horizon 监督需要 num_h 个未来帧（horizons=[15,30,45,60] → frame_idx 1..4）。
        数据层 build_delta_timestamps 拉 [0, h1, h2, h3, h4] 共 5 帧（frame_idx 0 是当前）。
        这里取 frame_idx 1..4，stack 成 [B, num_h, n_view, C,H,W]，再 flatten(0,1) 成
        [B*num_h, n_view, C,H,W] 喂给 model（一次 DaViT 出所有 horizon 的 z_tar）。

        若 batch 只有单帧（4D [B,C,H,W]，推理/兼容），未来帧退化为当前帧（model 内复制）。
        """
        input_ids = batch[OBS_LANGUAGE_TOKENS]
        batch_size = input_ids.shape[0]
        images_t, mask_t = self._prepare_images_frame(batch, frame_idx=0)
        domain_id = self._get_domain_id(batch, batch_size, images_t.device)
        proprio = self._prepare_state(batch, batch_size, images_t.device)

        # 未来帧：num_h 个 horizon（frame_idx 1..4），fold 进 batch 维
        any_img_key = next(k for k in self.image_features if k in batch)
        if batch[any_img_key].ndim == 5:  # [B, n_frames, C, H, W]
            # 取 cfg.horizons 对应的 frame_idx（数据层拉的帧序 = [0] + horizons）
            horizons = self.model.cfg.horizons          # (num_h,) e.g. (15,30,45,60)
            # 数据层拉的帧索引顺序：[0, h0, h1, ...]（见 strategy.build_delta_timestamps）
            # 所以 horizon hi 对应 frame_idx = horizons 里位置 +1
            tar_imgs, tar_masks = [], []
            for i, hv in enumerate(horizons):
                img_i, mask_i = self._prepare_images_frame(batch, frame_idx=i + 1)
                tar_imgs.append(img_i)            # [B, n_view, C,H,W]
                tar_masks.append(mask_i)          # [B, n_view]
            # stack: [B, num_h, n_view, C,H,W] → flatten(0,1): [B*num_h, n_view, C,H,W]
            images_tar = torch.stack(tar_imgs, dim=1).flatten(0, 1)
            mask_tar = torch.stack(tar_masks, dim=1).flatten(0, 1)
        else:
            # 单帧（推理/兼容）：退化为当前帧，model 内会复制 num_h 份
            images_tar, mask_tar = images_t, mask_t

        return {
            "input_ids": input_ids,
            "image_input": images_t,       # 当前帧 [B, n_view, C,H,W]
            "image_mask": mask_t,
            "image_input_tar": images_tar, # 未来多 horizon [B*num_h, n_view, C,H,W] 或单帧
            "image_mask_tar": mask_tar,
            "domain_id": domain_id,
            "proprio": proprio,
        }

    def _prepare_images_frame(self, batch: dict[str, Tensor], frame_idx: int = 0) -> tuple[Tensor, Tensor]:
        """从 batch 取某一帧的图像。

        batch[key] 可能是 5D [B, n_frames, C, H, W]（双帧）或 4D [B, C, H, W]（单帧）。
        返回 [B, n_view, C, H, W]（把各视图 key 堆成 view 维）。
        """
        present_img_keys = [key for key in self.image_features if key in batch]
        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features missing. Batch keys: {list(batch.keys())}, "
                f"expected one of {self.image_features}."
            )
        images, masks = [], []
        for key in present_img_keys:
            img = batch[key]
            if img.ndim == 5:           # [B, n_frames, C, H, W] → 取某帧
                img = img[:, frame_idx]
            # 此时 img = [B, C, H, W]
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
        # gain_h/mag_*/cos_* 是监控项（no_grad），不能加进 total_loss
        # VICReg 四项 + delta-VICReg 要进 total；loss_temporal 已 detach（delta 模式下仅监控）
        loss_keys = {"loss_g",
                     "loss_enc_var", "loss_enc_cov",     # encoder 输出 z_t 的 VICReg（跨样本防塌）
                     "loss_vic_var", "loss_vic_cov",     # g 输出 z_goal 的 VICReg（g 坍缩）
                     "loss_delta_var", "loss_delta_cov", # delta-VICReg（P1, Δz 每维 std）
                     "loss_temporal",                    # 时序增量正则（delta 模式下 detach=0,仅监控）
                     "loss_imle", "loss_force"}
        total_loss = sum(v for k, v in losses.items() if k in loss_keys)
        log_dict = {k: v.detach().item() if torch.is_tensor(v) else float(v) for k, v in losses.items()}
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
        # 推理不需要未来帧（z_tar）；generate_actions 只取当前帧 + 任务意图
        infer_inputs = {k: v for k, v in inputs.items()
                        if not k.endswith("_tar")}
        actions = self.model.generate_actions(steps=steps, **infer_inputs)
        return actions
