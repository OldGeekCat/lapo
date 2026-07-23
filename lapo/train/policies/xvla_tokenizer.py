"""X-VLA language tokenizer —— 把 batch 里的 task 文本转成 BART token id。

lerobot 0.4.4 的 XVLA 训练流程有个 gap：dataset（LeRobotDataset）产出的是
``task`` 文本字段，但 XVLAPolicy.forward 期望 batch 已含
``observation.language.tokens`` + ``observation.language.attention_mask``
（token id 张量）。tokenization 本应发生在 dataset transform 层，但 lerobot
标准 dataloader 不做这一步。

本模块在 compute_loss 层补这一步：检测 batch 是否缺 language tokens，
缺则用 facebook/bart-large tokenizer 现场转。这样 lrt 用标准
lerobot dataloader 也能训 xvla，不要求 dataset 自带 tokenized 字段。
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# 默认 tokenizer（BART-large，与 xvla 官方配置一致）
_DEFAULT_TOKENIZER_NAME = "facebook/bart-large"
_MAX_LENGTH = 64  # XVLAConfig.tokenizer_max_length 默认值

# OpenArm 标准 3 相机的默认 rename（global/wrist1/wrist2 → image/image2/image3）。
# 作为 fallback：策略应优先传入 cfg.dataset.rename_map，避免两处写死不一致。
_DEFAULT_RENAME_MAP = {
    "observation.images.global": "observation.images.image",
    "observation.images.wrist1": "observation.images.image2",
    "observation.images.wrist2": "observation.images.image3",
}

# 模块级缓存（避免每个 batch 都重新加载 tokenizer）
_tokenizer = None
_tokenizer_name = None


def _get_tokenizer(tokenizer_name: str = _DEFAULT_TOKENIZER_NAME):
    """延迟加载 + 缓存 BART tokenizer。"""
    global _tokenizer, _tokenizer_name
    if _tokenizer is not None and _tokenizer_name == tokenizer_name:
        return _tokenizer

    from transformers import BartTokenizer
    _tokenizer = BartTokenizer.from_pretrained(tokenizer_name)
    _tokenizer_name = tokenizer_name
    logger.info("xvla tokenizer loaded from %s (vocab=%d)", tokenizer_name, _tokenizer.vocab_size)
    return _tokenizer


def tokenize_language_xvla(batch: dict, max_length: int = _MAX_LENGTH,
                          tokenizer_name: str = _DEFAULT_TOKENIZER_NAME,
                          rename_map: dict[str, str] | None = None) -> dict:
    """给 batch 注入 language tokens（若缺失），并 rename 图像 key 对齐 xvla policy。

    检测 ``observation.language.tokens`` 是否存在；若否，从 ``task`` 文本
    现场 tokenize。同时把数据集的图像 key（global/wrist1/wrist2）rename 为
    policy 期望的 key（image/image2/image3）。

    rename_map 来源优先级：显式参数 > 内置默认（OpenArm 标准 3 相机）。
    注意：lerobot make_policy 的 rename_map 只跳过 feature 一致性校验，
    真正对 batch 的 rename 在此执行（lrt 用自己的 dataloader，不走 lerobot
    processor pipeline）。
    """
    TOK_KEY = "observation.language.tokens"
    MASK_KEY = "observation.language.attention_mask"

    if TOK_KEY in batch:
        return batch  # dataset 已自带，无需处理

    task = batch.get("task")
    if task is None:
        raise KeyError(
            "batch 既无 observation.language.tokens 也无 task 文本，"
            "xvla forward 无法获取语言输入。"
        )

    tok = _get_tokenizer(tokenizer_name)
    import torch
    if isinstance(task, str):
        task = [task]

    enc = tok(
        task,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
        truncation=True,
    )
    # Move to same device as other batch tensors
    device = next((v.device for v in batch.values() if isinstance(v, torch.Tensor)), torch.device("cpu"))
    batch[TOK_KEY] = enc["input_ids"].to(device)
    batch[MASK_KEY] = enc["attention_mask"].to(torch.bool).to(device)

    # Rename image keys: dataset 原始 key → policy 期望的 key
    eff_rename = rename_map or _DEFAULT_RENAME_MAP
    for old_key, new_key in eff_rename.items():
        if old_key in batch and new_key not in batch:
            batch[new_key] = batch.pop(old_key)

    return batch
