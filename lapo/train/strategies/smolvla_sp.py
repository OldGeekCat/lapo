"""SmolVLA 语言预处理策略：注入 language tokens + attention mask。

SmolVLA 的 forward 期望 batch 已含 observation.language.tokens +
observation.language.attention_mask，但我们的 dataloader 只产 task 文本。
本策略在 preprocess 时用 SmolVLA 自带的 processor tokenizer 现场转换。

SmolVLA 内置 MEAN_STD 归一化（normalization_mapping），无需手动 normalizer。
"""
from __future__ import annotations

from lapo.train.strategy import TrainStrategy


class SmolVLAStrategy(TrainStrategy):
    """SmolVLA 语言注入策略。"""

    def required_traits(self) -> set[str]:
        return set()  # SmolVLA 无特殊 trait 要求

    def build_policy(self, cfg, ds_meta):
        """构建 policy 并缓存 tokenizer。"""
        policy = super().build_policy(cfg, ds_meta)
        # SmolVLA 的 processor 在 model 里（vlm_with_expert.processor）
        self._tokenizer = None
        try:
            model = getattr(policy, "model", None)
            if model is not None and hasattr(model, "vlm_with_expert"):
                self._tokenizer = model.vlm_with_expert.processor.tokenizer
                import sys
                print(f"[smolvla_sp] tokenizer 就绪: {type(self._tokenizer).__name__}", file=sys.stderr)
        except Exception as e:
            import sys
            print(f"[smolvla_sp] ⚠️ 无法获取 tokenizer: {e}", file=sys.stderr)
        return policy

    def preprocess(self, batch):
        """注入 language tokens（若缺失），用 SmolVLA tokenizer。"""
        TOK_KEY = "observation.language.tokens"
        MASK_KEY = "observation.language.attention_mask"

        if TOK_KEY in batch:
            return batch  # dataset 已自带

        task = batch.get("task")
        if task is None:
            return batch  # 无语言输入（SmolVLA 也支持无语言模式）

        import torch
        if self._tokenizer is None:
            return batch

        if isinstance(task, str):
            task = [task]

        # SmolVLA tokenizer（来自 SmolVLM2 processor）
        enc = self._tokenizer(
            task,
            padding="max_length",
            max_length=48,  # SmolVLAConfig.tokenizer_max_length
            return_tensors="pt",
            truncation=True,
        )
        device = next((v.device for v in batch.values() if isinstance(v, torch.Tensor)), torch.device("cpu"))
        batch[TOK_KEY] = enc["input_ids"].to(device)
        batch[MASK_KEY] = enc["attention_mask"].to(torch.bool).to(device)
        return batch

    def compute_loss(self, policy, batch):
        """preprocess + forward 透传。"""
        batch = self.preprocess(batch)
        out = policy.forward(batch)
        main_loss = out[0] if isinstance(out, (tuple, list)) else out
        return main_loss
