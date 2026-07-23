"""distributed.py 单测。

纯逻辑部分（环境检测/unwrap）不依赖 torch.distributed，用环境变量 mock +
duck typing 对象测。FSDP wrap 的真实多卡行为靠 torchrun 集成验证。
"""
import os
from unittest.mock import patch

import pytest

from lapo.train.distributed import (
    is_distributed_enabled, is_main_process, get_local_rank,
    unwrap, _resolve_dtype,
)


# ---------- 环境检测 ----------

class TestEnvDetection:
    def _set_dist_env(self, rank=0, world_size=1, local_rank=0):
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(local_rank)

    def _clear_dist_env(self):
        for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
            os.environ.pop(k, None)

    def test_single_process_no_env(self):
        """无 WORLD_SIZE → 非分布式。"""
        self._clear_dist_env()
        assert is_distributed_enabled() is False
        assert is_main_process() is True  # 单进程恒 True

    def test_multi_process_enabled(self):
        """WORLD_SIZE > 1 → 分布式。"""
        self._set_dist_env(rank=0, world_size=3, local_rank=0)
        assert is_distributed_enabled() is True
        self._clear_dist_env()

    def test_rank0_is_main(self):
        """rank0 是主进程。"""
        self._set_dist_env(rank=0, world_size=3, local_rank=0)
        assert is_main_process() is True

    def test_non_rank0_not_main(self):
        """非 rank0 不是主进程。"""
        self._set_dist_env(rank=2, world_size=3, local_rank=2)
        assert is_main_process() is False
        self._clear_dist_env()

    def test_get_local_rank(self):
        self._set_dist_env(rank=1, world_size=3, local_rank=1)
        assert get_local_rank() == 1
        self._clear_dist_env()

    def test_invalid_world_size(self):
        """WORLD_SIZE 非法 → 不崩，返回 False。"""
        os.environ["WORLD_SIZE"] = "abc"
        assert is_distributed_enabled() is False
        os.environ.pop("WORLD_SIZE", None)


# ---------- unwrap ----------

class _FakeModule:
    """模拟 nn.Module（duck typing，有 named_parameters）。"""
    def named_parameters(self):
        return []


class _FakeDDP:
    """模拟 DDP wrapper。"""
    def __init__(self, module):
        self.module = module
    def named_parameters(self):
        return []


class _FakeFSDP:
    """模拟 FSDP wrapper（_orig_mod）。"""
    def __init__(self, module):
        self._orig_mod = module
    def named_parameters(self):
        return []


class TestUnwrap:
    def test_plain_module(self):
        m = _FakeModule()
        assert unwrap(m) is m

    def test_ddp_module(self):
        inner = _FakeModule()
        ddp = _FakeDDP(inner)
        assert unwrap(ddp) is inner

    def test_fsdp_module(self):
        inner = _FakeModule()
        fsdp = _FakeFSDP(inner)
        assert unwrap(fsdp) is inner

    def test_nested_ddp_fsdp(self):
        """FSDP(DDP(model)) 嵌套 → 递归剥到裸 module。"""
        inner = _FakeModule()
        ddp = _FakeDDP(inner)
        fsdp = _FakeFSDP(ddp)
        assert unwrap(fsdp) is inner

    def test_real_torch_module(self):
        """真实 nn.Module 不被误剥。"""
        torch = pytest.importorskip("torch")
        import torch.nn as nn
        m = nn.Linear(3, 3)
        assert unwrap(m) is m


# ---------- dtype 解析 ----------

class TestResolveDtype:
    def test_fp32(self):
        torch = pytest.importorskip("torch")
        assert _resolve_dtype("float32") == torch.float32

    def test_fp16(self):
        torch = pytest.importorskip("torch")
        assert _resolve_dtype("fp16") == torch.float16
        assert _resolve_dtype("half") == torch.float16

    def test_bf16_or_fp16_fallback(self):
        """bf16 在不支持的硬件上 fallback fp16，否则 bf16。"""
        torch = pytest.importorskip("torch")
        result = _resolve_dtype("bfloat16")
        assert result in (torch.bfloat16, torch.float16)
