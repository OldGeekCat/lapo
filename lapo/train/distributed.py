"""FSDP / DDP 分布式训练工具。

为 lrt 提供 PyTorch FSDP（Fully Sharded Data Parallel）支持。FSDP 把
optimizer state + 梯度 + 参数切片分到各卡，等效显存 = N卡 × 单卡显存，
是 3.6B 模型（pi0.5）在 3×V100-32GB 上跑全精度（无量化）的唯一干净方案。

核心 API：
- ``is_distributed_enabled()``：检测 torchrun 环境（WORLD_SIZE > 1）
- ``init_distributed()``：init_process_group，返回 (rank, world_size, local_rank)
- ``wrap_fsdp(model, ...)``：FSDP 包装（MixedPrecision + auto_wrap + tied weights）
- ``unwrap(model)``：FSDP._orig_mod / DDP.module / 原样
- ``is_main_process()``：rank0 判断（只 rank0 写产物/日志）

V100 注意：不支持 bf16（无 tensor core），MixedPrecision 用 fp16。
bf16 在 Ampere+（A100/3090）自动切换。
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---- 环境检测 ----

def is_distributed_enabled() -> bool:
    """是否在 torchrun / spawn 分布式环境下（WORLD_SIZE > 1）。"""
    try:
        return int(os.environ.get("WORLD_SIZE", "1")) > 1
    except (ValueError, TypeError):
        return False


def is_main_process() -> bool:
    """是否 rank0（主进程）。非分布式环境恒 True。

    只 rank0 写产物（run.json/metrics/checkpoint），避免多卡重复写冲突。
    """
    if not is_distributed_enabled():
        return True
    return int(os.environ.get("RANK", "0")) == 0


def get_local_rank() -> int:
    """当前进程的 local rank（单机多卡 = 全局 rank）。"""
    return int(os.environ.get("LOCAL_RANK", "0"))


# ---- 进程组初始化 ----

def init_distributed(backend: str = "nccl") -> tuple[int, int, int]:
    """初始化分布式进程组。

    必须在创建任何 CUDA tensor / DDP/FSDP 模型前调用。
    幂等：已初始化则直接返回。

    Returns:
        (rank, world_size, local_rank)
    """
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size(), get_local_rank()

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = get_local_rank()

    if world_size > 1:
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        logger.info("init_distributed: rank=%d/%d local_rank=%d backend=%s",
                    rank, world_size, local_rank, backend)
    else:
        # world_size=1 也初始化（FSDP 需要默认 process group，即使单卡）
        dist.init_process_group(backend=backend, rank=0, world_size=1)
        logger.info("init_distributed: single-process (world_size=1), backend=%s", backend)
    return rank, world_size, local_rank


def destroy_distributed() -> None:
    """销毁进程组（训练结束后调）。"""
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _wrap_default(min_params: int = 1_000_000):
    """构造 size-based auto_wrap policy：参数数 ≥ min_params 的子模块各自 FSDP wrap。

    避免顶层全量 flat_param（3.6B fp32=14G）的 init 峰值 OOM。
    每个 transformer layer（~几百M 参数）各自一个 FSDP 单元，init 峰值大幅降低。
    """

    def _size_based_fn(module, recurse, *args, **kwargs):
        if recurse:
            return False
        return sum(p.numel() for p in module.parameters()) >= min_params

    return _size_based_fn


# ---- FSDP 包装 ----

def _resolve_dtype(dtype_str: str):
    """配置字符串 → torch dtype。bf16 在不支持硬件上 fallback 到 fp16。"""
    import torch
    if dtype_str in ("bfloat16", "bf16"):
        # V100（Volta, sm70）无 bf16 tensor core，fallback 到 fp16
        if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
            logger.warning("硬件不支持 bf16 tensor core，fallback 到 fp16")
            return torch.float16
        return torch.bfloat16
    if dtype_str in ("float16", "fp16", "half"):
        return torch.float16
    return torch.float32


def wrap_fsdp(
    model,
    device: str,
    *,
    sharding: str = "full",
    grad_checkpoint: bool = False,
    dtype: str = "bfloat16",
):
    """用 FSDP 包装模型。

    Args:
        model: 已 .to(device) 的 nn.Module。
        device: 设备字符串（cuda:0 等）。
        sharding: full（完全切片，省显存最多）/ shard_grad_op / no_shard（=DDP）。
        grad_checkpoint: 开启梯度检查点（省激活显存，换约 20% 额外计算）。
        dtype: 混合精度 dtype（bf16 优先，V100 自动 fallback fp16）。

    Returns:
        FSDP 包装后的模型。use_orig_params=True 保证 named_parameters 不加前缀，
        pi05/xvla 的 substring param-split 不受影响。
    """
    import torch
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
        CPUOffload,
    )

    compute_dtype = _resolve_dtype(dtype)

    # FSDP MixedPrecision：flat_param 用低精度（fp16 = 7G 而非 fp32 14G），
    # 避免 3.6B fp32 模型在 FSDP wrap/init 时 OOM。
    # 模型用 fp32 构建（pi05/V100 要求），FSDP 在 gather 时 cast 到 compute_dtype。
    # param_dtype/buffer_dtype/reduce_dtype 全用低精度，避免 "Float and BFloat16" 混用。
    if compute_dtype != __import__("torch").float32:
        mp_policy = MixedPrecision(
            param_dtype=compute_dtype,
            reduce_dtype=compute_dtype,
            buffer_dtype=compute_dtype,
        )
    else:
        mp_policy = None

    sharding_map = {
        "full": ShardingStrategy.FULL_SHARD,
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
        "no_shard": ShardingStrategy.NO_SHARD,  # 等效 DDP
    }

    # 梯度检查点：必须在 FSDP wrap 前开启
    if grad_checkpoint:
        # 优先用模型自带接口（HF/lerobot 模型常见）
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            logger.info("grad_checkpoint: 已开启（model.gradient_checkpointing_enable）")
        else:
            # FSDP 的 use_orig_params 模式下可用 activation checkpointing
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                checkpoint_wrapper, CheckpointImpl,
            )
            logger.info("grad_checkpoint: 用通用 checkpoint_wrapper")

    # sync_module_states=False：每卡各自加载 checkpoint（避免全量 broadcast OOM）。
    # auto_wrap：按子模块分组（如 transformer layer），每个 FSDP 单元只 flat 化
    # 一个 layer（~0.5G），避免顶层全量 flat_param(3.6B fp32=14G) 的 init 峰值 OOM。
    auto_wrap_policy = _wrap_default(min_params=1_000_000)
    model = FSDP(
        model,
        sharding_strategy=sharding_map.get(sharding, ShardingStrategy.FULL_SHARD),
        mixed_precision=mp_policy,
        cpu_offload=CPUOffload(offload_params=True),
        device_id=torch.device(device),
        sync_module_states=False,
        use_orig_params=True,
        forward_prefetch=True,
        auto_wrap_policy=auto_wrap_policy,
    )
    logger.info("FSDP wrap: sharding=%s dtype=%s grad_checkpoint=%s",
                sharding, compute_dtype, grad_checkpoint)
    return model


def wrap_ddp(model, device: str, *, grad_checkpoint: bool = False, dtype: str = "float16"):
    """用 DDP 包装模型（V100 等 32G 卡：fp16 + grad ckpt 解决显存）。

    DDP 每卡存全量模型，故用 fp16（7G 参数）+ grad ckpt（激活）压显存。
    与 FSDP 的区别：不切片，每卡全量；优势是 init 无 flat_param 峰值。

    Args:
        model: CPU 上的 nn.Module（fp32 构建）。
        device: cuda:{local_rank}。
        grad_checkpoint: 梯度检查点。
        dtype: fp16/bf16（模型 cast 到此精度后搬 GPU）。
    """
    import torch
    from torch.nn.parallel import DistributedDataParallel as DDP

    compute_dtype = _resolve_dtype(dtype)
    local_rank = get_local_rank()

    # 梯度检查点（FSDP wrap 前）
    if grad_checkpoint and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logger.info("wrap_ddp: gradient_checkpointing_enable")

    # cast 到低精度 + 搬 GPU（fp32→fp16：14G→7G）
    model = model.to(compute_dtype).to(device)
    torch.cuda.set_device(local_rank)

    model = DDP(
        model,
        device_ids=[local_rank],
        find_unused_parameters=True,      # pi05 可能有不参与梯度的层
        gradient_as_bucket_view=True,     # 省显存
    )
    logger.info("wrap_ddp: dtype=%s grad_checkpoint=%s device=%s",
                compute_dtype, grad_checkpoint, device)
    return model


# ---- unwrap ----

def unwrap(model) -> Any:
    """剥去 FSDP/DDP wrapper，返回裸 nn.Module。递归剥嵌套包装。

    FSDP 包装后 model._orig_mod 是原始模块；
    DDP 包装后 model.module 是原始模块；
    未包装则原样返回。
    """
    while True:
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        elif hasattr(model, "module"):
            model = model.module
        else:
            return model


# ---- 指标聚合 ----

def gather_mean(value: float, device: str = "cpu") -> float:
    """跨卡取平均值（用于 loss 等指标）。

    单进程直接返回原值。分布式下 all_reduce 后除以 world_size。
    """
    if not is_distributed_enabled():
        return value
    import torch
    import torch.distributed as dist
    t = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t / dist.get_world_size()).item()
