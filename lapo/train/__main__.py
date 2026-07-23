"""torchrun 多卡训练入口。

用法（3 卡 FSDP）::

    torchrun --nproc_per_node=3 -m lapo.train --config exp.yaml

或在单机多卡脚本里 spawn。FSDP 由 TrainingConfig.fsdp=True 触发（YAML 里设
``training.fsdp: true``，或本入口自动检测 torchrun 环境开启）。

设计：复用 services.run_training 的完整编排（registry/strategy/engine/artifacts），
本入口只负责：(1) init_distributed (2) 自动开启 fsdp 标志 (3) 配 outputs_root。
进程组在 run_training 内部 init/destroy（见 services/training.py）。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _setup_hf_env() -> None:
    """Point HuggingFace tooling at the unified cache under $LAPO_HOME.

    Done lazily at CLI startup, NOT at module import, so that merely importing
    this module has no filesystem side effects. ``setdefault`` keeps any user
    override.
    """
    from lapo.paths import hf_cache_dir

    os.environ.setdefault("HF_HOME", str(hf_cache_dir()))
    os.environ.setdefault("HF_HUB_CACHE", str(hf_cache_dir()))
    # Default to the China mirror so downloads work out-of-the-box behind the GFW.
    # Users can override with `export HF_ENDPOINT=https://huggingface.co`.
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def main(argv: list[str] | None = None) -> int:
    import argparse

    # 设置 HF_ENDPOINT(镜像站) + HF_HOME(缓存路径)。
    # torchrun spawn 的子进程会继承这些环境变量，保证多卡训练也能走镜像站下载。
    _setup_hf_env()

    ap = argparse.ArgumentParser(
        prog="python -m lapo.train",
        description="lapo 训练入口（支持 torchrun 多卡 FSDP）",
    )
    ap.add_argument("--config", required=True, help="实验 YAML 路径")
    ap.add_argument("--outputs", default=None,
                    help="产物输出根目录（默认 $LAPO_HOME/outputs）")
    ap.add_argument("--ddp", action="store_true",
                    help="用 DDP（默认 FSDP）。V100 等 32G 卡推荐 DDP+fp16。")
    a = ap.parse_args(argv)

    from lapo.train.config import load_config
    from lapo.train.services.training import load_registry_with_builtins, run_training
    from lapo.train.distributed import is_distributed_enabled, is_main_process, init_distributed

    cfg = load_config(a.config)

    # torchrun 环境下自动开启分布式（默认 FSDP，--ddp 切 DDP）
    if is_distributed_enabled():
        if a.ddp:
            cfg.training.ddp = True
            cfg.training.fsdp = False
        elif not cfg.training.ddp:
            cfg.training.fsdp = True
        if is_main_process():
            mode = "DDP" if cfg.training.ddp else "FSDP"
            print(f"[lapo.train] 检测到 torchrun，开启 {mode}", file=sys.stderr)

    # outputs 根目录
    if a.outputs:
        outputs_root = a.outputs
    else:
        from lapo.paths import outputs_dir
        outputs_root = outputs_dir()

    registry = load_registry_with_builtins()

    if is_main_process():
        print(f"[lapo.train] policy={cfg.policy_name} strategy={cfg.strategy_name} "
              f"fsdp={cfg.training.fsdp}", file=sys.stderr)

    run_dir = run_training(
        cfg, registry=registry, outputs_root=outputs_root,
    )

    if is_main_process():
        print(f"[lapo.train] ✓ 训练完成。产物: {run_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
