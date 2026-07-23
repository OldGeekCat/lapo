"""训练编排 service：config → registry → strategy → engine → 训练。

纯逻辑层，无 typer/console。CLI 和未来 web 都调它。
resolve_strategy / load_registry_with_builtins 是纯逻辑（registry 操作），
run_training 调 engine（需 torch + lerobot，由调用方注入或延迟 import）。
"""
from __future__ import annotations

import hashlib
import importlib
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from lapo.train.config import RunConfig
from lapo.train.registry import Registry, TraitError
from lapo.train.services.registry_store import RegistryStore
from lapo.train.strategy import TrainStrategy


def load_registry_with_builtins(root: str | Path | None = None) -> Registry:
    """加载持久化 registry，并播入内置条目（用户自定义同名条目优先保留）。

    内置条目标记 _builtin: true。已存在的用户条目不被覆盖。
    """
    store = RegistryStore(root)
    reg = Registry(store)

    # 内置 policy
    from lapo.train.policies.builtin import BUILTIN_POLICIES
    existing_policies = store.load_policies()
    for name, entry in BUILTIN_POLICIES.items():
        if name not in existing_policies:
            existing_policies[name] = {
                "config_cls": entry.config_cls,
                "policy_cls": entry.policy_cls,
                "traits": sorted(entry.traits),
                "default_strategy": entry.default_strategy,
                "defaults": entry.defaults,
                "_builtin": True,
            }
    store.save_policies(existing_policies)

    # 内置 strategy
    from lapo.train.policies.builtin import BUILTIN_STRATEGIES
    existing_strategies = store.load_strategies()
    for name, entry in BUILTIN_STRATEGIES.items():
        if name not in existing_strategies:
            existing_strategies[name] = {
                "cls_path": entry.cls_path,
                "required_traits": sorted(entry.required_traits),
                "defaults": entry.defaults,
                "_builtin": True,
            }
    store.save_strategies(existing_strategies)

    return reg


def resolve_strategy(cfg: RunConfig, registry: Registry) -> TrainStrategy:
    """解析 strategy：YAML 显式 > policy 推荐默认 > 'default'。

    含 trait 兼容性校验（strategy.required_traits ⊆ policy.traits），
    不兼容抛 TraitError。
    """
    policy_entry = registry.get_policy(cfg.policy_name)
    if policy_entry is None:
        raise ValueError(f"policy '{cfg.policy_name}' 未注册")

    strategy_name = cfg.strategy_name or policy_entry.default_strategy or "default"
    strategy_entry = registry.get_strategy(strategy_name)
    if strategy_entry is None:
        raise ValueError(f"strategy '{strategy_name}' 未注册")

    registry.check_compatibility(cfg.policy_name, strategy_name)

    cls = _import_class(strategy_entry.cls_path)
    return cls(cfg, registry=registry)


def _import_class(dotted: str) -> type:
    """importlib 加载 'module.path.ClassName'。"""
    module_path, _, cls_name = dotted.rpartition(".")
    mod = importlib.import_module(module_path)
    return getattr(mod, cls_name)


def run_training(cfg: RunConfig, *, registry: Registry,
                 outputs_root: str | Path,
                 env_info: Optional[dict] = None) -> Path:
    """编排一次训练 run。返回 run_dir。

    流程: resolve_strategy → 建 run_dir → load ds_meta → build policy_cfg →
          resolve delta_timestamps → build dataloader → resolve device →
          起 TrainingEngine.run()。

    delta_timestamps 来自 policy config 的 ``*_delta_indices``（lerobot canonical
    语义，见 ``lerobot.datasets.factory.resolve_delta_timestamps``）。序列策略
    （ACT/diffusion/xvla）必须有它才能拿到 action chunk；不传会让 dataset 只产
    单帧，policy forward 维度不匹配崩（HANDOFF §B1 上机发现）。
    """
    from lapo.train.engine import TrainingEngine

    strategy = resolve_strategy(cfg, registry)

    run_id = _make_run_id(cfg.policy_name)
    run_dir = Path(outputs_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    ds_meta = _load_ds_meta(cfg.dataset)
    policy_cfg = _build_policy_cfg(cfg, registry)
    # 策略可自定义 delta_timestamps（如 SB-VLA 需未来帧）；默认走 lerobot 解析
    delta_timestamps = strategy.build_delta_timestamps(ds_meta)
    if delta_timestamps is None:
        delta_timestamps = _resolve_delta_timestamps(policy_cfg, ds_meta)

    # FSDP/DDP：初始化进程组（torchrun 环境下）
    if cfg.training.fsdp or cfg.training.ddp:
        from lapo.train.distributed import init_distributed
        init_distributed()

    dataloader = _build_dataloader(cfg.dataset, cfg.training,
                                   policy_cfg=policy_cfg, ds_meta=ds_meta,
                                   delta_timestamps=delta_timestamps)
    device = _resolve_device(cfg.training.device)

    # 验证集切分（val_ratio > 0 时，按 episode 边界切，不切断序列）
    val_dataloader = None
    if cfg.dataset.val_ratio > 0 and cfg.training.val_every > 0:
        total_eps = ds_meta.total_episodes
        n_val = max(1, int(total_eps * cfg.dataset.val_ratio))
        train_episodes = list(range(0, total_eps - n_val))
        val_episodes = list(range(total_eps - n_val, total_eps))
        # 重建 train dataloader 只用 train episodes
        dataloader = _build_dataloader(
            cfg.dataset, cfg.training, ds_meta=ds_meta,
            episodes=train_episodes, delta_timestamps=delta_timestamps)
        # val dataloader：batch=1 减小显存压力，不套 DistributedSampler（只用 rank0）
        import copy
        val_training_cfg = copy.copy(cfg.training)
        val_training_cfg.batch_size = 1
        val_dataloader = _build_dataloader(
            cfg.dataset, val_training_cfg, ds_meta=ds_meta,
            episodes=val_episodes, delta_timestamps=delta_timestamps,
            distributed=False)
        from lapo.train.distributed import is_main_process
        if is_main_process():
            print(f"[lapo.train] val split: train={len(train_episodes)} episodes, "
                  f"val={len(val_episodes)} episodes (batch=1), eval every {cfg.training.val_every} steps",
                  file=__import__('sys').stderr)

    engine = TrainingEngine(
        cfg, strategy=strategy, run_dir=run_dir, registry=registry,
        ds_meta=ds_meta, dataloader=dataloader, device=device,
        env_info=env_info, delta_timestamps=delta_timestamps,
        val_dataloader=val_dataloader,
    )
    try:
        engine.run()
    finally:
        if cfg.training.fsdp or cfg.training.ddp:
            from lapo.train.distributed import destroy_distributed
            destroy_distributed()
    return run_dir


def _make_run_id(policy: str) -> str:
    """<timestamp>_<policy>_<short_hash>，时间排序 + 一眼看出 policy + 防冲突。"""
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    h = hashlib.sha1(str(time.time()).encode()).hexdigest()[:4]
    short = policy.split(".")[-1] if "." in policy else policy
    return f"{ts}_{short}_{h}"


def _load_ds_meta(dataset_cfg) -> Any:
    """加载 lerobot dataset.meta（延迟 import lerobot）。"""
    return _lerobot_dataset_module().LeRobotDataset(
        dataset_cfg.repo_id, root=dataset_cfg.root).meta


def _lerobot_dataset_module():
    """延迟 import lerobot 的 lerobot_dataset 模块（供测试 monkeypatch）。"""
    from lapo.train.compat import import_lerobot
    prefix = import_lerobot()
    return importlib.import_module(f"{prefix}.datasets.lerobot_dataset")


def _build_policy_cfg(cfg: RunConfig, registry: Registry) -> Any:
    """解析 policy config（轻量，不实例化完整 model）。

    用于在 build dataloader 前拿到 policy config 的 ``*_delta_indices``。
    与 ``compat.build_policy_for`` 的 config 解析同路径：短名→registry→
    config_cls→实例化；自定义路径（含点）直 import。
    """
    from lapo.train.compat import _instantiate_config, load_xvla_config

    policy_entry = registry.get_policy(cfg.policy_name)
    if policy_entry is None:
        # 自定义路径（含点）：rpartition 取类
        config_cls_path = cfg.policy_name
    else:
        config_cls_path = policy_entry.config_cls

    base_model = cfg.policy_overrides.get("base_model")
    short_name = cfg.policy_name
    # xvla 路由到 load_xvla_config（绕 draccus 'type' bug #8），与 build_policy_for 一致
    if "xvla" in short_name and base_model:
        import importlib as _il
        _mp, _, _cn = config_cls_path.rpartition(".")
        _cls = getattr(_il.import_module(_mp), _cn)
        return load_xvla_config(_cls, base_model)
    cfg_obj = _instantiate_config(config_cls_path, cfg.policy_overrides, base_model)
    return cfg_obj


def _resolve_delta_timestamps(policy_cfg: Any, ds_meta: Any) -> Optional[dict]:
    """从 policy config 的 *_delta_indices 推 lerobot delta_timestamps。

    委托 lerobot canonical 实现 ``resolve_delta_timestamps``（把 frame indices
    除以 fps 得 seconds，按 dataset features 过滤 key）。policy 三个 indices 都
    None 时返回 None（不阻塞非 chunk 策略）。
    """
    from lapo.train.compat import import_lerobot
    prefix = import_lerobot()
    factory = importlib.import_module(f"{prefix}.datasets.factory")
    return factory.resolve_delta_timestamps(policy_cfg, ds_meta)


def _build_dataloader(dataset_cfg, training_cfg, *, policy_cfg=None,
                      ds_meta=None, episodes=None,
                      delta_timestamps=None, distributed=True) -> Any:
    """构建 lerobot dataloader（延迟 import torch + lerobot）。

    若给了 ``policy_cfg``+``ds_meta``（且未显式传 delta_timestamps），先解析
    delta_timestamps 传给 ``LeRobotDataset``——序列策略必须有它才能产 chunk。

    episodes: 只加载指定 episode 列表（用于验证集切分）；None=全量。
    distributed: True 时套 DistributedSampler（训练用）；False 时不套（验证用，
        只在 rank0 评估，避免多卡重复）。
    """
    import torch
    if delta_timestamps is None and policy_cfg is not None and ds_meta is not None:
        delta_timestamps = _resolve_delta_timestamps(policy_cfg, ds_meta)
    ds = _lerobot_dataset_module().LeRobotDataset(
        dataset_cfg.repo_id, root=dataset_cfg.root,
        delta_timestamps=delta_timestamps, episodes=episodes,
        tolerance_s=0.05)  # 放宽容差：合并视频 PTS 有微小偏差（默认 1e-4 太严）

    # FSDP/DDP：用 DistributedSampler 切分数据（替代 shuffle=True）
    # 验证集（distributed=False）不套 sampler——只用 rank0 评估
    sampler = None
    shuffle = True
    if distributed and (training_cfg.fsdp or training_cfg.ddp):
        sampler = torch.utils.data.distributed.DistributedSampler(
            ds, shuffle=True, drop_last=True)
        shuffle = False  # sampler 接管 shuffle

    return torch.utils.data.DataLoader(
        ds, batch_size=training_cfg.batch_size, shuffle=shuffle,
        sampler=sampler,
        num_workers=training_cfg.num_workers, drop_last=True,
    )


def _resolve_device(device_str: str) -> str:
    """设备解析（auto → cuda/mps/cpu）。与 lri 对齐。

    分布式（FSDP/DDP）环境下：无论输入是 "auto" 还是 "cuda"，都绑定到
    当前进程的 local_rank（cuda:{LOCAL_RANK}），确保每卡用自己对应的 GPU。
    """
    # 分布式环境：优先按 local_rank 绑定
    from lapo.train.distributed import is_distributed_enabled, get_local_rank
    if is_distributed_enabled():
        local_rank = get_local_rank()
        return f"cuda:{local_rank}"

    if device_str != "auto":
        return device_str
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"
