"""ArtifactWriter: 写标准训练产物（spec §4 契约）。

每个方法对应一个产物文件。run.json 原子写，metrics.jsonl append-only。
torch 仅在 save_checkpoint 内延迟 import；其余方法是纯 JSON/YAML I/O，
可在无 torch 环境完整单测。
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

SCHEMA_VERSION = 1


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(obj: Any) -> Any:
    """JSON fallback for non-stdlib scalars (numpy/torch).

    lerobot dataset meta ships ``numpy.int64``/``float32`` for fields like
    ``total_episodes``/``fps``; torch tensors can sneak in via stats. stdlib
    ``json`` can't serialize them, so ``write_dataset_info`` crashed on the
    first real training run (HANDOFF §B1 上机发现). Convert any numpy/torch
    scalar to a Python primitive here; lists/arrays via ``.tolist()``.
    """
    # numpy scalar (int64/float32/bool_ ...) — use .item()
    if hasattr(obj, "item") and not isinstance(obj, (list, tuple, dict, str)):
        try:
            return obj.item()
        except (ValueError, AttributeError):
            pass
    # numpy array / torch tensor → list
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except (ValueError, AttributeError):
            pass
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


class ArtifactWriter:
    """写一次训练 run 的全部标准产物。

    Args:
        run_dir: $LAPO_HOME/outputs/<run_id>，不存在则创建。
    """

    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)
        self._run_state: dict = {}

    # ---- run.json（原子写 + 增量更新）----
    def write_run_json(self, *, run_id: str, status: str, policy: str,
                       strategy: Optional[str], dataset: str, num_steps: int,
                       device: str, config_path: Optional[str] = None,
                       error: Optional[str] = None) -> None:
        """训练开始时写一次 run.json（status=running）。"""
        self._run_state = {
            "run_id": run_id, "status": status, "policy": policy,
            "strategy": strategy, "dataset": dataset, "config_path": config_path,
            "created_at": _now_iso(), "started_at": _now_iso(), "ended_at": None,
            "current_step": 0, "num_steps": num_steps, "device": device,
            "last_checkpoint": None, "error": error,
            "metrics_path": "metrics.jsonl",
            "schema_version": SCHEMA_VERSION,
        }
        self._atomic_write_json("run.json", self._run_state)

    def update_run_json(self, **fields) -> None:
        """更新 run.json 的部分字段（如 current_step / status）。原子覆写。"""
        if not self._run_state:
            raise RuntimeError("run.json 未初始化，先调 write_run_json")
        self._run_state.update(fields)
        self._atomic_write_json("run.json", self._run_state)

    # ---- run_config.yaml ----
    def write_run_config(self, cfg: Any) -> None:
        """写解析后冻结的完整配置快照（便于复现 + 前端展示）。"""
        import yaml
        import dataclasses

        def _to_dict(o):
            if dataclasses.is_dataclass(o):
                return {k: _to_dict(v) for k, v in dataclasses.asdict(o).items()}
            return o

        data = {
            "policy": cfg.policy_name,
            "strategy": cfg.strategy_name,
            "dataset": _to_dict(cfg.dataset),
            "training": _to_dict(cfg.training),
            "policy_overrides": cfg.policy_overrides,
        }
        (self.run_dir / "run_config.yaml").write_text(
            yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8",
        )

    # ---- metrics.jsonl（append-only）----
    def append_metrics(self, *, step: int, loss: Any,
                       extra: Optional[dict] = None) -> None:
        """每步 append 一行 JSON。loss 转 float（detach 防计算图）；extra 任意键合并。"""
        _loss = float(loss.detach() if hasattr(loss, "detach") else loss)
        record = {"step": step, "loss": _loss, "timestamp": _now_iso()}
        if extra:
            # 确保值可 JSON 序列化（tensor → item）
            record.update({k: (_v.item() if hasattr(_v, "item") else _v)
                           for k, _v in extra.items()})
        with open(self.run_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")

    # ---- model_info.json ----
    def write_model_info(self, model: Any, *, policy: str,
                         base_model: Optional[str], param_groups: list[dict],
                         dtype: str = "bfloat16",
                         freeze_policy: Optional[dict] = None) -> None:
        """从 model 直接统计参数量 + 冻结比例。param_groups 是分组描述。"""
        all_params = list(model.parameters())
        total = sum(p.numel() for p in all_params)
        trainable = sum(p.numel() for p in all_params if p.requires_grad)
        self._write_json("model_info.json", {
            "policy": policy, "base_model": base_model,
            "total_params": total, "trainable_params": trainable,
            "trainable_pct": round(100.0 * trainable / total, 2) if total else 0.0,
            "param_groups": param_groups, "dtype": dtype,
            "freeze_policy": freeze_policy or {},
            "pretrained_source": base_model,
            "schema_version": SCHEMA_VERSION,
        })

    # ---- model_graph.json ----
    def write_model_graph(self, graph: dict, *, policy: str) -> None:
        graph = {**graph, "policy": policy, "schema_version": SCHEMA_VERSION}
        self._write_json("model_graph.json", graph)

    # ---- dataset_info.json ----
    def write_dataset_info(self, ds_meta: Any, *, repo_id: str,
                           delta_timestamps: Optional[dict] = None) -> None:
        """从 lerobot DatasetMeta 提取数据集全景信息。尽力提取，缺失字段为 None。"""
        features = getattr(ds_meta, "features", {}) or {}
        info = getattr(ds_meta, "info", {}) or {}

        def _stats(key):
            stats = getattr(ds_meta, "stats", {}) or {}
            s = stats.get(key, {}) or {}
            return {k: list(v) if hasattr(v, "tolist") else v for k, v in s.items()}

        self._write_json("dataset_info.json", {
            "repo_id": repo_id,
            "format_version": str(info.get("codebase_version", "3.0")),
            "robot_type": info.get("robot_type"),
            "fps": info.get("fps"),
            "total_episodes": info.get("total_episodes"),
            "total_frames": info.get("total_frames"),
            "features": features,
            "cameras": [k.split(".")[-1] for k in features
                        if k.startswith("observation.images.")],
            "action_stats": _stats("action"),
            "state_stats": _stats("observation.state"),
            "delta_timestamps_used": delta_timestamps or {},
            "schema_version": SCHEMA_VERSION,
        })

    # ---- env_info.json（接受 dict，env_check 模块负责收集）----
    def write_env_info(self, env: dict) -> None:
        self._write_json("env_info.json", {**env, "schema_version": SCHEMA_VERSION})

    # ---- diff.json ----
    def write_diff(self, *, baseline_run: Optional[str], changes: list[dict]) -> None:
        self._write_json("diff.json", {
            "baseline_run": baseline_run, "changes": changes,
            "schema_version": SCHEMA_VERSION,
        })

    # ---- checkpoint（需 torch）----
    def save_checkpoint(self, model: Any, *, step: Optional[int] = None,
                        optimizer: Any = None, tag: str = "final") -> Path:
        """保存 model.safetensors + training_state.json。需 torch + safetensors。

        自动 unwrap FSDP/DDP wrapper（防御性：engine 已传 unwrap 的，
        这里再剥一层保证 state_dict key 无前缀，lerobot from_pretrained 兼容）。
        """
        import torch

        # 防御性 unwrap（FSDP._orig_mod / DDP.module / 原样）
        from lapo.train.distributed import unwrap
        model = unwrap(model)

        tag = f"step_{step}" if step is not None else tag
        ckpt_dir = self.run_dir / "checkpoints" / tag
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        try:
            # save_model 自动处理 tied weights（如 Gemma 的 lm_head/embed_tokens）；
            # save_file 遇共享内存的 tensor 会报 RuntimeError。
            from safetensors.torch import save_model
            save_model(model, str(ckpt_dir / "model.safetensors"))
        except ImportError:
            torch.save(state_dict, str(ckpt_dir / "model.pt"))

        ts: dict = {"step": step}
        if optimizer is not None:
            ts["lr_groups"] = {f"g{i}": g.get("lr") for i, g in enumerate(optimizer.param_groups)}
        (ckpt_dir / "training_state.json").write_text(
            json.dumps(ts, indent=2), encoding="utf-8",
        )
        rel = f"checkpoints/{tag}"
        self.update_run_json(last_checkpoint=rel)
        return ckpt_dir

    # ---- helpers ----
    def _write_json(self, name: str, data: dict) -> None:
        """普通 JSON 写（非原子）。用于一次性写的产物。

        ``default=_json_default`` 让 numpy/torch 标量自动转 Python 原语——
        lerobot dataset meta 常带 numpy.int64，否则真训练写 dataset_info.json
        会崩（HANDOFF §B1）。
        """
        (self.run_dir / name).write_text(
            json.dumps(data, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )

    def _atomic_write_json(self, name: str, data: dict) -> None:
        """写临时文件 + os.replace，保证 run.json 不出现半写损坏。"""
        target = self.run_dir / name
        fd, tmp = tempfile.mkstemp(dir=self.run_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=_json_default)
            os.replace(tmp, target)
        except Exception:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
