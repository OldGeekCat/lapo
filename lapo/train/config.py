"""配置 dataclass + YAML 加载。

v2: YAML → RunConfig。registry 短名解析（短名→类）在 registry/service 层做，
本模块只负责把 YAML 文件解析成 dataclass，不触及 lerobot/torch。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class DatasetConfig:
    repo_id: str
    root: Optional[str] = None
    task_instruction: Optional[str] = None
    image_key: str = "observation.images.global"
    state_key: str = "observation.state"
    action_key: str = "action"
    rename_map: Optional[dict[str, str]] = None
    val_ratio: float = 0.0  # 0=不切验证集；如 0.1=留 10% episodes 做验证（按 episode 边界切，不切断序列）


@dataclass
class TrainingConfig:
    num_steps: int = 1
    lr: float = 1e-4
    weight_decay: float = 0.0
    grad_clip_norm: Optional[float] = 1.0
    batch_size: int = 1
    num_workers: int = 0
    save_every: int = 0          # 0 = 只在结束时存
    log_every: int = 10
    val_every: int = 0           # 0 = 不验证；如 200=每 200 步跑一次验证集 eval
    lr_vlm_scale: float = 0.1
    lr_soft_prompt_scale: float = 1.0
    seed: int = 42
    device: str = "auto"
    dtype: str = "bfloat16"
    output_dir: Optional[str] = None  # None = $LAPO_HOME/outputs/<run_id>
    # ---- AdamW betas（对齐 XVLA 官方 0.9/0.99）----
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    # ---- LR Scheduler (cosine warmup + decay，对齐 XVLA 官方) ----
    # scheduler=None 时恒定 lr。设了 warmup_steps/decay_steps 则启用 cosine。
    scheduler_warmup_steps: int = 0       # 0 = 不 warmup
    scheduler_decay_steps: int = 0        # 0 = 不 decay
    scheduler_decay_lr: float = 2.5e-6    # decay 终点 lr（论文默认）
    # ---- FSDP 多卡训练 ----
    fsdp: bool = False                 # 开启 FSDP（torchrun 环境下用）
    fsdp_sharding: str = "full"        # full / shard_grad_op / no_shard
    grad_checkpoint: bool = False      # 梯度检查点（省激活换计算，允许更大 batch）
    # ---- DDP 多卡训练（V100 等 32G 卡：fp16 + DDP + grad ckpt）----
    ddp: bool = False                  # 开启 DDP（与 fsdp 互斥，ddp 优先）
    # ---- Gradient Accumulation ----
    grad_accumulation_steps: int = 1   # 梯度累积步数（有效 batch = batch_size * world_size * grad_accum)
    # ---- 断点续训 ----
    resume_from: Optional[str] = None  # checkpoint 目录路径（含 model.safetensors + training_state.json）
    # ---- chunk 连续性损失（抑制推理时 chunk 间跳变/抖动）----
    smooth_loss_weight: float = 0.0    # 0=不加；0.1=推荐（惩罚 chunk 内 action 二阶差分=加速度）
    # ---- 散热休息（热保护）----
    cooldown_temp: int = 80            # GPU 温度阈值（°C），任一卡达到则触发休息
    cooldown_seconds: int = 1200       # 休息时长（秒，默认 20 分钟）


@dataclass
class RunConfig:
    policy_name: str             # 内置短名 或 自定义完整路径
    strategy_name: Optional[str] # None = 用 policy 推荐策略
    dataset: DatasetConfig = field(default_factory=lambda: DatasetConfig(""))
    training: TrainingConfig = field(default_factory=TrainingConfig)
    policy_overrides: dict[str, Any] = field(default_factory=dict)

    @property
    def is_custom_policy(self) -> bool:
        """True = policy_name 是完整 Python 路径（含点），需 importlib 加载。"""
        return "." in self.policy_name


def load_config(path: str | Path) -> RunConfig:
    """加载 YAML → RunConfig。

    不解析 registry 短名（那是 registry 层的职责）。只做字段映射 + 默认值填充，
    忽略未知字段（不报错，便于向前兼容）。
    """
    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    ds_raw = raw.get("dataset", {})
    t_raw = raw.get("training", {})
    ds_fields = {f.name for f in DatasetConfig.__dataclass_fields__.values()}
    t_fields = {f.name for f in TrainingConfig.__dataclass_fields__.values()}
    return RunConfig(
        policy_name=raw["policy"],
        strategy_name=raw.get("strategy"),
        dataset=DatasetConfig(**{k: v for k, v in ds_raw.items() if k in ds_fields}),
        training=TrainingConfig(**{k: v for k, v in t_raw.items() if k in t_fields}),
        policy_overrides=raw.get("policy_overrides", {}),
    )
