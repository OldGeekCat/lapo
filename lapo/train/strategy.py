"""TrainStrategy 钩子基类 + DefaultStrategy + StepContext。

三档钩子（详见 spec §2.1）：
  档位1 构建期: build_policy / build_optimizer / build_scheduler
  档位2 步级:   compute_loss / on_step_end / should_save / describe_graph
  档位3 循环级: train_loop（逃生舱）

所有方法有默认实现（标准 lerobot 行为）。自定义策略按需覆写。
torch 仅在 build_optimizer/compute_loss 内延迟 import，模块本身可在无 torch
环境 import（纯逻辑钩子如 required_traits/should_save 不触发 torch）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from lapo.train.config import RunConfig


@dataclass
class StepContext:
    """传给 on_step_end 的上下文。metrics 可写入，会被收入 metrics.jsonl。"""
    step: int
    policy: Any
    optimizer: Any
    loss: Any
    batch: dict
    metrics: dict = field(default_factory=dict)


class TrainStrategy:
    """训练策略基类。所有方法有默认实现，按需覆写。

    默认实现 = 标准 lerobot 训练行为（make_policy + AdamW + forward loss）。
    自定义策略只需覆写需要的方法，其余继承默认。
    """

    def __init__(self, cfg: "RunConfig", registry: Any = None):
        self.cfg = cfg
        self.registry = registry

    def required_traits(self) -> set[str]:
        """本策略对 policy 的要求（trait 校验用）。默认空集。"""
        return set()

    def build_delta_timestamps(self, ds_meta: Any) -> Any:
        """自定义 delta_timestamps（数据加载用）。默认 None = 走 lerobot
        ``resolve_delta_timestamps(policy_cfg, ds_meta)``。

        覆写场景：策略需要 delta_indices 但 policy_cfg 的只读 property 给不了
        （如 SB-VLA 需要未来帧 frame_{t+H}，而 XVLAConfig.observation_delta_indices
        硬编码 None）。返回 dict[str, list[float]]（key=feature, val=秒列表）。
        """
        return None

    # ---- 档位1：构建期 ----
    def build_policy(self, cfg: "RunConfig", ds_meta: Any) -> Any:
        """默认: compat.build_policy_for（短名→config 实例化→按 policy 路由 compat）。

        需要 self.registry 来解析短名（由 resolve_strategy 注入）。
        子类可覆写做完全自定义的权重加载。
        """
        from lapo.train.compat import build_policy_for
        return build_policy_for(
            cfg.policy_name, self.registry, ds_meta,
            overrides=cfg.policy_overrides,
            rename_map=cfg.dataset.rename_map,
        )

    def build_optimizer(self, policy: Any) -> Any:
        """默认: AdamW，所有可训练参数同 lr。"""
        import torch
        params = [p for p in policy.parameters() if p.requires_grad]
        return torch.optim.AdamW(
            params,
            lr=self.cfg.training.lr,
            weight_decay=self.cfg.training.weight_decay,
        )

    def build_scheduler(self, optimizer: Any) -> Any:
        """默认: None。覆写场景: warmup、cosine decay。"""
        return None

    # ---- 档位2：步级 ----
    def compute_loss(self, policy: Any, batch: dict) -> Any:
        """默认: policy.forward(batch) 取 loss。

        lerobot 训练时 policy.forward 规范返回值有两种：
        - 单个 loss tensor（多数 policy）
        - ``(loss, loss_dict)`` 元组（ACT 等，见 modeling_act.py:162）
        这里统一兜底：tuple/list 取第一个元素作为 loss，其余（loss_dict 等）
        丢弃。自定义策略需要拿 loss_dict 做事时覆写本方法。
        """
        out = policy.forward(batch)
        if isinstance(out, (tuple, list)):
            return out[0]
        return out

    def on_step_end(self, step: int, ctx: StepContext) -> None:
        """默认: 空操作。ctx 暴露 policy/optimizer/loss/metrics。

        覆写场景: 动态调 lr、记录自定义指标（写 ctx.metrics）、梯度诊断。
        """
        pass

    def should_save(self, step: int) -> bool:
        """默认: step % save_every == 0（save_every=0 时永不中途存，仅结束存）。

        step 是 0-based；用 (step+1) 对齐"第 N 步存"的直觉。
        """
        se = self.cfg.training.save_every
        return se > 0 and (step + 1) % se == 0

    def describe_graph(self, policy: Any) -> dict | None:
        """默认: None（触发 forward hook 自动提图）。手写覆盖返回 dict。"""
        return None

    # ---- 档位3：循环级（逃生舱）----
    def train_loop(self, engine: "TrainingEngine") -> list[dict]:
        """默认: engine.standard_loop()。覆写 = 完全接管训练。

        极少覆写；覆写时可调用 engine 基础设施（dataloader/ckpt/writer）。
        """
        return engine.standard_loop()


class DefaultStrategy(TrainStrategy):
    """显式的默认策略，便于 registry 注册为 'default' 短名。

    与 TrainStrategy 行为完全一致，单独命名是为了在 registry 里有一个
    清晰的 'default' 条目（而非直接引用基类）。
    """
    pass


if TYPE_CHECKING:
    from lapo.train.engine import TrainingEngine  # noqa: F401
