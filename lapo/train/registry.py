"""Registry: policy/strategy 短名注册 + trait 兼容性校验。

两类零件（policy / strategy）+ trait 词表。持久化到 $LAPO_HOME/registry/。
两层校验：注册时（default_strategy 存在性、trait 在词表中），运行时
（strategy.required_traits ⊆ policy.traits）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from lapo.train.services.registry_store import RegistryStore


class TraitError(Exception):
    """policy 与 strategy trait 不兼容。"""


@dataclass
class PolicyEntry:
    name: str
    config_cls: str                          # 完整 Python 路径
    policy_cls: Optional[str] = None         # 可选
    traits: set[str] = field(default_factory=set)
    default_strategy: Optional[str] = None
    defaults: dict = field(default_factory=dict)


@dataclass
class StrategyEntry:
    name: str
    cls_path: str                            # 完整 Python 路径
    required_traits: set[str] = field(default_factory=set)
    defaults: dict = field(default_factory=dict)


class Registry:
    """管理 policy/strategy/trait 三类注册条目 + 兼容性校验。"""

    # 内置 trait 词表（随包分发；用户可通过 add-trait 扩展）
    BUILTIN_TRAITS = {
        "has_soft_prompts", "has_language", "is_diffusion",
        "is_transformer", "supports_amp", "multi_camera",
    }

    def __init__(self, store: RegistryStore):
        self.store = store

    # ---- policy ----
    def register_policy(self, entry: PolicyEntry) -> None:
        # 校验 default_strategy 已注册（若声明了）
        if entry.default_strategy is not None:
            if self.get_strategy(entry.default_strategy) is None:
                raise ValueError(f"推荐 strategy '{entry.default_strategy}' 未注册")
        self._check_traits_known(entry.traits)
        data = self.store.load_policies()
        data[entry.name] = {
            "config_cls": entry.config_cls,
            "policy_cls": entry.policy_cls,
            "traits": sorted(entry.traits),
            "default_strategy": entry.default_strategy,
            "defaults": entry.defaults,
        }
        self.store.save_policies(data)

    def get_policy(self, name: str) -> Optional[PolicyEntry]:
        data = self.store.load_policies().get(name)
        if data is None:
            return None
        return PolicyEntry(
            name=name,
            config_cls=data["config_cls"],
            policy_cls=data.get("policy_cls"),
            traits=set(data.get("traits", [])),
            default_strategy=data.get("default_strategy"),
            defaults=data.get("defaults", {}),
        )

    def list_policies(self) -> list[str]:
        return list(self.store.load_policies().keys())

    def remove_policy(self, name: str) -> None:
        data = self.store.load_policies()
        data.pop(name, None)
        self.store.save_policies(data)

    # ---- strategy ----
    def register_strategy(self, entry: StrategyEntry) -> None:
        self._check_traits_known(entry.required_traits)
        data = self.store.load_strategies()
        data[entry.name] = {
            "cls_path": entry.cls_path,
            "required_traits": sorted(entry.required_traits),
            "defaults": entry.defaults,
        }
        self.store.save_strategies(data)

    def get_strategy(self, name: str) -> Optional[StrategyEntry]:
        data = self.store.load_strategies().get(name)
        if data is None:
            return None
        return StrategyEntry(
            name=name,
            cls_path=data["cls_path"],
            required_traits=set(data.get("required_traits", [])),
            defaults=data.get("defaults", {}),
        )

    def list_strategies(self) -> list[str]:
        return list(self.store.load_strategies().keys())

    def remove_strategy(self, name: str) -> None:
        data = self.store.load_strategies()
        data.pop(name, None)
        self.store.save_strategies(data)

    # ---- trait ----
    def register_trait(self, name: str) -> None:
        traits = self.store.load_traits()
        if name not in traits:
            traits.append(name)
            self.store.save_traits(traits)

    def list_traits(self) -> list[str]:
        """返回内置 + 用户自定义的并集（排序）。"""
        user_traits = self.store.load_traits()
        return sorted(set(self.BUILTIN_TRAITS) | set(user_traits))

    def remove_trait(self, name: str) -> bool:
        """删除用户自定义 trait。内置 trait 不可删（返回 False）。

        返回 True 表示已删除，False 表示不存在或是内置 trait。
        """
        if name in self.BUILTIN_TRAITS:
            return False  # 内置 trait 不可删
        traits = self.store.load_traits()
        if name not in traits:
            return False
        traits.remove(name)
        self.store.save_traits(traits)
        return True

    # ---- 兼容性校验（运行时）----
    def check_compatibility(self, policy_name: str, strategy_name: str) -> None:
        """strategy.required_traits ⊆ policy.traits，否则抛 TraitError。"""
        policy = self.get_policy(policy_name)
        strategy = self.get_strategy(strategy_name)
        if policy is None:
            raise ValueError(f"policy '{policy_name}' 未注册")
        if strategy is None:
            raise ValueError(f"strategy '{strategy_name}' 未注册")
        missing = strategy.required_traits - policy.traits
        if missing:
            raise TraitError(
                f"策略 '{strategy_name}' 需要 policy 具备 {sorted(missing)}，"
                f"但 policy '{policy_name}' 没有。"
                f"建议：换用具备这些特性的 policy，或去掉 strategy 用默认。"
            )

    def _check_traits_known(self, traits: set[str]) -> None:
        """注册时校验：trait 必须在词表中（内置或用户登记的）。"""
        known = set(self.list_traits())
        unknown = traits - known
        if unknown:
            raise ValueError(
                f"未知 trait: {sorted(unknown)}。"
                f"先用 `lr train registry add-trait` 登记新 trait。"
            )
