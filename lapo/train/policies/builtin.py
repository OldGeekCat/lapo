"""内置 policy/strategy 短名 → 注册条目。

随包分发，用户可通过 registry CLI 覆盖同名条目。
类路径指向 lerobot 0.4.4 的实际模块（兼容 lerobot.common.* / lerobot.*
双前缀由 compat.import_lerobot 在运行期决定，这里写新路径风格）。
"""
from __future__ import annotations

from lapo.train.registry import PolicyEntry, StrategyEntry
from lapo.train.strategies.xvla_sp import XVLASoftPromptStrategy
from lapo.train.strategies.xvla_ee6d import XVLAEE6DStrategy
from lapo.train.strategies.smolvla_sp import SmolVLAStrategy
from lapo.train.strategy import DefaultStrategy

# SB-VLA：独立 policy 家族（config_cls 指向本仓库的 SBVLAConfig）。
# 实际 policy 构建（加载 Florence2 + 注入 SBVLAHead）由 SBVLAStrategy.build_policy
# 完全接管，不走 lerobot make_policy。config_cls 这里仅为注册占位。
SBVLA_CONFIG_CLS = "lapo.train.policies.sb.config.SBVLAConfig"


BUILTIN_POLICIES: dict[str, PolicyEntry] = {
    "act": PolicyEntry(
        name="act",
        # lerobot 0.4.4 把 config 放进 configuration_<policy> 子模块（不再
        # 从包 __init__ 重新导出）。旧路径 lerobot.policies.act.ACTConfig 在
        # 0.4.4 是 ImportError；上机验证（HANDOFF §B1）抓出这条。
        config_cls="lerobot.policies.act.configuration_act.ACTConfig",
        traits={"is_transformer"},
    ),
    "diffusion": PolicyEntry(
        name="diffusion",
        config_cls="lerobot.policies.diffusion.configuration_diffusion.DiffusionConfig",
        traits={"is_diffusion"},
    ),
    "smolvla": PolicyEntry(
        name="smolvla",
        config_cls="lerobot.policies.smolvla.configuration_smolvla.SmolVLAConfig",
        traits={"is_transformer", "has_language"},
        default_strategy="smolvla_sp",
    ),
    "xvla": PolicyEntry(
        name="xvla",
        config_cls="lerobot.policies.xvla.configuration_xvla.XVLAConfig",
        traits={"has_soft_prompts", "has_language", "is_transformer"},
        default_strategy="xvla_sp",
    ),
    "sbvla": PolicyEntry(
        name="sbvla",
        # SB-VLA：薛定谔桥 VLA（encoder/g/f/SB，对齐 docs/next-gen-architecture-v2.md）。
        # config_cls 是本仓库 SBVLAConfig（非 lerobot）；build_policy 完全由
        # SBVLAStrategy 接管（加载 Florence2 + 注入 SBVLAHead），不走 make_policy。
        config_cls=SBVLA_CONFIG_CLS,
        traits={"has_language"},
        default_strategy="sbvla",
    ),
    "lapo": PolicyEntry(
        name="lapo",
        # LAPo: Local Action Bridge with Endpoint Prior。
        # Stage 1: oracle endpoint + direct decoder, action-space loss。
        # build_policy 完全由 LapoStrategy 接管（复用 xvla-base Florence2 + DaViT）。
        config_cls="lapo.train.policies.lapo.config.LapoConfig",
        traits={"has_language"},
        default_strategy="lapo",
    ),
}

BUILTIN_STRATEGIES: dict[str, StrategyEntry] = {
    "default": StrategyEntry(
        name="default",
        cls_path=f"{DefaultStrategy.__module__}.{DefaultStrategy.__name__}",
    ),
    "xvla_sp": StrategyEntry(
        name="xvla_sp",
        cls_path=f"{XVLASoftPromptStrategy.__module__}.{XVLASoftPromptStrategy.__name__}",
        required_traits={"has_soft_prompts"},
    ),
    "smolvla_sp": StrategyEntry(
        name="smolvla_sp",
        cls_path=f"{SmolVLAStrategy.__module__}.{SmolVLAStrategy.__name__}",
        required_traits={"has_language"},
    ),
    "xvla_ee6d": StrategyEntry(
        name="xvla_ee6d",
        cls_path=f"{XVLAEE6DStrategy.__module__}.{XVLAEE6DStrategy.__name__}",
        required_traits={"has_soft_prompts"},
    ),
    "sbvla": StrategyEntry(
        name="sbvla",
        cls_path="lapo.train.strategies.sbvla.SBVLAStrategy",
        required_traits=set(),  # 独立 policy，无 trait 要求
    ),
    "lapo": StrategyEntry(
        name="lapo",
        cls_path="lapo.train.strategies.lapo.LapoStrategy",
        required_traits=set(),
    ),
}
