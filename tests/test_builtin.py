"""内置 policy registry 完整性测试。

回归 HANDOFF §B1 上机发现的真实 bug：lerobot 0.4.4 把 policy config 放进
``configuration_<policy>`` 子模块，不再从包 ``__init__`` 重新导出，于是旧的注册
路径 ``lerobot.policies.act.ACTConfig`` 一 import 就 ``ImportError``/``AttributeError``。

本测试在有 lerobot 的环境里跑：对每个内置 policy 的 ``config_cls`` 做真 import +
getattr，确保注册路径始终与已装的 lerobot 版本一致。
"""
import importlib

import pytest

from lapo.train.policies.builtin import BUILTIN_POLICIES


def _has_lerobot() -> bool:
    try:
        importlib.import_module("lerobot.policies")  # noqa: F401
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _has_lerobot(),
    reason="需 lerobot 才能验证 config_cls 路径（服务器跑；本机 Mac skip）",
)


@pytest.mark.parametrize("name", sorted(BUILTIN_POLICIES))
def test_builtin_policy_config_cls_importable(name):
    """每个内置 policy 的 config_cls 路径在已装的 lerobot 里能 import + getattr。

    这是 build_policy_for 默认路径的第一步（compat._instantiate_config）。
    路径写错（如旧的 lerobot.policies.act.ACTConfig）会在这里直接暴露。
    """
    entry = BUILTIN_POLICIES[name]
    module_path, _, cls_name = entry.config_cls.rpartition(".")
    mod = importlib.import_module(module_path)
    assert hasattr(mod, cls_name), (
        f"{name}: {entry.config_cls} 的模块 {module_path} 里没有属性 {cls_name}。"
        f"lerobot 0.4.4 把 config 放进 configuration_<policy> 子模块——"
        f"检查 builtin.py 的 config_cls 是否对齐当前 lerobot 版本。"
    )
