"""Registry 测试：注册/查询/trait 校验/持久化。纯逻辑，无 torch。"""
import pytest

from lapo.train.registry import Registry, PolicyEntry, StrategyEntry, TraitError
from lapo.train.services.registry_store import RegistryStore


def _store(tmp_path):
    return RegistryStore(root=tmp_path)


def test_register_and_get_policy(tmp_path):
    r = Registry(_store(tmp_path))
    r.register_policy(PolicyEntry(name="act", config_cls="lerobot.policies.act.ACTConfig"))
    got = r.get_policy("act")
    assert got.name == "act"
    assert got.config_cls == "lerobot.policies.act.ACTConfig"
    assert got.traits == set()
    assert got.default_strategy is None


def test_register_policy_with_traits_and_default_strategy(tmp_path):
    r = Registry(_store(tmp_path))
    # 先注册被引用的 strategy
    r.register_strategy(StrategyEntry(
        name="xvla_sp", cls_path="x", required_traits={"has_soft_prompts"},
    ))
    r.register_policy(PolicyEntry(
        name="xvla", config_cls="x.XConfig",
        traits={"has_soft_prompts", "has_language"}, default_strategy="xvla_sp",
    ))
    got = r.get_policy("xvla")
    assert got.traits == {"has_soft_prompts", "has_language"}
    assert got.default_strategy == "xvla_sp"


def test_register_policy_rejects_unknown_default_strategy(tmp_path):
    """default_strategy 指向未注册的 strategy → 报错。"""
    r = Registry(_store(tmp_path))
    with pytest.raises(ValueError, match="未注册"):
        r.register_policy(PolicyEntry(name="x", config_cls="c", default_strategy="ghost"))


def test_persistence_across_instances(tmp_path):
    """注册后新建 Registry 实例仍能读到（YAML 持久化）。"""
    store = _store(tmp_path)
    Registry(store).register_policy(PolicyEntry(name="act", config_cls="x"))
    r2 = Registry(store)
    assert r2.get_policy("act") is not None


def test_trait_check_passes(tmp_path):
    """strategy.required_traits ⊆ policy.traits → 通过（不抛异常）。"""
    r = Registry(_store(tmp_path))
    r.register_policy(PolicyEntry(name="xvla", config_cls="x", traits={"has_soft_prompts"}))
    r.register_strategy(StrategyEntry(name="xvla_sp", cls_path="x", required_traits={"has_soft_prompts"}))
    r.check_compatibility("xvla", "xvla_sp")  # 不抛即通过


def test_trait_check_fails_with_human_message(tmp_path):
    """policy 缺 trait → TraitError 含人话说明。"""
    r = Registry(_store(tmp_path))
    r.register_policy(PolicyEntry(name="act", config_cls="x", traits=set()))
    r.register_strategy(StrategyEntry(name="xvla_sp", cls_path="x", required_traits={"has_soft_prompts"}))
    with pytest.raises(TraitError) as ei:
        r.check_compatibility("act", "xvla_sp")
    assert "has_soft_prompts" in str(ei.value)
    assert "act" in str(ei.value)


def test_check_compatibility_unknown_policy(tmp_path):
    r = Registry(_store(tmp_path))
    r.register_strategy(StrategyEntry(name="default", cls_path="x"))
    with pytest.raises(ValueError, match="policy 'ghost' 未注册"):
        r.check_compatibility("ghost", "default")


def test_list_policies_and_strategies(tmp_path):
    r = Registry(_store(tmp_path))
    r.register_policy(PolicyEntry(name="act", config_cls="x"))
    r.register_policy(PolicyEntry(name="xvla", config_cls="y"))
    r.register_strategy(StrategyEntry(name="default", cls_path="z"))
    assert set(r.list_policies()) == {"act", "xvla"}
    assert set(r.list_strategies()) == {"default"}


def test_remove_policy(tmp_path):
    r = Registry(_store(tmp_path))
    r.register_policy(PolicyEntry(name="act", config_cls="x"))
    r.remove_policy("act")
    assert r.get_policy("act") is None


def test_remove_strategy(tmp_path):
    r = Registry(_store(tmp_path))
    r.register_strategy(StrategyEntry(name="default", cls_path="x"))
    r.remove_strategy("default")
    assert r.get_strategy("default") is None


def test_register_trait_and_list(tmp_path):
    r = Registry(_store(tmp_path))
    r.register_trait("custom_trait")
    all_traits = r.list_traits()
    assert "custom_trait" in all_traits
    # 内置 trait 也在
    assert "has_soft_prompts" in all_traits
    assert "is_transformer" in all_traits


def test_register_policy_rejects_unknown_trait(tmp_path):
    """注册时用未登记的 trait → 报错。"""
    r = Registry(_store(tmp_path))
    with pytest.raises(ValueError, match="未知 trait"):
        r.register_policy(PolicyEntry(name="x", config_cls="c", traits={"totally_made_up"}))


def test_builtin_traits_always_listed_even_when_store_empty(tmp_path):
    """空 store 也能列出内置 trait。"""
    r = Registry(_store(tmp_path))
    assert len(r.list_traits()) >= 6  # 至少 6 个内置


def test_remove_user_trait(tmp_path):
    """删除用户自定义 trait → 返回 True，list 不再含它。"""
    r = Registry(_store(tmp_path))
    r.register_trait("my_trait")
    assert r.remove_trait("my_trait") is True
    assert "my_trait" not in r.list_traits()


def test_remove_builtin_trait_returns_false(tmp_path):
    """删除内置 trait → 返回 False（不可删），list 仍含它。"""
    r = Registry(_store(tmp_path))
    assert r.remove_trait("has_soft_prompts") is False
    assert "has_soft_prompts" in r.list_traits()


def test_remove_nonexistent_trait_returns_false(tmp_path):
    r = Registry(_store(tmp_path))
    assert r.remove_trait("ghost") is False
