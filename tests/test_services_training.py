"""services/training 编排测试。

resolve_strategy / load_registry_with_builtins 纯逻辑全测。
run_training 端到端需 torch（mock 掉 lerobot 加载 + build_policy）。
"""
from unittest.mock import MagicMock, patch

import pytest

from lapo.train.services.training import (
    resolve_strategy, run_training, load_registry_with_builtins, _make_run_id,
)
from lapo.train.config import RunConfig, DatasetConfig, TrainingConfig


# ---------- resolve_strategy（纯逻辑）----------

def test_resolve_strategy_uses_policy_default(tmp_path):
    """YAML 省略 strategy → 用 policy 的 default_strategy（xvla → xvla_sp）。"""
    reg = load_registry_with_builtins(root=tmp_path)
    cfg = RunConfig(policy_name="xvla", strategy_name=None,
                    dataset=DatasetConfig(repo_id="x/y"))
    strategy = resolve_strategy(cfg, reg)
    from lapo.train.strategies.xvla_sp import XVLASoftPromptStrategy
    assert isinstance(strategy, XVLASoftPromptStrategy)


def test_resolve_strategy_explicit_override(tmp_path):
    """显式 strategy='default' 覆盖 xvla 的推荐 xvla_sp。"""
    reg = load_registry_with_builtins(root=tmp_path)
    cfg = RunConfig(policy_name="xvla", strategy_name="default",
                    dataset=DatasetConfig(repo_id="x/y"))
    strategy = resolve_strategy(cfg, reg)
    from lapo.train.strategy import DefaultStrategy
    assert isinstance(strategy, DefaultStrategy)


def test_resolve_strategy_incompatible_raises(tmp_path):
    """act + xvla_sp → TraitError（act 无 has_soft_prompts）。"""
    reg = load_registry_with_builtins(root=tmp_path)
    cfg = RunConfig(policy_name="act", strategy_name="xvla_sp",
                    dataset=DatasetConfig(repo_id="x/y"))
    from lapo.train.registry import TraitError
    with pytest.raises(TraitError):
        resolve_strategy(cfg, reg)


def test_resolve_strategy_unknown_policy(tmp_path):
    reg = load_registry_with_builtins(root=tmp_path)
    cfg = RunConfig(policy_name="ghost", strategy_name=None,
                    dataset=DatasetConfig(repo_id="x/y"))
    with pytest.raises(ValueError, match="未注册"):
        resolve_strategy(cfg, reg)


def test_resolve_strategy_act_uses_default_when_no_recommendation(tmp_path):
    """act 无 default_strategy → 用 'default'。"""
    reg = load_registry_with_builtins(root=tmp_path)
    cfg = RunConfig(policy_name="act", strategy_name=None,
                    dataset=DatasetConfig(repo_id="x/y"))
    strategy = resolve_strategy(cfg, reg)
    from lapo.train.strategy import DefaultStrategy
    assert isinstance(strategy, DefaultStrategy)


# ---------- load_registry_with_builtins（纯逻辑）----------

def test_load_registry_seeds_builtins(tmp_path):
    reg = load_registry_with_builtins(root=tmp_path)
    policies = set(reg.list_policies())
    strategies = set(reg.list_strategies())
    assert {"act", "diffusion", "smolvla", "xvla"} <= policies
    assert {"default", "xvla_sp"} <= strategies


def test_load_registry_user_override_preserved(tmp_path):
    """用户先注册同名条目，再 load builtins → 用户条目不被覆盖。"""
    from lapo.train.registry import PolicyEntry
    reg = load_registry_with_builtins(root=tmp_path)
    # 用户覆盖 act
    reg.register_policy(PolicyEntry(
        name="act", config_cls="my.custom.ACTConfig", traits={"is_transformer"},
    ))
    # 重新 load（模拟重启）
    reg2 = load_registry_with_builtins(root=tmp_path)
    got = reg2.get_policy("act")
    # 用户条目保留（注意：register 会覆写整条，_builtin 标记丢失）
    assert got.config_cls == "my.custom.ACTConfig"


# ---------- _make_run_id ----------

def test_make_run_id_format():
    rid = _make_run_id("xvla")
    parts = rid.split("_")
    # YYYYMMDD_HHMM_<policy>_<hash>
    assert len(parts) == 4
    assert parts[2] == "xvla"
    assert len(parts[3]) == 4  # short hash


def test_make_run_id_custom_policy_uses_last_segment():
    """自定义 policy 'a.b.MyPolicy' → run_id 用 'MyPolicy'。"""
    rid = _make_run_id("a.b.MyPolicy")
    assert "MyPolicy" in rid
    assert "a.b" not in rid


# ---------- run_training（端到端，需 torch + mock lerobot）----------

def test_run_training_writes_run_dir(tmp_path):
    """端到端冒烟：mock 掉 lerobot 加载和 build_policy，验证产物齐全。"""
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    reg = load_registry_with_builtins(root=tmp_path / "reg")
    cfg = RunConfig(
        policy_name="act", strategy_name="default",
        dataset=DatasetConfig(repo_id="x/y"),
        training=TrainingConfig(num_steps=2, lr=1e-3),
    )
    fake_model = nn.Linear(4, 2)
    fake_dl = [{"x": torch.randn(4)} for _ in range(2)]
    # 不 mock _build_policy_cfg：真 ACTConfig() 实例化很快（纯 dataclass），
    # 且让这条路径真跑能顺带验证 registry→config_cls 解析没退化。
    # _resolve_delta_timestamps 必须 mock（真跑会进 lerobot factory 读 ds_meta.fps，
    # 而 ds_meta 是 MagicMock，会触发自动递归）。
    with patch("lapo.train.services.training._build_dataloader", return_value=fake_dl), \
         patch("lapo.train.services.training._load_ds_meta", return_value=MagicMock()), \
         patch("lapo.train.services.training._resolve_delta_timestamps", return_value=None), \
         patch("lapo.train.strategy.DefaultStrategy.build_policy", return_value=fake_model):
        run_dir = run_training(cfg, registry=reg,
                               outputs_root=tmp_path / "outputs")
    assert (run_dir / "run.json").exists()
    assert (run_dir / "metrics.jsonl").exists()
    assert (run_dir / "model_info.json").exists()
    assert (run_dir / "run_config.yaml").exists()
    import json
    run = json.loads((run_dir / "run.json").read_text())
    assert run["status"] == "completed"
