"""X-VLA SP 策略 + builtin 注册表测试。

required_traits 和 builtin 字典纯逻辑全测；build_optimizer 需 torch（skip）。
"""
from unittest.mock import MagicMock

import pytest

from lapo.train.config import RunConfig, DatasetConfig, TrainingConfig
from lapo.train.strategies.xvla_sp import XVLASoftPromptStrategy
from lapo.train.policies.builtin import BUILTIN_POLICIES, BUILTIN_STRATEGIES


def _cfg():
    return RunConfig(
        policy_name="xvla", strategy_name=None,
        dataset=DatasetConfig(repo_id="x/y"),
        training=TrainingConfig(lr=1e-4, weight_decay=1.0,
                                lr_vlm_scale=0.1, lr_soft_prompt_scale=1.0),
    )


def test_xvla_sp_required_traits():
    assert XVLASoftPromptStrategy(_cfg()).required_traits() == {"has_soft_prompts"}


def test_build_optimizer_three_param_groups():
    """三档差异学习率。需 torch。"""
    torch = pytest.importorskip("torch")
    s = XVLASoftPromptStrategy(_cfg())
    policy = MagicMock()
    vlm_p = torch.nn.Parameter(torch.randn(2))
    sp_p = torch.nn.Parameter(torch.randn(2))
    other_p = torch.nn.Parameter(torch.randn(2))
    policy.named_parameters.return_value = [
        ("vlm.encoder.weight", vlm_p),
        ("transformer.soft_prompt_hub.weight", sp_p),
        ("transformer.blocks.0.weight", other_p),
    ]
    opt = s.build_optimizer(policy)
    by_name = {g["name"]: g for g in opt.param_groups}
    assert len(opt.param_groups) == 3
    assert by_name["vlm"]["lr"] == pytest.approx(1e-5)           # 1e-4 * 0.1
    assert by_name["vlm"]["weight_decay"] == pytest.approx(0.1)  # 1.0 * 0.1
    assert by_name["soft_prompts"]["lr"] == pytest.approx(1e-4)
    assert by_name["other"]["lr"] == pytest.approx(1e-4)
    assert by_name["other"]["weight_decay"] == pytest.approx(1.0)


def test_build_optimizer_skips_frozen():
    """requires_grad=False 的参数被排除。需 torch。"""
    torch = pytest.importorskip("torch")
    s = XVLASoftPromptStrategy(_cfg())
    policy = MagicMock()
    trainable = torch.nn.Parameter(torch.randn(2))
    frozen = torch.nn.Parameter(torch.randn(2))
    frozen.requires_grad = False
    policy.named_parameters.return_value = [
        ("vlm.frozen", frozen), ("other.trainable", trainable),
    ]
    opt = s.build_optimizer(policy)
    all_p = [p for g in opt.param_groups for p in g["params"]]
    assert len(all_p) == 1
    assert trainable in all_p


def test_build_optimizer_empty_group_dropped():
    """某组无参数 → 该组不出现（避免空 param_group）。需 torch。"""
    torch = pytest.importorskip("torch")
    s = XVLASoftPromptStrategy(_cfg())
    policy = MagicMock()
    # 只有 other 组的参数，vlm/sp 组为空
    p = torch.nn.Parameter(torch.randn(2))
    policy.named_parameters.return_value = [("transformer.weight", p)]
    opt = s.build_optimizer(policy)
    group_names = {g["name"] for g in opt.param_groups}
    assert group_names == {"other"}  # vlm/sp 空组被丢


def test_builtin_policies_contain_act_and_xvla():
    assert "act" in BUILTIN_POLICIES
    assert "xvla" in BUILTIN_POLICIES
    assert "has_soft_prompts" in BUILTIN_POLICIES["xvla"].traits
    assert BUILTIN_POLICIES["xvla"].default_strategy == "xvla_sp"
    assert "is_diffusion" in BUILTIN_POLICIES["diffusion"].traits


def test_builtin_strategies_contain_default_and_xvla_sp():
    assert "default" in BUILTIN_STRATEGIES
    assert "xvla_sp" in BUILTIN_STRATEGIES
    assert BUILTIN_STRATEGIES["xvla_sp"].required_traits == {"has_soft_prompts"}
    assert BUILTIN_STRATEGIES["default"].required_traits == set()


def test_builtin_strategy_cls_path_resolvable():
    """builtin 的 cls_path 必须能 import（防止拼写错误）。

    lapo/sbvla 策略模块顶层 import torch，无 torch 环境整体 skip。
    """
    pytest.importorskip("torch")
    import importlib
    for name, entry in BUILTIN_STRATEGIES.items():
        mod_path, _, cls_name = entry.cls_path.rpartition(".")
        cls = getattr(importlib.import_module(mod_path), cls_name)
        assert cls is not None, f"{name}: {entry.cls_path} 无法 import"


def test_builtin_policy_config_cls_well_formed():
    """所有 builtin policy 的 config_cls 是点分路径（可被 importlib 解析的结构）。"""
    for name, entry in BUILTIN_POLICIES.items():
        assert "." in entry.config_cls, f"{name} config_cls 必须是完整路径"
        # act/diffusion/smolvla/xvla 指向 lerobot；sbvla/lapo 指向本仓库的 config 类。
        assert entry.config_cls.startswith(("lerobot.policies.", "lapo.train.")), (
            f"{name} 应指向 lerobot.policies 或 lapo.train"
        )
