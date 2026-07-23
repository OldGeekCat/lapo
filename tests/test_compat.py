"""compat.py 测试。

核心逻辑（import_lerobot 路径选择、shape 过滤）用纯 Python 对象测，不依赖
torch/draccus/lerobot。需要重依赖的函数（load_xvla_config 等）用 mock 拦截。
"""
import json
import sys
from unittest.mock import MagicMock, patch

import pytest

from lapo.train.compat import (
    import_lerobot,
    _filter_state_dict_by_shape,
    _shape_of,
    load_xvla_config,
    load_state_dict_shape_filtered,
    build_policy_for,
    LEROBOT_VERSION,
)


# ---------- import_lerobot（双路径，#12）----------

def test_import_lerobot_new_path_when_available():
    """lerobot.policies 可 import → 返回 'lerobot'（新路径风格）。"""
    fake_pkg = MagicMock()
    fake_policies = MagicMock()
    with patch.dict(sys.modules, {"lerobot": fake_pkg, "lerobot.policies": fake_policies}):
        assert import_lerobot() == "lerobot"


def test_import_lerobot_falls_back_to_common():
    """lerobot.policies 不可 import 但 lerobot.common.policies 可 → 返回 'lerobot.common'。"""
    # 让 lerobot.policies 的 import 抛 ImportError（设 None → import 语句触发 ImportError）
    fake_pkg = MagicMock()
    fake_common_pkg = MagicMock()
    fake_common_policies = MagicMock()
    with patch.dict(sys.modules, {
        "lerobot": fake_pkg,
        "lerobot.policies": None,                       # 新路径 import 失败
        "lerobot.common": fake_common_pkg,
        "lerobot.common.policies": fake_common_policies,
    }):
        assert import_lerobot() == "lerobot.common"


def test_lerobot_version_constant():
    assert LEROBOT_VERSION == "0.4.4"


# ---------- _shape_of（duck typing）----------

class _FakeTensor:
    """模拟有 .shape 属性的对象（duck typing torch.Tensor）。"""
    def __init__(self, shape):
        self.shape = tuple(shape)


def test_shape_of_faketensor():
    assert _shape_of(_FakeTensor([2, 3])) == (2, 3)


def test_shape_of_plain_tuple():
    assert _shape_of((4, 5)) == (4, 5)


def test_shape_of_scalar():
    assert _shape_of(42) == ()


# ---------- _filter_state_dict_by_shape（核心纯逻辑）----------

def test_filter_keeps_matching_keys():
    state_dict = {"a": _FakeTensor([2, 3]), "b": _FakeTensor([4])}
    model_state = {"a": _FakeTensor([2, 3]), "b": _FakeTensor([4])}
    filtered, skipped = _filter_state_dict_by_shape(state_dict, model_state)
    assert set(filtered.keys()) == {"a", "b"}
    assert skipped == []


def test_filter_skips_shape_mismatch():
    state_dict = {
        "weight": _FakeTensor([2, 3]),   # 匹配
        "bias": _FakeTensor([99]),        # 不匹配（model 是 [2]）
    }
    model_state = {"weight": _FakeTensor([2, 3]), "bias": _FakeTensor([2])}
    filtered, skipped = _filter_state_dict_by_shape(state_dict, model_state)
    assert "weight" in filtered
    assert "bias" not in filtered           # 被过滤
    assert len(skipped) == 1
    assert "bias" in skipped[0]
    assert "99" in skipped[0]               # 报告里含 ckpt shape


def test_filter_keeps_model_missing_keys():
    """state_dict 里有 model 没有的 key → 保留（交给 strict=False 处理 unexpected）。"""
    state_dict = {"a": _FakeTensor([2, 3]), "extra": _FakeTensor([1])}
    model_state = {"a": _FakeTensor([2, 3])}
    filtered, skipped = _filter_state_dict_by_shape(state_dict, model_state)
    assert "extra" in filtered              # model 没有 → 不过滤（shape 无法比对）
    assert skipped == []


# ---------- load_xvla_config（draccus，#8，mock）----------

def test_load_xvla_config_strips_type_field(tmp_path):
    """'type' 字段必须被剥离后再 parse。用 mock draccus 验证。"""
    config_json = {"type": "xvla", "chunk_size": 30, "action_mode": "ee6d"}
    (tmp_path / "config.json").write_text(json.dumps(config_json))

    captured = {}

    fake_draccus = MagicMock()

    def fake_parse(cls, path, args=None):
        # 读被 parse 的临时文件，确认 type 已被剥离
        with open(path) as f:
            data = json.load(f)
        captured["data"] = data
        return MagicMock(name="parsed_config")

    fake_draccus.parse = fake_parse

    # config_type 当 context manager（draccus.config_type("json") 返回的 cm）
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=None)
    cm.__exit__ = MagicMock(return_value=None)
    fake_draccus.config_type = MagicMock(return_value=cm)

    config_cls = MagicMock()
    with patch.dict(sys.modules, {"draccus": fake_draccus}):
        result = load_xvla_config(config_cls, str(tmp_path))

    assert result is not None
    assert "type" not in captured["data"], "type 字段必须被剥离"
    assert captured["data"]["chunk_size"] == 30


# ---------- load_state_dict_shape_filtered（#9，需 torch）----------

def test_load_state_dict_shape_filtered_skips_mismatch():
    """需要 torch。本机若无 torch 则 skip。"""
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    from lapo.train.compat import load_state_dict_shape_filtered

    model = nn.Linear(4, 2)
    good_weight = model.state_dict()["weight"].clone()
    state_dict = {
        "weight": good_weight,          # shape [2,4] 匹配
        "bias": torch.zeros(99),         # shape [99] 不匹配 → 应被跳过
    }
    load_state_dict_shape_filtered(model, state_dict)
    # bias 未被覆盖（仍保持原 shape [2]，不是 [99]）
    assert model.bias.shape == (2,)


# ---------- build_policy_for（短名解析 + compat 路由）----------

def test_build_policy_for_unknown_short_name_raises():
    """未注册的短名 → ValueError。"""
    reg = MagicMock()
    reg.get_policy.return_value = None
    with pytest.raises(ValueError, match="未注册"):
        build_policy_for("ghost", reg, MagicMock())


def test_build_policy_for_resolves_short_name_to_config_cls():
    """已注册短名 → 取 config_cls → import 实例化 → 调 make_policy。

    用一个真实可 import 的简单 config 类 + mock make_policy 验证编排。
    """
    import sys
    import types

    # 造一个假的 config 类（真实可 import）
    fake_mod = types.ModuleType("_fake_cfg_mod")
    class _FakeConfig:
        def __init__(self):
            self.dtype = "float32"
    fake_mod._FakeConfig = _FakeConfig
    sys.modules["_fake_cfg_mod"] = fake_mod

    try:
        reg = MagicMock()
        entry = MagicMock()
        entry.config_cls = "_fake_cfg_mod._FakeConfig"
        reg.get_policy.return_value = entry

        fake_policy = MagicMock(name="policy")
        with patch("lapo.train.compat.make_policy", return_value=fake_policy) as mp:
            result = build_policy_for("act", reg, MagicMock(), overrides={"dtype": "bfloat16"})

        assert result is fake_policy
        # make_policy 被调用，第一个参数是实例化的 config
        call_cfg = mp.call_args.args[0]
        assert isinstance(call_cfg, _FakeConfig)
        assert call_cfg.dtype == "bfloat16"  # override 生效
    finally:
        del sys.modules["_fake_cfg_mod"]


def test_build_policy_for_custom_dotted_path_skips_registry():
    """自定义完整路径（含点）→ 不查 registry，直接 import。"""
    import sys
    import types

    fake_mod = types.ModuleType("_custom_mod")
    class _MyConfig:
        def __init__(self):
            self.x = 1
    fake_mod._MyConfig = _MyConfig
    sys.modules["_custom_mod"] = fake_mod

    try:
        reg = MagicMock()
        with patch("lapo.train.compat.make_policy", return_value="policy") as mp:
            build_policy_for("_custom_mod._MyConfig", reg, MagicMock())
        # registry.get_policy 不应被调用（自定义路径直 import）
        reg.get_policy.assert_not_called()
        assert isinstance(mp.call_args.args[0], _MyConfig)
    finally:
        del sys.modules["_custom_mod"]


def test_build_policy_for_xvla_routes_through_load_xvla_config():
    """policy 名含 'xvla' → 走 load_xvla_config（绕 draccus bug）+ shape 过滤。

    需 torch（monkey-patch load_state_dict）。mock 掉 load_xvla_config 和 make_policy。
    """
    torch = pytest.importorskip("torch")

    reg = MagicMock()
    entry = MagicMock()
    entry.config_cls = "lerobot.policies.xvla.configuration_xvla.XVLAConfig"
    reg.get_policy.return_value = entry

    fake_cfg = MagicMock(name="xvla_cfg")
    fake_policy = MagicMock(name="policy")
    fake_policy.load_state_dict = MagicMock()  # 防止真调

    with patch("lapo.train.compat.load_xvla_config", return_value=fake_cfg) as lxc, \
         patch("lapo.train.compat.make_policy", return_value=fake_policy) as mp:
        build_policy_for("xvla", reg, MagicMock(),
                         overrides={"base_model": "lerobot/xvla-folding"})

    # load_xvla_config 被调用（绕过 draccus bug）
    lxc.assert_called_once()
    # make_policy 被调用，参数是 load_xvla_config 的返回
    assert mp.call_args.args[0] is fake_cfg
