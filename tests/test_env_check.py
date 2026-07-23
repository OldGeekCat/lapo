"""env_check 测试。纯逻辑（版本解析 + 矩阵比对），无 torch 依赖，全测。"""
from lapo.train.env_check import (
    collect_env, check_env, COMPAT_MATRIX, EnvInfo, _parse_ver, _cmp,
)


def _info(**kw):
    """构造 EnvInfo，默认全 None。"""
    defaults = dict(python="3.11", torch=None, lerobot=None, transformers=None,
                    gpu_name=None, cuda=None, cudnn=None, git_commit="abc")
    defaults.update(kw)
    return EnvInfo(**defaults)


# ---------- _parse_ver ----------

def test_parse_ver_basic():
    assert _parse_ver("2.10.0") == (2, 10, 0)


def test_parse_ver_strips_local_suffix():
    """'2.10.0+cu128' → (2,10,0)，去掉 +cu128。"""
    assert _parse_ver("2.10.0+cu128") == (2, 10, 0)


def test_parse_ver_none():
    assert _parse_ver(None) == ()


def test_parse_ver_empty():
    assert _parse_ver("") == ()


# ---------- _cmp ----------

def test_cmp_in_range():
    assert _cmp((2, 10, 0), "2.6.0", "2.11.0") == "in_range"


def test_cmp_above():
    assert _cmp((2, 11, 0), "2.6.0", "2.11.0") == "above"


def test_cmp_below():
    assert _cmp((2, 5, 0), "2.6.0", "2.11.0") == "below"


def test_cmp_unknown_when_empty():
    assert _cmp((), "2.6.0", "2.11.0") == "unknown"


# ---------- collect_env ----------

def test_collect_env_has_required_fields():
    info = collect_env()
    assert info.python
    assert hasattr(info, "torch")
    assert hasattr(info, "lerobot")
    assert hasattr(info, "gpu_name")
    assert hasattr(info, "git_commit")


def test_env_info_to_dict_excludes_schema():
    """to_dict 不含 schema_version（由 writer 补）。"""
    info = _info()
    d = info.to_dict()
    assert "schema_version" not in d
    assert d["python"] == "3.11"


# ---------- check_env ----------

def test_check_env_verified_combo_ok():
    """0.4.4 + transformers 4.57.6 + torch 2.10 → 全 ok。"""
    info = _info(lerobot="0.4.4", transformers="4.57.6", torch="2.10.0")
    results = check_env(info)
    statuses = {r.name: r.status for r in results}
    assert statuses.get("transformers") == "ok"
    assert statuses.get("lerobot") == "ok"
    assert statuses.get("torch") == "ok"


def test_check_env_transformers_5x_fails():
    """transformers 5.x + lerobot 0.4.x → transformers 项 fail。"""
    info = _info(lerobot="0.4.4", transformers="5.12.1", torch="2.10.0")
    results = check_env(info)
    t = next(r for r in results if r.name == "transformers")
    assert t.status == "fail"
    assert "5.0.0" in t.message
    assert "TROUBLESHOOTING" in t.message


def test_check_env_unknown_lerobot_warns():
    info = _info(lerobot="0.9.9", transformers="4.57.6", torch="2.10.0")
    results = check_env(info)
    l = next(r for r in results if r.name == "lerobot")
    assert l.status == "warn"


def test_check_env_torch_above_warns():
    info = _info(lerobot="0.4.4", transformers="4.57.6", torch="2.12.0")
    results = check_env(info)
    t = next(r for r in results if r.name == "torch")
    assert t.status == "warn"


def test_check_env_torch_below_fails():
    info = _info(lerobot="0.4.4", transformers="4.57.6", torch="2.5.0")
    results = check_env(info)
    t = next(r for r in results if r.name == "torch")
    assert t.status == "fail"


def test_check_env_no_deps_gives_warn():
    """什么都没装 → 兜底 warn。"""
    results = check_env(_info())
    statuses = [r.status for r in results]
    assert "warn" in statuses


def test_compat_matrix_has_verified_entry():
    assert any(m["lerobot"] == "0.4.4" for m in COMPAT_MATRIX)
