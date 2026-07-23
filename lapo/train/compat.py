"""lerobot 0.4.4 兼容 hack 收容所。

所有 lerobot 版本相关 hack 集中在此。升级 lerobot 只改本文件。
覆盖 TROUBLESHOOTING 的 #8(draccus type)、#9(strict load)、#12(import 双路径)。

设计要点：核心过滤逻辑用 duck typing（不直接依赖 torch 张量），使其在
无 torch 的环境也可单测；只有真正调用 torch API 的地方才延迟 import torch。
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, get_type_hints


LEROBOT_VERSION = "0.4.4"


def import_lerobot() -> str:
    """双路径 import（#12）。返回路径风格。

    优先尝试新路径 'lerobot'，失败回退旧路径 'lerobot.common'。
    供其它函数拼接子模块路径用。
    """
    try:
        import lerobot.policies  # noqa: F401
        return "lerobot"
    except ImportError:
        import lerobot.common.policies  # noqa: F401
        return "lerobot.common"


def _module_path(sub: str) -> str:
    """根据 lerobot 路径风格拼接子模块路径。"""
    return f"{import_lerobot()}.{sub}"


def _resolve_base_model(model_id: str) -> str:
    """把 base_model 解析到本地路径（若有），否则原样返回 repo_id。

    解析顺序：
      1. ``model_id`` 本身是已存在的本地目录 → 直接返回
      2. ``$LAPO_HOME/models/downloaded/<basename>/`` 存在且有 config.json →
         返回该本地目录（lr model download 的产物）
      3. 否则原样返回 ``model_id``，走 HuggingFace Hub 下载

    这样训练 config 里写 ``base_model: lerobot/xvla-folding`` 时，若已用
    ``lr model download hf lerobot/xvla-folding`` 下载到本地，会优先用本地副本，
    离线/弱网环境也能训练。
    """
    from pathlib import Path
    p = Path(model_id)
    if p.is_dir():
        return model_id  # 已是本地目录
    # 尝试 lr model 管理的 downloaded 树
    basename = model_id.split("/")[-1]
    try:
        from lapo.paths import models_dir
        local = models_dir() / "downloaded" / basename
    except Exception:
        return model_id  # LAPO_HOME 不可用，放弃本地解析
    if local.is_dir() and (local / "config.json").exists():
        return str(local)
    return model_id


def make_policy(policy_cfg: Any, ds_meta: Any, rename_map: dict[str, str] | None = None) -> Any:
    """lerobot make_policy 包装（兼容双路径）。

    延迟 import：本函数调用时才拉起 lerobot，避免模块导入期触发重依赖。
    """
    import importlib
    mod = importlib.import_module(_module_path("policies.factory"))
    return mod.make_policy(policy_cfg, ds_meta=ds_meta, rename_map=rename_map)


def load_xvla_config(config_cls: type, model_id: str) -> Any:
    """绕过 draccus 'type' 字段 bug（#8）。

    lerobot 0.4.4 的 from_pretrained 先 draccus.parse 原始 config.json 再 pop 'type'，
    但 parse 时就因 'type' 崩溃。这里手动读 config.json、剥离 'type'、再 parse。

    model_id 先经 ``_resolve_base_model`` 解析：若 ``lr model download`` 已下载
    到本地 ``models/downloaded/<name>/``，优先用本地副本（离线友好）。
    """
    import draccus

    model_id = _resolve_base_model(model_id)
    config_path = Path(model_id)
    if config_path.is_dir():
        config_file = str(config_path / "config.json")
    else:
        from huggingface_hub import hf_hub_download
        config_file = hf_hub_download(repo_id=model_id, filename="config.json")

    with open(config_file) as f:
        config = json.load(f)
    config.pop("type", None)  # 剥离 draccus 保留字段

    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as f:
        json.dump(config, f)
        tmp_path = f.name
    try:
        with draccus.config_type("json"):
            return draccus.parse(config_cls, tmp_path, args=[])
    finally:
        os.unlink(tmp_path)


def _filter_state_dict_by_shape(state_dict: dict, model_state: dict) -> tuple[dict, list[str]]:
    """纯逻辑：过滤掉 model_state 中 shape 不匹配的 key。

    用 duck typing（只要对象有 .shape 属性），不直接依赖 torch。
    返回 (过滤后的 dict, 被跳过的 key 描述列表)。
    """
    filtered: dict = {}
    skipped: list[str] = []
    for k, v in state_dict.items():
        if k in model_state:
            model_shape = _shape_of(model_state[k])
            ckpt_shape = _shape_of(v)
            if model_shape != ckpt_shape:
                skipped.append(f"{k}: ckpt {list(ckpt_shape)} vs model {list(model_shape)}")
                continue
        filtered[k] = v
    return filtered, skipped


def _shape_of(t: Any) -> tuple:
    """取 shape（duck typing，支持 torch.Tensor / MagicMock 带 .shape / 纯 tuple）。"""
    s = getattr(t, "shape", t)
    if hasattr(s, "__iter__") and not isinstance(s, (str, bytes)):
        try:
            return tuple(s)
        except TypeError:
            return ()
    return ()


def load_state_dict_shape_filtered(module: Any, state_dict: dict) -> None:
    """跨任务 shape 不匹配过滤（#9）的便利调用形式。

    直接对 module 调用：过滤 shape 不匹配的 key 后，用原始 load_state_dict
    (strict=False) 加载。等价于把 _shape_filtered_load_state_dict 作为
    monkey-patch 应用一次。
    """
    _shape_filtered_load_state_dict(module, state_dict, strict=False, assign=False)


def _shape_filtered_load_state_dict(module: Any, state_dict: dict,
                                    strict: bool = False,
                                    assign: bool = False) -> Any:
    """torch.nn.Module.load_state_dict 的过滤替代版（签名兼容，供 monkey-patch 用）。

    lerobot 0.4.4 的 from_pretrained 硬编码 strict=True。torch 2.10 的
    strict=False 不再忽略 shape mismatch（只忽略 missing/unexpected keys）。
    本函数先过滤掉 shape 不匹配的 key，再调原始 load_state_dict(strict=False)。
    """
    import torch

    orig = torch.nn.Module.load_state_dict
    model_state: dict = {}
    model_state.update(dict(module.named_parameters()))
    model_state.update(dict(module.named_buffers()))

    filtered, skipped = _filter_state_dict_by_shape(state_dict, model_state)
    if skipped:
        import logging
        logging.info("compat: skipping %d shape-mismatched params:\n  %s",
                     len(skipped), "\n  ".join(skipped))
    return orig(module, filtered, strict=False, assign=assign)


def make_policy_safe(policy_cfg: Any, ds_meta: Any) -> Any:
    """make_policy 的安全包装（普通 policy 直通）。

    仅供已实例化 config 对象的场景。短名→config 实例化 + 按 policy 名路由
    compat hack 的完整链路见 build_policy_for。
    """
    return make_policy(policy_cfg, ds_meta)


def _instantiate_config(config_cls_path: str, overrides: dict,
                        base_model: str | None = None) -> Any:
    """import config 类并实例化，应用 overrides。

    config 类实例化的方式与 lerobot 一致：先 from_pretrained(base_model) 或
    直接构造，再用 overrides 覆盖属性。对 xvla 这类有 from_pretrained bug 的，
    由 build_policy_for 路由到 load_xvla_config 绕过。
    """
    import importlib
    module_path, _, cls_name = config_cls_path.rpartition(".")
    config_cls = getattr(importlib.import_module(module_path), cls_name)

    if base_model is not None:
        # 多数 lerobot config 支持 from_pretrained（加载预训练默认值）
        # 先解析本地副本（lr model download 产物），离线友好
        resolved = _resolve_base_model(base_model)
        try:
            cfg = config_cls.from_pretrained(resolved)
        except Exception:
            cfg = config_cls()
    else:
        cfg = config_cls()

    for k, v in (overrides or {}).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def build_policy_for(policy_name: str, registry: Any, ds_meta: Any,
                     overrides: dict | None = None,
                     rename_map: dict[str, str] | None = None) -> Any:
    """完整 policy 构建链路：短名/路径 → config 实例化 → 按 policy 路由 compat。

    这是 DefaultStrategy.build_policy 的默认实现调用的入口，集中处理：
    1. 短名经 registry 解析为 config_cls 路径（自定义路径直用）。
    2. 实例化 config（xvla 走 load_xvla_config 绕过 draccus bug）。
    3. make_policy 加载权重（xvla 额外套 load_state_dict_shape_filtered 过滤 shape）。

    所有 lerobot 0.4.4 bug 规避（#8/#9/#12）在此触发，主路径零 hack。
    """
    overrides = overrides or {}

    # 1. 短名 → config_cls 路径
    if "." in policy_name:
        # 自定义完整路径
        config_cls_path = policy_name
        short_name = policy_name.rsplit(".", 1)[-1].lower()
        base_model = overrides.get("base_model")
    else:
        entry = registry.get_policy(policy_name)
        if entry is None:
            raise ValueError(f"policy '{policy_name}' 未注册")
        config_cls_path = entry.config_cls
        short_name = policy_name.lower()
        base_model = overrides.get("base_model")

    # 2. config 实例化（xvla 走 load_xvla_config 绕 draccus 'type' bug，#8）
    if "xvla" in short_name and base_model is not None:
        import importlib
        module_path, _, cls_name = config_cls_path.rpartition(".")
        config_cls = getattr(importlib.import_module(module_path), cls_name)
        policy_cfg = load_xvla_config(config_cls, base_model)
        for k, v in overrides.items():
            if hasattr(policy_cfg, k):
                setattr(policy_cfg, k, v)
    else:
        policy_cfg = _instantiate_config(config_cls_path, overrides, base_model)

    # 3. make_policy（xvla 额外套 shape 过滤，#9）
    if "xvla" in short_name:
        import torch
        orig = torch.nn.Module.load_state_dict
        torch.nn.Module.load_state_dict = _shape_filtered_load_state_dict
        try:
            return make_policy(policy_cfg, ds_meta, rename_map=rename_map)
        finally:
            torch.nn.Module.load_state_dict = orig
    return make_policy(policy_cfg, ds_meta, rename_map=rename_map)
