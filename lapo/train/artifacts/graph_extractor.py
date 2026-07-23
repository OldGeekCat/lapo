"""forward hook 自动提取模型拓扑图。

注册 register_forward_hook 到每个顶层子模块，跑一个微型 batch，记录每个子模块的
输入输出 shape + 参数量 + 是否冻结，自动构建 model_graph.json 数据。

torch 在 extract_graph 内延迟 import，模块本身可在无 torch 环境 import。
_shape_no_batch 是纯函数，单测无需 torch。
"""
from __future__ import annotations

from typing import Any


def _shape_no_batch(t: Any) -> list:
    """去掉 batch 维（index 0）后的 shape。纯函数，duck typing。

    支持任何带 .shape 属性的对象（torch.Tensor / numpy / MagicMock）。
    无 .shape 时返回 []。
    """
    shape = getattr(t, "shape", None)
    if shape is None:
        return []
    try:
        s = list(shape)
    except TypeError:
        return []
    return s[1:] if len(s) > 1 else s


def extract_graph(model: Any, sample_input: dict) -> dict:
    """跑一次前向，用 hook 提取每个顶层子模块的结构信息。

    Args:
        model: nn.Module（policy）。
        sample_input: 喂给 model.forward 的 kwargs（如 {"x": tensor}）。

    Returns:
        {nodes, edges, frozen} 结构（schema_version 由 writer 补）。
    """
    import torch

    top_modules = {name: m for name, m in model.named_children()}
    records: dict[str, dict] = {}
    handles = []

    def make_hook(name: str):
        def hook(module, inp, out):
            inp_shape = _shape_no_batch(inp[0]) if inp else []
            out_shape = _shape_no_batch(out)
            records[name] = {
                "id": name,
                "type": module.__class__.__name__.lower(),
                "class": module.__class__.__name__,
                "params": sum(p.numel() for p in module.parameters(recurse=False)),
                "in_shape": inp_shape,
                "out_shape": out_shape,
                "trainable": any(p.requires_grad for p in module.parameters(recurse=False)),
            }
        return hook

    for name, mod in top_modules.items():
        handles.append(mod.register_forward_hook(make_hook(name)))

    try:
        with torch.no_grad():
            model(**sample_input)
    finally:
        for h in handles:
            h.remove()

    nodes = list(records.values())

    # 边：按 hook 调用顺序（= 前向执行顺序）连接相邻顶层模块。
    # 对线性前向结构足够；复杂分支结构需要 tensor identity 追踪（未来增强）。
    ordered = [n["id"] for n in nodes]
    edges = [{"from": ordered[i], "to": ordered[i + 1], "tensor": "auto"}
             for i in range(len(ordered) - 1)]

    frozen = [n["id"] for n in nodes if not n["trainable"]]

    return {"nodes": nodes, "edges": edges, "frozen": frozen}
