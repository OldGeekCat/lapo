"""graph_extractor 测试。

_shape_no_batch 纯函数全测；extract_graph 端到端需 torch，本机无则 skip。
"""
import pytest

from lapo.train.artifacts.graph_extractor import _shape_no_batch, extract_graph


class _T:
    """模拟 tensor（带 .shape）。"""
    def __init__(self, shape):
        self.shape = tuple(shape)


# ---------- _shape_no_batch（纯函数）----------

def test_shape_no_batch_drops_batch_dim():
    """4D [B,C,H,W] → [C,H,W]。"""
    assert _shape_no_batch(_T([2, 3, 224, 224])) == [3, 224, 224]


def test_shape_no_batch_single_dim_kept():
    """1D [N] → [N]（len<=1 时原样返回，不丢）。"""
    assert _shape_no_batch(_T([7])) == [7]


def test_shape_no_batch_empty_when_no_shape_attr():
    class NoShape:
        pass
    assert _shape_no_batch(NoShape()) == []


def test_shape_no_batch_handles_list_shape():
    """shape 是 list 而非 tuple 也能处理。"""
    assert _shape_no_batch(_T([4, 8])) == [8]


# ---------- extract_graph（需 torch）----------

def test_extract_graph_basic_structure():
    """encoder→head 两层网络的节点和边。需 torch。"""
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(3, 8)
            self.head = nn.Linear(8, 2)

        def forward(self, x):
            return self.head(self.encoder(x))

    graph = extract_graph(Tiny(), sample_input={"x": torch.randn(1, 3)})
    node_ids = {n["id"] for n in graph["nodes"]}
    assert "encoder" in node_ids
    assert "head" in node_ids
    edge_pairs = {(e["from"], e["to"]) for e in graph["edges"]}
    assert ("encoder", "head") in edge_pairs


def test_extract_graph_captures_param_counts():
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(3, 8)  # 3*8 + 8 = 32
            self.head = nn.Linear(8, 2)     # 8*2 + 2 = 18

        def forward(self, x):
            return self.head(self.encoder(x))

    graph = extract_graph(Tiny(), sample_input={"x": torch.randn(1, 3)})
    by_id = {n["id"]: n for n in graph["nodes"]}
    assert by_id["encoder"]["params"] == 32
    assert by_id["head"]["params"] == 18


def test_extract_graph_captures_shapes():
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(3, 8)
            self.head = nn.Linear(8, 2)

        def forward(self, x):
            return self.head(self.encoder(x))

    graph = extract_graph(Tiny(), sample_input={"x": torch.randn(1, 3)})
    by_id = {n["id"]: n for n in graph["nodes"]}
    assert by_id["encoder"]["in_shape"] == [3]
    assert by_id["encoder"]["out_shape"] == [8]


def test_extract_graph_frozen_flag():
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(3, 8)
            self.head = nn.Linear(8, 2)

        def forward(self, x):
            return self.head(self.encoder(x))

    m = Tiny()
    m.encoder.requires_grad_(False)
    graph = extract_graph(m, sample_input={"x": torch.randn(1, 3)})
    assert "encoder" in graph["frozen"]
    assert "head" not in graph["frozen"]
