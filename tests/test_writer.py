"""ArtifactWriter 测试。

非 checkpoint 方法（run.json / metrics / run_config / model_info / dataset /
env / diff）用 MagicMock model/ds_meta，不依赖 torch，全测。checkpoint 需 torch。
"""
import json
from unittest.mock import MagicMock

import pytest

import yaml

from lapo.train.artifacts.writer import ArtifactWriter
from lapo.train.config import RunConfig, DatasetConfig


def _run_dir(tmp_path):
    return tmp_path / "test_run"


def test_write_run_json_initial(tmp_path):
    w = ArtifactWriter(_run_dir(tmp_path))
    w.write_run_json(run_id="r1", status="running", policy="act", strategy="default",
                     dataset="x/y", num_steps=10, device="cpu")
    data = json.loads((w.run_dir / "run.json").read_text())
    assert data["run_id"] == "r1"
    assert data["status"] == "running"
    assert data["num_steps"] == 10
    assert data["schema_version"] == 1
    assert data["current_step"] == 0
    assert data["metrics_path"] == "metrics.jsonl"


def test_update_run_json_atomic(tmp_path):
    w = ArtifactWriter(_run_dir(tmp_path))
    w.write_run_json(run_id="r1", status="running", policy="x", strategy=None,
                     dataset="x/y", num_steps=10, device="cpu")
    w.update_run_json(current_step=5)
    data = json.loads((w.run_dir / "run.json").read_text())
    assert data["current_step"] == 5
    assert data["status"] == "running"
    w.update_run_json(status="completed", ended_at="2026-06-17T10:00:00Z")
    data = json.loads((w.run_dir / "run.json").read_text())
    assert data["status"] == "completed"


def test_update_run_json_requires_init(tmp_path):
    """未先 write_run_json 就 update → 报错（防止半写 run.json）。"""
    w = ArtifactWriter(_run_dir(tmp_path))
    with pytest.raises(RuntimeError, match="未初始化"):
        w.update_run_json(current_step=1)


def test_append_metrics_jsonl(tmp_path):
    w = ArtifactWriter(_run_dir(tmp_path))
    w.write_run_json(run_id="r1", status="running", policy="x", strategy=None,
                     dataset="x", num_steps=10, device="cpu")
    w.append_metrics(step=1, loss=2.0, extra={"gpu_mem_mb": 100})
    w.append_metrics(step=2, loss=1.5)
    lines = (w.run_dir / "metrics.jsonl").read_text().strip().split("\n")
    assert len(lines) == 2
    d1 = json.loads(lines[0])
    assert d1["step"] == 1 and d1["loss"] == 2.0 and d1["gpu_mem_mb"] == 100
    assert "timestamp" in d1
    d2 = json.loads(lines[1])
    assert d2["step"] == 2 and d2["loss"] == 1.5


def test_append_metrics_tensor_loss_handled(tmp_path):
    """loss 若是带 .item() 的对象（如 torch loss）→ 自动转 float。需 torch。"""
    torch = pytest.importorskip("torch")
    w = ArtifactWriter(_run_dir(tmp_path))
    w.write_run_json(run_id="r1", status="running", policy="x", strategy=None,
                     dataset="x", num_steps=1, device="cpu")
    w.append_metrics(step=0, loss=torch.tensor(3.5))
    line = (w.run_dir / "metrics.jsonl").read_text().strip()
    assert json.loads(line)["loss"] == 3.5


def test_write_run_config(tmp_path):
    w = ArtifactWriter(_run_dir(tmp_path))
    cfg = RunConfig(policy_name="act", strategy_name=None,
                    dataset=DatasetConfig(repo_id="x/y"))
    w.write_run_config(cfg)
    data = yaml.safe_load((w.run_dir / "run_config.yaml").read_text())
    assert data["policy"] == "act"
    assert data["strategy"] is None
    assert data["dataset"]["repo_id"] == "x/y"


def test_write_model_info_counts(tmp_path):
    """用 MagicMock model 模拟 parameters()。验证 total/trainable 统计。"""
    w = ArtifactWriter(_run_dir(tmp_path))

    class _P:
        def __init__(self, n, trainable=True):
            self._n = n
            self.requires_grad = trainable

        def numel(self):
            return self._n

    model = MagicMock()
    model.parameters.return_value = [_P(100, True), _P(50, False)]
    w.write_model_info(model, policy="act", base_model=None,
                       param_groups=[{"name": "all", "lr_scale": 1.0}])
    data = json.loads((w.run_dir / "model_info.json").read_text())
    assert data["total_params"] == 150
    assert data["trainable_params"] == 100
    assert data["trainable_pct"] == 66.67
    assert data["policy"] == "act"
    assert data["schema_version"] == 1


def test_write_model_graph(tmp_path):
    w = ArtifactWriter(_run_dir(tmp_path))
    w.write_model_graph({"nodes": [{"id": "a"}], "edges": [], "frozen": []},
                        policy="act")
    data = json.loads((w.run_dir / "model_graph.json").read_text())
    assert data["policy"] == "act"
    assert data["nodes"] == [{"id": "a"}]
    assert data["schema_version"] == 1


def test_write_dataset_info(tmp_path):
    w = ArtifactWriter(_run_dir(tmp_path))
    ds_meta = MagicMock()
    ds_meta.features = {
        "observation.images.global": {"dtype": "video", "shape": [480, 640, 3]},
        "observation.state": {"dtype": "float32", "shape": [48]},
        "action": {"dtype": "float32", "shape": [32]},
    }
    ds_meta.info = {"fps": 30, "total_episodes": 50, "total_frames": 135000,
                    "codebase_version": "3.0", "robot_type": "openarm"}
    ds_meta.stats = {"action": {"mean": [0.1, 0.2]}, "observation.state": {"mean": [0.5]}}
    w.write_dataset_info(ds_meta, repo_id="org/task",
                         delta_timestamps={"action": [0.0, 0.1]})
    data = json.loads((w.run_dir / "dataset_info.json").read_text())
    assert data["repo_id"] == "org/task"
    assert data["fps"] == 30
    assert data["total_episodes"] == 50
    assert data["cameras"] == ["global"]
    assert data["action_stats"]["mean"] == [0.1, 0.2]
    assert data["delta_timestamps_used"]["action"] == [0.0, 0.1]
    assert data["schema_version"] == 1


def test_write_dataset_info_handles_numpy_scalars(tmp_path):
    """numpy 标量（lerobot dataset meta 实际类型）必须能 JSON 序列化。

    回归 HANDOFF §B1：真训练时 ``info.get('total_episodes')`` 是 numpy.int64，
    旧实现 ``json.dumps`` 直接崩 ``TypeError: Object of type int64 is not JSON
    serializable``。本机 mock 用纯 Python int 测不出，必须用 numpy 标量。
    """
    np = pytest.importorskip("numpy")  # 服务器/numpy 装了才跑
    w = ArtifactWriter(_run_dir(tmp_path))
    ds_meta = MagicMock()
    ds_meta.features = {
        "observation.images.global": {"dtype": "video", "shape": [480, 640, 3]},
        "action": {"dtype": "float32", "shape": [np.int32(32)]},
    }
    # 模拟 lerobot meta 的真实类型：numpy int/float 标量 + numpy array stats
    ds_meta.info = {
        "fps": np.int64(30),
        "total_episodes": np.int64(50),
        "total_frames": np.int64(135766),
        "codebase_version": "3.0",
    }
    ds_meta.stats = {
        "action": {"mean": np.array([0.1, 0.2, 0.3]), "std": np.array([0.4, 0.5])},
    }
    w.write_dataset_info(ds_meta, repo_id="org/task")  # 不抛即通过
    data = json.loads((w.run_dir / "dataset_info.json").read_text())
    assert data["fps"] == 30
    assert data["total_episodes"] == 50
    assert data["total_frames"] == 135766
    assert data["action_stats"]["mean"] == [0.1, 0.2, 0.3]


def test_write_env_info(tmp_path):
    w = ArtifactWriter(_run_dir(tmp_path))
    w.write_env_info({"python": "3.11", "torch": "2.10", "git_commit": "abc"})
    data = json.loads((w.run_dir / "env_info.json").read_text())
    assert data["python"] == "3.11"
    assert data["schema_version"] == 1


def test_write_diff_none_baseline(tmp_path):
    w = ArtifactWriter(_run_dir(tmp_path))
    w.write_diff(baseline_run=None, changes=[])
    data = json.loads((w.run_dir / "diff.json").read_text())
    assert data["baseline_run"] is None
    assert data["changes"] == []


def test_save_checkpoint(tmp_path):
    """需 torch。验证 safetensors 或 model.pt 落盘 + training_state.json。"""
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    w = ArtifactWriter(_run_dir(tmp_path))
    w.write_run_json(run_id="r1", status="running", policy="x", strategy=None,
                     dataset="x", num_steps=10, device="cpu")
    model = nn.Linear(4, 2)
    opt = torch.optim.AdamW(model.parameters())
    ckpt_dir = w.save_checkpoint(model, step=5, optimizer=opt)
    assert (ckpt_dir / "model.safetensors").exists() or (ckpt_dir / "model.pt").exists()
    assert (ckpt_dir / "training_state.json").exists()
    ts = json.loads((ckpt_dir / "training_state.json").read_text())
    assert ts["step"] == 5
    # run.json 的 last_checkpoint 更新
    run = json.loads((w.run_dir / "run.json").read_text())
    assert run["last_checkpoint"] == "checkpoints/step_5"
