"""TrainingEngine 测试。

所有测试需 torch（loss.backward / optimizer），本机无 torch 则整文件 skip。
服务器上跑。engine 的状态机（completed/failed）也通过真 torch 训练验证。
"""
import json
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from lapo.train.config import RunConfig, DatasetConfig, TrainingConfig
from lapo.train.strategy import DefaultStrategy, TrainStrategy
from lapo.train.engine import TrainingEngine


def _cfg(num_steps=2, save_every=0):
    return RunConfig(
        policy_name="act", strategy_name=None,
        dataset=DatasetConfig(repo_id="x/y"),
        training=TrainingConfig(num_steps=num_steps, save_every=save_every, lr=1e-3),
    )


def _engine(tmp_path, cfg, strategy=None):
    return TrainingEngine(
        cfg, strategy=strategy or DefaultStrategy(cfg),
        run_dir=tmp_path, registry=MagicMock(), ds_meta=MagicMock(),
        dataloader=[{"x": torch.randn(4)} for _ in range(5)],
        device="cpu",
    )


def test_standard_loop_runs_steps(tmp_path):
    cfg = _cfg(num_steps=3)
    model = nn.Linear(4, 2)
    engine = _engine(tmp_path, cfg)
    with patch.object(engine, "_build_policy", return_value=model), \
         patch.object(engine, "_extract_graph", return_value={"nodes": [], "edges": [], "frozen": []}):
        # 关掉 describe_graph 的真实 dataloader 提取
        infos = engine.standard_loop()
    assert len(infos) == 3
    assert all("loss" in i and "lr_groups" in i for i in infos)
    # metrics.jsonl 至少 1 行（num_steps=3, log_every=10, 但最后一步必写）
    lines = (tmp_path / "metrics.jsonl").read_text().strip().split("\n")
    assert len(lines) >= 1


def test_run_sets_completed_status(tmp_path):
    cfg = _cfg(num_steps=1)
    model = nn.Linear(4, 2)
    engine = _engine(tmp_path, cfg)
    with patch.object(engine, "_build_policy", return_value=model), \
         patch.object(engine, "_extract_graph", return_value={"nodes": [], "edges": [], "frozen": []}):
        engine.run()
    data = json.loads((tmp_path / "run.json").read_text())
    assert data["status"] == "completed"
    assert data["current_step"] == 1
    assert data["last_checkpoint"] == "checkpoints/final"


def test_run_sets_failed_status_on_error(tmp_path):
    cfg = _cfg(num_steps=1)
    engine = _engine(tmp_path, cfg)
    with patch.object(engine, "_build_policy", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError, match="boom"):
            engine.run()
    data = json.loads((tmp_path / "run.json").read_text())
    assert data["status"] == "failed"
    assert "boom" in data["error"]


def test_on_step_end_custom_metrics_recorded(tmp_path):
    """自定义策略 on_step_end 写 ctx.metrics → 进 metrics.jsonl。"""
    cfg = _cfg(num_steps=1)

    class MyStrategy(TrainStrategy):
        def on_step_end(self, step, ctx):
            ctx.metrics["custom_metric"] = 42

    model = nn.Linear(4, 2)
    engine = _engine(tmp_path, cfg, strategy=MyStrategy(cfg))
    with patch.object(engine, "_build_policy", return_value=model), \
         patch.object(engine, "_extract_graph", return_value={"nodes": [], "edges": [], "frozen": []}):
        engine.run()
    line = (tmp_path / "metrics.jsonl").read_text().strip()
    assert json.loads(line)["custom_metric"] == 42


def test_should_save_triggers_intermediate_checkpoint(tmp_path):
    """save_every=2 → step 2（0-based 1）存中间 checkpoint。"""
    cfg = _cfg(num_steps=2, save_every=2)
    model = nn.Linear(4, 2)
    engine = _engine(tmp_path, cfg)
    with patch.object(engine, "_build_policy", return_value=model), \
         patch.object(engine, "_extract_graph", return_value={"nodes": [], "edges": [], "frozen": []}):
        engine.run()
    # 应有 step_2 和 final 两个 checkpoint 目录
    ckpts = list((tmp_path / "checkpoints").iterdir())
    ckpt_names = {p.name for p in ckpts}
    assert "step_2" in ckpt_names
    assert "final" in ckpt_names


def test_run_writes_static_artifacts(tmp_path):
    """run() 前置产物齐全：run.json / run_config.yaml / dataset_info.json。"""
    cfg = _cfg(num_steps=1)
    model = nn.Linear(4, 2)
    engine = _engine(tmp_path, cfg, )
    engine._env_info = {"python": "3.11"}
    with patch.object(engine, "_build_policy", return_value=model), \
         patch.object(engine, "_extract_graph", return_value={"nodes": [], "edges": [], "frozen": []}):
        engine.run()
    assert (tmp_path / "run.json").exists()
    assert (tmp_path / "run_config.yaml").exists()
    assert (tmp_path / "dataset_info.json").exists()
    assert (tmp_path / "env_info.json").exists()
