"""lr train CLI 测试。薄壳，用 CliRunner 调，不触发真实训练（mock service 层）。"""
from unittest.mock import patch

from typer.testing import CliRunner

from lapo.train.cli import app

runner = CliRunner()


def test_help_lists_subcommands():
    r = runner.invoke(app, ["--help"])
    assert r.exit_code == 0
    assert "train" in r.stdout
    assert "registry" in r.stdout
    assert "env" in r.stdout


def test_registry_list_runs_with_builtins(tmp_path, monkeypatch):
    """registry list 不崩溃，且能显示内置条目。"""
    monkeypatch.setenv("LR_HOME", str(tmp_path))
    r = runner.invoke(app, ["registry", "list"])
    assert r.exit_code == 0
    assert "act" in r.stdout or "default" in r.stdout


def test_registry_add_policy_then_list(tmp_path, monkeypatch):
    """add-policy → list 能看到。"""
    monkeypatch.setenv("LR_HOME", str(tmp_path))
    r = runner.invoke(app, ["registry", "add-policy", "myp",
                            "--config-class", "pkg.MyConfig",
                            "--trait", "is_transformer"])
    assert r.exit_code == 0
    r2 = runner.invoke(app, ["registry", "list", "--type", "policy"])
    assert r2.exit_code == 0
    assert "myp" in r2.stdout


def test_registry_add_policy_unknown_trait_errors(tmp_path, monkeypatch):
    """用未登记的 trait → 非零退出。"""
    monkeypatch.setenv("LR_HOME", str(tmp_path))
    r = runner.invoke(app, ["registry", "add-policy", "bad",
                            "--config-class", "x",
                            "--trait", "totally_made_up"])
    assert r.exit_code != 0


def test_registry_remove_trait_roundtrip(tmp_path, monkeypatch):
    """add-trait → remove trait → list 不再含。"""
    monkeypatch.setenv("LR_HOME", str(tmp_path))
    runner.invoke(app, ["registry", "add-trait", "tmp_trait"])
    r = runner.invoke(app, ["registry", "remove", "tmp_trait", "--type", "trait"])
    assert r.exit_code == 0
    # 再删一次应失败（已不存在）
    r2 = runner.invoke(app, ["registry", "remove", "tmp_trait", "--type", "trait"])
    assert r2.exit_code != 0


def test_registry_remove_builtin_trait_fails(tmp_path, monkeypatch):
    """内置 trait 不可删 → 非零退出。"""
    monkeypatch.setenv("LR_HOME", str(tmp_path))
    r = runner.invoke(app, ["registry", "remove", "has_soft_prompts",
                            "--type", "trait"])
    assert r.exit_code != 0


def test_env_command_runs():
    """env 命令不崩溃，输出含 python。"""
    r = runner.invoke(app, ["env"])
    assert r.exit_code == 0
    assert "python" in r.stdout


def test_env_json_command():
    """env --json 输出合法 JSON。"""
    import json
    r = runner.invoke(app, ["env", "--json"])
    assert r.exit_code == 0
    data = json.loads(r.stdout)
    assert "env" in data
    assert "checks" in data
    assert "python" in data["env"]


def test_train_command_calls_run_training(tmp_path, monkeypatch):
    """train 子命令调 run_training（mock 掉避免真实训练）。"""
    monkeypatch.setenv("LR_HOME", str(tmp_path))
    config_path = tmp_path / "exp.yaml"
    config_path.write_text("policy: act\ndataset:\n  repo_id: x/y\n")

    # run_training 在 train() 内部从 services.training import，patch 源模块
    with patch("lapo.train.services.training.run_training",
               return_value=tmp_path / "fake_run") as mock_run:
        r = runner.invoke(app, ["train", "--config", str(config_path)])
    assert r.exit_code == 0
    assert mock_run.called
