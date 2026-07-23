"""RunConfig + load_config 测试。"""
from lapo.train.config import RunConfig, load_config


def test_load_minimal_config(tmp_path):
    """最简 YAML: policy + dataset.repo_id，其余用默认值。"""
    yaml_text = """
policy: act
dataset:
  repo_id: openarm/test
"""
    p = tmp_path / "exp.yaml"
    p.write_text(yaml_text)
    cfg = load_config(p)
    assert cfg.policy_name == "act"
    assert cfg.dataset.repo_id == "openarm/test"
    # 默认值
    assert cfg.training.num_steps == 1
    assert cfg.training.lr == 1e-4
    assert cfg.training.batch_size == 1
    assert cfg.strategy_name is None  # None = 用 policy 推荐策略


def test_load_full_config(tmp_path):
    """显式覆盖所有字段。"""
    yaml_text = """
policy: xvla
strategy: xvla_sp
dataset:
  repo_id: org/task
  root: /data/task
  task_instruction: "fold shirt"
  image_key: observation.images.global
  state_key: observation.state
  action_key: action
policy_overrides:
  dtype: float32
  freeze_vision_encoder: true
training:
  num_steps: 2000
  lr: 0.0001
  batch_size: 2
  save_every: 500
  lr_vlm_scale: 0.1
  lr_soft_prompt_scale: 1.0
"""
    p = tmp_path / "exp.yaml"
    p.write_text(yaml_text)
    cfg = load_config(p)
    assert cfg.policy_name == "xvla"
    assert cfg.strategy_name == "xvla_sp"
    assert cfg.dataset.task_instruction == "fold shirt"
    assert cfg.training.num_steps == 2000
    assert cfg.training.save_every == 500
    assert cfg.policy_overrides["dtype"] == "float32"


def test_custom_policy_dotted_path(tmp_path):
    """自定义 policy 用完整 Python 路径。"""
    yaml_text = """
policy: mypkg.MyPolicy
dataset:
  repo_id: org/task
"""
    p = tmp_path / "exp.yaml"
    p.write_text(yaml_text)
    cfg = load_config(p)
    assert cfg.policy_name == "mypkg.MyPolicy"
    assert cfg.is_custom_policy is True


def test_builtin_policy_short_name_not_custom(tmp_path):
    """内置短名（无点）→ is_custom_policy False。"""
    yaml_text = """
policy: act
dataset:
  repo_id: org/task
"""
    p = tmp_path / "exp.yaml"
    p.write_text(yaml_text)
    cfg = load_config(p)
    assert cfg.is_custom_policy is False
