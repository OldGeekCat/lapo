"""TrainStrategy 测试。

纯逻辑钩子（required_traits / build_scheduler / on_step_end / should_save /
describe_graph / 自定义覆写继承）不依赖 torch，全测。build_optimizer /
compute_loss 需要 torch，本机若无则 skip。
"""
from unittest.mock import MagicMock

import pytest

from lapo.train.strategy import TrainStrategy, DefaultStrategy, StepContext
from lapo.train.config import RunConfig, TrainingConfig, DatasetConfig


def _cfg(save_every=0):
    return RunConfig(
        policy_name="act",
        strategy_name=None,
        dataset=DatasetConfig(repo_id="x/y"),
        training=TrainingConfig(lr=1e-4, weight_decay=0.0,
                                lr_vlm_scale=0.1, save_every=save_every),
    )


# ---------- 纯逻辑钩子 ----------

def test_required_traits_default_empty():
    assert DefaultStrategy(_cfg()).required_traits() == set()


def test_build_scheduler_default_none():
    assert DefaultStrategy(_cfg()).build_scheduler(MagicMock()) is None


def test_on_step_end_default_noop():
    s = DefaultStrategy(_cfg())
    ctx = StepContext(step=1, policy=MagicMock(), optimizer=MagicMock(),
                      loss=MagicMock(), batch={})
    s.on_step_end(1, ctx)  # 不抛异常
    assert ctx.metrics == {}  # 默认不写


def test_should_save_default_respects_save_every():
    cfg = _cfg(save_every=100)
    s = DefaultStrategy(cfg)
    assert s.should_save(0) is False     # step 1 (0-based 0)
    assert s.should_save(99) is True     # step 100
    assert s.should_save(199) is True    # step 200


def test_should_save_zero_means_never_mid():
    """save_every=0 → 永不中途存（仅结束存）。"""
    s = DefaultStrategy(_cfg(save_every=0))
    for step in [0, 1, 99, 1000]:
        assert s.should_save(step) is False


def test_describe_graph_default_none():
    assert DefaultStrategy(_cfg()).describe_graph(MagicMock()) is None


def test_custom_strategy_only_overrides_build_optimizer():
    """自定义策略只覆写一个方法，其余继承默认。"""
    class MyStrategy(TrainStrategy):
        def build_optimizer(self, policy):
            return "custom-opt"
        # compute_loss/on_step_end/... 全继承默认

    s = MyStrategy(_cfg())
    assert s.build_optimizer(MagicMock()) == "custom-opt"
    # 继承的默认行为
    assert s.required_traits() == set()
    assert s.build_scheduler(MagicMock()) is None
    assert s.describe_graph(MagicMock()) is None


def test_custom_strategy_overrides_required_traits():
    class MyStrategy(TrainStrategy):
        def required_traits(self):
            return {"custom_trait"}
    assert MyStrategy(_cfg()).required_traits() == {"custom_trait"}


def test_step_context_metrics_writable():
    """StepContext.metrics 默认空 dict，可写入。"""
    ctx = StepContext(step=0, policy=MagicMock(), optimizer=MagicMock(),
                      loss=MagicMock(), batch={})
    ctx.metrics["custom"] = 42
    assert ctx.metrics == {"custom": 42}


# ---------- 需要 torch 的钩子 ----------

def test_build_optimizer_default_all_params_same_lr():
    """默认: AdamW, 所有可训练参数同 lr。需 torch。"""
    torch = pytest.importorskip("torch")
    s = DefaultStrategy(_cfg())
    policy = MagicMock()
    p1 = torch.nn.Parameter(torch.randn(2))
    p2 = torch.nn.Parameter(torch.randn(2))
    frozen = torch.nn.Parameter(torch.randn(2))
    frozen.requires_grad = False
    policy.parameters.return_value = [p1, p2, frozen]
    opt = s.build_optimizer(policy)
    assert len(opt.param_groups) == 1
    assert opt.param_groups[0]["lr"] == 1e-4
    # 冻结参数被排除
    assert len(opt.param_groups[0]["params"]) == 2


def test_compute_loss_default_calls_policy_forward():
    """默认: policy.forward(batch) 取 loss。需 torch（loss 是 tensor）。"""
    torch = pytest.importorskip("torch")
    s = DefaultStrategy(_cfg())
    policy = MagicMock()
    policy.forward.return_value = torch.tensor(1.5)
    loss = s.compute_loss(policy, {"x": 1})
    assert loss.item() == 1.5
    policy.forward.assert_called_once_with({"x": 1})


def test_compute_loss_default_unwraps_tuple_loss():
    """lerobot 规范：训练时 policy.forward 常返回 (loss, loss_dict)。

    ACT 就是这样（modeling_act.py:162 `return loss, loss_dict`）。
    DefaultStrategy.compute_loss 必须从 tuple 里取第一个元素（loss tensor），
    否则 engine 拿到 tuple 调 .backward() 会崩 AttributeError（HANDOFF §B1）。
    """
    torch = pytest.importorskip("torch")
    s = DefaultStrategy(_cfg())
    policy = MagicMock()
    loss_tensor = torch.tensor(2.5, requires_grad=True)
    policy.forward.return_value = (loss_tensor, {"l1": 1.0, "l2": 1.5})
    loss = s.compute_loss(policy, {"x": 1})
    # 返回的应是单个 tensor，可 .backward()
    assert isinstance(loss, torch.Tensor)
    assert loss.item() == 2.5
    loss.backward()  # 不抛即通过


def test_compute_loss_default_handles_list_return():
    """list 返回也按同规则取第一个（防御性）。"""
    torch = pytest.importorskip("torch")
    s = DefaultStrategy(_cfg())
    policy = MagicMock()
    policy.forward.return_value = [torch.tensor(3.5), "extra"]
    loss = s.compute_loss(policy, {"x": 1})
    assert isinstance(loss, torch.Tensor)
    assert loss.item() == 3.5
