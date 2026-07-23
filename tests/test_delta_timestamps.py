"""delta_timestamps 解析测试 —— 序列策略数据加载的关键缺口。

回归 HANDOFF §B1 上机发现的真 bug：``_build_dataloader`` 直接
``LeRobotDataset(repo_id, root)`` 没传 ``delta_timestamps``，导致 ACT/diffusion/
xvla 这类 chunk 类策略拿不到 action chunk（``(B, S, D)``），只拿到单帧
``(B, D)``，forward 时 ``torch.cat`` 维度不匹配崩。

lerobot 的 canonical 解析是 ``lerobot.datasets.factory.resolve_delta_timestamps``
（从 policy config 的 ``*_delta_indices`` 属性 × 1/fps 得到 seconds）。
lrt 必须在 build dataloader 时调它，把结果传给 ``LeRobotDataset(delta_timestamps=...)``。

本测试在有 lerobot 的环境（服务器）跑；本机 Mac skip。
"""
import importlib

import pytest

import lapo.train.services.training as tmod


def _has_lerobot() -> bool:
    try:
        importlib.import_module("lerobot.policies")  # noqa: F401
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _has_lerobot(),
    reason="需 lerobot 才能验证 delta_timestamps 解析（服务器跑）",
)


def test_resolve_delta_timestamps_from_policy_cfg():
    """_resolve_delta_timestamps 把 policy config 的 *_delta_indices 转成
    {feature_key: [seconds]} 格式（lerobot canonical 语义）。

    ACT 的 action_delta_indices = list(range(chunk_size))；在 fps=30 下应转成
    [0/30, 1/30, ...]。observation_delta_indices=None → 不给 observation key。
    """
    from lerobot.policies.act.configuration_act import ACTConfig

    cfg = ACTConfig()
    cfg.input_features = {}  # 避免 validate_features 报错（本测试不碰 features）

    class _FakeMeta:
        fps = 30
        features = {
            "observation.state": {},
            "action": {},
            "observation.images.global": {},
        }

    dt = tmod._resolve_delta_timestamps(cfg, _FakeMeta())
    assert dt is not None
    assert "action" in dt
    # ACT chunk_size 默认 100 → 100 个 delta，0/30..99/30
    assert len(dt["action"]) == cfg.chunk_size
    assert dt["action"][0] == pytest.approx(0.0)
    assert dt["action"][1] == pytest.approx(1 / 30)
    # observation_delta_indices=None → 不含 observation key
    assert "observation.state" not in dt


def test_resolve_delta_timestamps_none_when_policy_has_no_indices():
    """policy config 三个 *_delta_indices 都 None → 返回 None（不阻塞非 chunk 策略）。"""

    class _NoChunkCfg:
        observation_delta_indices = None
        action_delta_indices = None
        reward_delta_indices = None

    class _FakeMeta:
        fps = 30
        features = {"observation.state": {}, "action": {}}

    assert tmod._resolve_delta_timestamps(_NoChunkCfg(), _FakeMeta()) is None


def test_build_dataloader_passes_delta_timestamps_to_dataset(monkeypatch):
    """_build_dataloader 构造 LeRobotDataset 时必须传 delta_timestamps。

    用 monkeypatch 拦截 LeRobotDataset，捕获 kwargs，断言 delta_timestamps 在里面。
    """
    captured = {}

    class _FakeDataset:
        def __init__(self, repo_id, root=None, delta_timestamps=None, **kw):
            captured["delta_timestamps"] = delta_timestamps
            captured["repo_id"] = repo_id

        def __len__(self):
            return 4

    class _FakeDatasetMod:
        LeRobotDataset = _FakeDataset

    class _FakeDatasetCfg:
        repo_id = "org/task"
        root = "/tmp/x"

    class _FakeTrainingCfg:
        batch_size = 1
        num_workers = 0
        fsdp = False
        ddp = False

    monkeypatch.setattr(tmod, "_lerobot_dataset_module", lambda: _FakeDatasetMod)
    monkeypatch.setattr(tmod, "_resolve_delta_timestamps",
                        lambda policy_cfg, ds_meta: {"action": [0.0, 0.03]})

    # 给一个假的 policy_cfg（实际调用方会传真 policy.config）
    fake_policy_cfg = object()
    tmod._build_dataloader(_FakeDatasetCfg(), _FakeTrainingCfg(),
                           policy_cfg=fake_policy_cfg, ds_meta=object())
    assert captured["delta_timestamps"] == {"action": [0.0, 0.03]}, (
        "_build_dataloader 必须把 _resolve_delta_timestamps 的结果透传给 LeRobotDataset"
    )
