"""Schrödinger Bridge 训练 loss —— 可复用技术能力（policy-agnostic）。

对齐 flows/flow_matching.py 的设计风格：纯编排对象，把"采噪声 → 桥积分 →
IMLE 选最近 → 归约 loss"串起来。不持有 policy；policy 由调用方传入。

与 FlowMatchingLoss 的区别：
  - FM: 学速度场 v(x_t,t)，单峰路径（模式平均，压死多模态）
  - SB: 二阶加速度场 a(x_t,t) + IMLE 多模态（K 候选各认领一 mode）

桥数学（Chronos 源码 ground-truth，见 policies/sb/bridge.py 文档）：
  - 三次样条插值 q_t = q0 + 3δt² − 2δt³（双零点边界）
  - quartic 噪声包络 σ_t = 16·sigma_peak·(t(1−t))²
  - 简单 Euler 积分 v+=a·dt; q+=v·dt
  - IMLE: torch.min 取每个 expert 的最近候选，只惩罚该对
"""
from __future__ import annotations

from typing import Any, Callable, Optional

# 默认超参（对齐 Chronos 源码）
DEFAULT_SIGMA_PEAK = 0.03   # Chronos sigma_t = 16 * 0.03 * (t*(1-t))**2 的 0.03
DEFAULT_K_P = 1.0           # PD 位置反馈
DEFAULT_K_D = 1.0           # PD 速度反馈
DEFAULT_STEPS = 5           # 推理积分步数


class SchrodingerBridgeLoss:
    """薛定谔桥 + IMLE 训练 loss 编排对象。

    纯编排：采噪声/时间 → 调 policy.forward(observation, actions, ...) → 归约。
    policy 需支持 forward 返回 {"imle": ..., "force": ...}（对齐 SBVLAHead.sb.loss）。

    Args:
        sigma_peak: 噪声包络峰值系数
        k_p / k_d: 桥力 PD 反馈系数
        steps: 推理积分步数（训练时用单步近似）
        noise_sampler: 可选自定义噪声采样器（测试用）
    """

    def __init__(
        self,
        *,
        sigma_peak: float = DEFAULT_SIGMA_PEAK,
        k_p: float = DEFAULT_K_P,
        k_d: float = DEFAULT_K_D,
        steps: int = DEFAULT_STEPS,
        noise_sampler: Optional[Callable[..., Any]] = None,
    ):
        self.sigma_peak = sigma_peak
        self.k_p = k_p
        self.k_d = k_d
        self.steps = steps
        self._noise_sampler = noise_sampler

    def sample_noise(self, shape: tuple, device: Any = None) -> Any:
        """采 noise ~ N(0,1)，shape = actions.shape。"""
        if self._noise_sampler is not None:
            return self._noise_sampler(shape, device)
        import torch
        n = torch.normal(mean=0.0, std=1.0, size=tuple(shape), dtype=torch.float32)
        if device is not None:
            n = n.to(device)
        return n

    def __call__(
        self,
        policy: Any,
        observation: Any,
        actions: Any,
        z_t: Any = None,
        z_goal: Any = None,
        *,
        noise: Any = None,
    ) -> Any:
        """计算 SB+IMLE loss 标量。

        Args:
            policy: 需支持 forward(observation, actions, z_t, z_goal) →
                    {"imle": L_imle, "force": L_force}
            observation: 观测（含任务意图等）
            actions: (batch, chunk, dim) 专家动作块
            z_t / z_goal: latent 锚点（两端物理约束）
            noise: 可选注入（测试用）

        Returns:
            标量 loss = L_imle + λ_force·L_force（λ 由 policy 内部定）
        """
        if noise is None:
            noise = self.sample_noise(_shape_of(actions), _device_of(actions))

        losses = policy(observation, actions, z_t=z_t, z_goal=z_goal, noise=noise)
        l_imle = losses.get("imle", 0)
        l_force = losses.get("force", 0)
        return _reduce(l_imle) + _reduce(l_force)


# ---- duck typing 小工具（不依赖 torch，对齐 flow_matching.py）----
def _shape_of(x: Any) -> tuple:
    s = getattr(x, "shape", None)
    if s is None:
        raise TypeError(f"schrodinger_bridge: 需要 .shape，得到 {type(x)}")
    return tuple(s)


def _reduce(per_element_loss: Any) -> Any:
    """per-element loss → 标量（对齐 flow_matching._reduce）。"""
    return per_element_loss.mean()


def _device_of(x: Any) -> Any:
    return getattr(x, "device", None)


# ---- 桥数学工具（纯函数，可独立单测）----
def cubic_spline(q0: Any, q1: Any, t: Any) -> tuple:
    """三次样条插值 (q,v,a)。

    q_t = q0 + (q1−q0)·(3t² − 2t³)
    v_t = (q1−q0)·(6t − 6t²)
    a_t = (q1−q0)·(6 − 12t)
    """
    delta = q1 - q0
    q_t = q0 + delta * (3 * t ** 2 - 2 * t ** 3)
    v_t = delta * (6 * t - 6 * t ** 2)
    a_t = delta * (6 - 12 * t)
    return q_t, v_t, a_t


def quartic_envelope(t: Any, sigma_peak: float = DEFAULT_SIGMA_PEAK) -> Any:
    """quartic 噪声包络 σ_t = 16·sigma_peak·(t(1−t))²（双零点）。"""
    return 16 * sigma_peak * (t * (1 - t)).pow(2)
