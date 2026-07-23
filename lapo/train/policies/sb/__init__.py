"""SB-VLA: Schrödinger Bridge VLA policy family.

实现 docs/next-gen-architecture-v2.md 的短程单 chunk 闭环：
  encoder(frame) -> z            (LeWM 式 JEPA + SIGReg 守护)
  g(z_t, h)      -> z_goal       (act-free 目标提议)
  f(z_t, chunk)  -> z_after      (跳步世界模型，训练时算 L_reach)
  SB([z_t, z_goal]) -> chunk     (薛定谔桥 + IMLE，生成多模态 action)

组件来源（逐字移植后裁剪）：
  - SIGReg / Encoder / g / f  : LeWM (lucas-maes/le-wm) module.py + jepa.py
  - SB 桥 / IMLE / Euler 积分 : Chronos (yulinzhouZYL/Chronos)
                                mamba_policy_par_3D_IMLE.py（去掉 Mamba/DINO/3D RoPE）

关键设计选择（已与用户确认）：
  1. 去掉 EMA target encoder —— LeWM 源码证实靠 SIGReg 单独防坍塌
  2. SB 桥照搬 Chronos 跑通算法 —— 三次样条 + 简单 Euler 积分（非 leapfrog）
  3. 不加 Mamba 记忆主干 —— 单帧 z_t + 任务意图 h，对齐 MVP
"""
from __future__ import annotations

from lapo.train.policies.sb.config import SBVLAConfig

__all__ = ["SBVLAConfig"]
