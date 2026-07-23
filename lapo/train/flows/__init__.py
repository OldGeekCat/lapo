"""可复用技术能力层（flows/）。

放与具体 policy 无关、可被多个 policy/strategy 组合复用的技术模块：
- schrodinger_bridge: 薛定谔桥（SB）训练 loss / 采样编排（LAPo / SB-VLA 共用）

设计原则：每个模块用 duck typing / 延迟 import，使核心逻辑在无 torch 的环境
也能单测；只有真正调用 torch/numpy API 的入口才延迟 import 重依赖。
"""
from lapo.train.flows.schrodinger_bridge import SchrodingerBridgeLoss

__all__ = ["SchrodingerBridgeLoss"]
