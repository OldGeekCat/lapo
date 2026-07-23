"""环境检测：collect_env 收集版本信息，check_env 对照兼容矩阵。

训练时 collect_env() → env_info.json（由 ArtifactWriter 写）；
CLI lr env 打印——同一数据源两出口。
"""
from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class EnvInfo:
    """环境快照。to_dict() 供 ArtifactWriter.write_env_info。"""
    python: str
    torch: Optional[str]
    lerobot: Optional[str]
    transformers: Optional[str]
    gpu_name: Optional[str]
    cuda: Optional[str]
    cudnn: Optional[str]
    git_commit: Optional[str]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CheckResult:
    """单项检查结果。status: ok | warn | fail。"""
    name: str
    status: str
    message: str


# 已验证的版本组合（与 TROUBLESHOOTING 对齐）。
COMPAT_MATRIX = [
    {
        "lerobot": "0.4.4",
        "transformers_min": "4.57.1",
        "transformers_max": "5.0.0",  # 上界不含
        "torch_min": "2.6.0",
        "torch_max": "2.11.0",         # 上界不含
        "status": "verified",
    },
]


def _pkg_version(name: str) -> Optional[str]:
    """取已安装包版本，未装返回 None。"""
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def _git_commit() -> Optional[str]:
    """当前仓库的 short commit，非 git 仓库返回 None。"""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, timeout=3,
        ).decode().strip()
        return out or None
    except Exception:
        return None


def _gpu_info() -> tuple[Optional[str], Optional[str], Optional[str]]:
    """(gpu_name, cuda_version, cudnn_version)。无 torch 或无 GPU 返回 (None,*,None)。"""
    try:
        import torch
        if torch.cuda.is_available():
            return (
                torch.cuda.get_device_name(0),
                torch.version.cuda,
                str(torch.backends.cudnn.version()),
            )
        return (None, torch.version.cuda, None)
    except Exception:
        return None, None, None


def collect_env() -> EnvInfo:
    """收集当前环境信息。torch/lerobot 未装时对应字段为 None。"""
    gpu, cuda, cudnn = _gpu_info()
    return EnvInfo(
        python=platform.python_version(),
        torch=_pkg_version("torch"),
        lerobot=_pkg_version("lerobot"),
        transformers=_pkg_version("transformers"),
        gpu_name=gpu, cuda=cuda, cudnn=cudnn,
        git_commit=_git_commit(),
    )


def _parse_ver(v: Optional[str]) -> tuple:
    """'2.10.0+cu128' → (2, 10, 0)。None/无法解析 → ()。"""
    if not v:
        return ()
    parts = []
    for p in v.split("+")[0].split("."):
        try:
            parts.append(int(p))
        except ValueError:
            break
    return tuple(parts)


def _cmp(v: tuple, lo: Optional[str], hi: Optional[str]) -> str:
    """v 与 [lo, hi) 比较，返回 'in_range' | 'below' | 'above' | 'unknown'。"""
    if not v:
        return "unknown"
    if lo and v < _parse_ver(lo):
        return "below"
    if hi and v >= _parse_ver(hi):
        return "above"
    return "in_range"


def check_env(info: Optional[EnvInfo] = None) -> list[CheckResult]:
    """对照兼容矩阵检查，返回每项 ok/warn/fail + 建议。

    检查项:
    - transformers 与 lerobot 0.4.x 的兼容性（必须 <5.0.0，TROUBLESHOOTING #4）
    - lerobot 版本是否在已验证矩阵中
    - torch 版本是否在已验证区间
    """
    info = info or collect_env()
    results: list[CheckResult] = []

    tv = _parse_ver(info.transformers)
    lv = _parse_ver(info.lerobot)
    torchv = _parse_ver(info.torch)

    # transformers 5.x 与 lerobot 0.4.x 不兼容
    if lv and lv < (0, 5):
        if tv and tv >= (5, 0):
            results.append(CheckResult(
                "transformers", "fail",
                f"transformers {info.transformers} 与 lerobot {info.lerobot} 不兼容"
                f"（需 <5.0.0）。详见 TROUBLESHOOTING #4。",
            ))
        elif tv:
            results.append(CheckResult(
                "transformers", "ok",
                f"transformers {info.transformers} 满足 <5.0.0 约束。",
            ))

    # lerobot 是否在矩阵
    matrix_lerobot_versions = {_parse_ver(m["lerobot"]) for m in COMPAT_MATRIX}
    if lv and lv in matrix_lerobot_versions:
        results.append(CheckResult(
            "lerobot", "ok", f"lerobot {info.lerobot} 在已验证矩阵中。",
        ))
    elif lv:
        results.append(CheckResult(
            "lerobot", "warn",
            f"lerobot {info.lerobot} 未在已验证矩阵，可能遇坑。",
        ))

    # torch 区间
    if torchv:
        m = COMPAT_MATRIX[0]
        pos = _cmp(torchv, m["torch_min"], m["torch_max"])
        if pos == "in_range":
            results.append(CheckResult(
                "torch", "ok",
                f"torch {info.torch} 在已验证区间 [{m['torch_min']}, {m['torch_max']})。",
            ))
        elif pos == "above":
            results.append(CheckResult(
                "torch", "warn",
                f"torch {info.torch} 高于已验证上界 {m['torch_max']}，可能有兼容问题。",
            ))
        elif pos == "below":
            results.append(CheckResult(
                "torch", "fail",
                f"torch {info.torch} 低于已验证下界 {m['torch_min']}。",
            ))

    if not results:
        results.append(CheckResult(
            "env", "warn",
            "无法检测 lerobot/transformers，请确认已安装。",
        ))
    return results
