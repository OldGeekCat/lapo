"""pytest 共享配置：把仓库根加入 sys.path，使 lapo 包可被 import。"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
