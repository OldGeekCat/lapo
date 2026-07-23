"""Registry YAML 持久化存储。

读写 $LAPO_HOME/registry/ 下的 policies.yaml / strategies.yaml / traits.yaml。
纯 I/O，无业务逻辑，便于 Registry 类测。
"""
from __future__ import annotations

from pathlib import Path

import yaml


class RegistryStore:
    """读写 $LAPO_HOME/registry/ 下的三类 YAML。"""

    def __init__(self, root: str | Path | None = None):
        if root is None:
            from lapo.paths import registry_dir
            root = registry_dir()
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        return self.root / f"{name}.yaml"

    def _load(self, name: str) -> dict:
        p = self._path(name)
        if not p.exists():
            return {}
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}

    def _save(self, name: str, data: dict) -> None:
        self._path(name).write_text(
            yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8",
        )

    # ---- policies ----
    def load_policies(self) -> dict:
        return self._load("policies")

    def save_policies(self, data: dict) -> None:
        self._save("policies", data)

    # ---- strategies ----
    def load_strategies(self) -> dict:
        return self._load("strategies")

    def save_strategies(self, data: dict) -> None:
        self._save("strategies", data)

    # ---- traits ----
    def load_traits(self) -> list[str]:
        return self._load("traits").get("traits", [])

    def save_traits(self, traits: list[str]) -> None:
        self._save("traits", {"traits": traits})
