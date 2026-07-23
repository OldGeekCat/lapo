"""Unified storage root for the ``lapo`` toolset.

All persistent artifacts (datasets, model weights, training outputs, logs,
caches) live under a single root resolved once here:

    $LAPO_HOME  →  $XDG_DATA_HOME/lapo-home  →  ~/lapo-home  (fallbacks)

The fallback honors ``$XDG_DATA_HOME`` (or ``$HOME``) so it always lands in
the current user's writable data dir — never under ``/root`` (which crashes
non-root users; see HANDOFF §9.7).

Layout::

    $LAPO_HOME/
        datasets/      LeRobot datasets (downloaded / imported)
        models/        managed model tree (downloaded/, custom/)
        outputs/       training outputs, video conversions, logs
        cache/         misc caches
            hf_hub/    HuggingFace Hub cache (HF_HOME)
        registry/      central model index.yaml

Every helper auto-creates its directory on first call, so callers can treat
the returned path as guaranteed to exist. CLI options like ``--output-dir``
still override these defaults locally.
"""
from __future__ import annotations

import os
from pathlib import Path


def _default_home() -> Path:
    """Resolve a writable fallback root when ``$LAPO_HOME`` is unset.

    Prefers ``$XDG_DATA_HOME/lapo-home``, then ``~/lapo-home``. Both are per-user
    and writable, avoiding the old hardcoded ``/root/gpufree-data/lr`` which
    crashed any non-root user (HANDOFF §9.7).
    """
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg) if xdg else Path.home()
    return base / "lapo-home"


# Kept as a Path for backwards-compat (some callers/tests import it). Read it
# via _default_home() at call time so env changes (XDG_DATA_HOME) take effect.
_DEFAULT_HOME = _default_home()

__all__ = [
    "lapo_home",
    "datasets_dir",
    "models_dir",
    "outputs_dir",
    "cache_dir",
    "hf_cache_dir",
    "registry_dir",
    "resolve_output_path",
]


def lapo_home() -> Path:
    """Resolve the unified storage root.

    Order: ``$LAPO_HOME`` env var → ``$XDG_DATA_HOME/lapo-home`` → ``~/lapo-home``.
    The directory and every standard subdirectory are created on first access.
    The fallback is always writable by the current user (never under ``/root``).
    """
    env = os.environ.get("LAPO_HOME")
    home = Path(env).expanduser() if env else _default_home()
    home.mkdir(parents=True, exist_ok=True)
    return home


def _subdir(name: str) -> Path:
    p = lapo_home() / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def datasets_dir() -> Path:
    """Datasets root: ``$LAPO_HOME/datasets``."""
    return _subdir("datasets")


def models_dir() -> Path:
    """Managed model tree root: ``$LAPO_HOME/models`` (downloaded/, custom/)."""
    root = _subdir("models")
    for sub in ("downloaded", "custom"):
        (root / sub).mkdir(exist_ok=True)
    return root


def outputs_dir() -> Path:
    """Outputs root: ``$LAPO_HOME/outputs`` (training runs, conversions, logs)."""
    return _subdir("outputs")


def cache_dir() -> Path:
    """Misc caches root: ``$LAPO_HOME/cache``."""
    return _subdir("cache")


def hf_cache_dir() -> Path:
    """HuggingFace Hub cache: ``$LAPO_HOME/cache/hf_hub``."""
    p = cache_dir() / "hf_hub"
    p.mkdir(parents=True, exist_ok=True)
    return p


def registry_dir() -> Path:
    """Central registry root: ``$LAPO_HOME/registry``."""
    return _subdir("registry")


def resolve_output_path(path: str | Path) -> Path:
    """Resolve a possibly-relative output path against ``outputs_dir()``.

    Absolute paths and ``~``-prefixed paths are returned as-is (expanded).
    Relative paths are anchored under ``$LAPO_HOME/outputs`` so a config value
    like ``outputs/openarm_xvla`` lands inside the storage root rather than the
    process CWD.
    """
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return outputs_dir() / p
