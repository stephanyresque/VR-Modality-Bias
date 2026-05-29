"""Configuration loading and snapshotting"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import yaml

__all__ = ["load_config", "snapshot_config"]


def load_config(path: str | Path) -> dict[str, Any]:
    """Parse the YAML file at ``path`` and return the top-level mapping."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {path} must be a YAML mapping at the top level.")
    return cfg


def snapshot_config(src: Path, dest_dir: Path, *, name: str = "config.yaml") -> Path:
    """Copy ``src`` into ``dest_dir/name``. Returns the destination path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / name
    shutil.copyfile(src, dest)
    return dest
