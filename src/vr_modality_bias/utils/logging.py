"""Centralised logging setup"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

__all__ = ["configure_logging", "get_logger"]

_DEFAULT_FORMAT = "%(asctime)s %(levelname)s %(name)s :: %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%dT%H:%M:%S"


def get_logger(name: str | None = None) -> logging.Logger:
    """Return ``logging.getLogger(name or "vr_modality_bias")``."""
    return logging.getLogger(name or "vr_modality_bias")


def configure_logging(
    level: int | str = logging.INFO,
    log_file: Path | None = None,
    *,
    fmt: str = _DEFAULT_FORMAT,
    datefmt: str = _DEFAULT_DATEFMT,
) -> None:
    
    root = logging.getLogger()
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    root.setLevel(level)

    for h in list(root.handlers):
        root.removeHandler(h)

    formatter = logging.Formatter(fmt, datefmt=datefmt)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)
    root.addHandler(stderr_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
