"""Run-directory management"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

__all__ = ["make_run_dir", "current_run_dir", "pointer_path"]

_TIMESTAMP_FMT = "%Y-%m-%d_%H%M%S"


def pointer_path(output_root: Path | str, run_name: str) -> Path:
    """Return the path to the ``<run_name>_LATEST.txt`` pointer file."""
    return Path(output_root) / f"{run_name}_LATEST.txt"


def make_run_dir(
    output_root: Path | str,
    run_name: str,
    *,
    timestamp: datetime | None = None,
) -> Path:
    
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    ts = (timestamp or datetime.now()).strftime(_TIMESTAMP_FMT)
    run_dir = output_root / f"{run_name}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=False)
    pointer_path(output_root, run_name).write_text(
        str(run_dir.resolve()) + "\n", encoding="utf-8"
    )
    return run_dir


def current_run_dir(output_root: Path | str, run_name: str) -> Path:
    
    output_root = Path(output_root)
    pointer = pointer_path(output_root, run_name)
    if not pointer.is_file():
        raise FileNotFoundError(
            f"No active run for {run_name!r} under {output_root}. "
            "Run scripts/03_generate_refs.py first."
        )
    run_dir = Path(pointer.read_text(encoding="utf-8").strip())
    if not run_dir.is_dir():
        raise FileNotFoundError(
            f"Pointer for {run_name!r} references missing dir: {run_dir}. "
            "Edit the LATEST file or rerun scripts/03_generate_refs.py."
        )
    return run_dir
