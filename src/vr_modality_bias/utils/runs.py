"""Run-directory management."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

__all__ = [
    "make_run_dir",
    "current_run_dir",
    "pointer_path",
    "area_root",
    "length_from_prompt_key",
]

_TIMESTAMP_FMT = "%Y-%m-%d_%H%M%S"


_LENGTH_NAMES = ("short", "medium", "long")


def length_from_prompt_key(prompt_key: str) -> str:
    """Return ``"short" | "medium" | "long"`` for a caption prompt_key.

    Active prompt keys are ``caption_short``, ``caption_medium`` and
    ``caption_long``. The length name becomes part of the organized
    output path: ``results/<area>/<model>/<length>/<run>/``.
    """
    for length in _LENGTH_NAMES:
        if prompt_key.endswith(f"_{length}"):
            return length
    raise ValueError(
        f"prompt_key={prompt_key!r} does not end in one of {_LENGTH_NAMES}; "
        "cannot derive a length name for the organized results path."
    )


def area_root(
    output_root: Path | str,
    *,
    area: str,
    model_key: str,
    length: str,
) -> Path:
    """Return ``<output_root>/<area>/<model_key>/<length>/``.

    The path returned is then handed to :func:`make_run_dir` /
    :func:`current_run_dir` as their ``output_root`` — those helpers
    don't know about the nesting, they just append the timestamped run
    dir + pointer file at whatever prefix they get.

    Args:
        area: ``"diagnostico"`` or ``"avaliacao"`` (the two top-level
            buckets — see ``results/README.md``).
        model_key: the registry key (e.g. ``"llava-1.5-7b"``).
        length: ``"short" | "medium" | "long"`` — usually obtained from
            :func:`length_from_prompt_key`.
    """
    if area not in ("diagnostico", "avaliacao"):
        raise ValueError(
            f"area must be 'diagnostico' or 'avaliacao', got {area!r}."
        )
    if length not in _LENGTH_NAMES:
        raise ValueError(
            f"length must be one of {_LENGTH_NAMES}, got {length!r}."
        )
    return Path(output_root) / area / model_key / length


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
