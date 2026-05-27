"""Light-weight profiling primitives: timing, CUDA peak memory, disk size, formatting.

Used by ``scripts/04_collect_hidden_states.py`` to measure per-pair cost and
by ``scripts/06_summarize.py`` to consolidate the run-level diagnostics
required by EXPERIMENT.md §7 Phase 6 (tempo total, tempo por par mediano,
VRAM peak, tamanho em disco por par, tamanho total).
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path

__all__ = [
    "Timer",
    "cuda_peak_bytes",
    "dir_size_bytes",
    "format_bytes",
    "format_seconds",
    "reset_cuda_peak",
    "summarize_seconds",
]


class Timer:
    """Context manager that exposes elapsed wall-clock seconds in ``self.seconds``."""

    def __init__(self) -> None:
        self.seconds: float = 0.0
        self._start: float | None = None

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._start is not None:
            self.seconds = time.perf_counter() - self._start
            self._start = None


def reset_cuda_peak() -> None:
    """Reset ``torch.cuda.max_memory_allocated()``; no-op without CUDA / torch."""
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def cuda_peak_bytes() -> int:
    """Return ``torch.cuda.max_memory_allocated()`` or ``0`` if CUDA is unavailable."""
    try:
        import torch
    except ImportError:
        return 0
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.max_memory_allocated())


def dir_size_bytes(path: Path) -> int:
    """Recursive total byte size of every file under ``path``.

    Returns ``0`` for missing paths and silently ignores files whose
    ``stat()`` fails (e.g. a permission error mid-walk).
    """
    path = Path(path)
    if not path.exists():
        return 0
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                continue
    return total


_BYTE_UNITS: tuple[str, ...] = ("B", "KB", "MB", "GB", "TB", "PB")


def format_bytes(n: int | float) -> str:
    """Human-friendly byte size, e.g. ``"1.34 GB"``."""
    value = float(n)
    for unit in _BYTE_UNITS[:-1]:
        if abs(value) < 1024.0:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} {_BYTE_UNITS[-1]}"


def format_seconds(n: float | None) -> str:
    """Human-friendly seconds: ``"1.23 s"``, ``"45.00 s"``, ``"1m 15.0s"``, ``"2h 5m"``.

    Accepts ``None`` and returns ``"n/a"`` — convenient for stats over empty
    collections.
    """
    if n is None:
        return "n/a"
    if n < 0:
        return "-" + format_seconds(-n)
    if n < 60:
        return f"{n:.2f} s"
    minutes, seconds = divmod(n, 60.0)
    if minutes < 60:
        return f"{int(minutes)}m {seconds:04.1f}s"
    hours, minutes = divmod(minutes, 60.0)
    return f"{int(hours)}h {int(minutes)}m"


def summarize_seconds(values: Iterable[float]) -> dict[str, float | int | None]:
    """Aggregate timings into ``{n, total, median, mean, min, max}``.

    Returns ``None`` for the central tendency stats when the input is empty.
    """
    import statistics

    arr = [float(v) for v in values]
    if not arr:
        return {
            "n": 0,
            "total": 0.0,
            "median": None,
            "mean": None,
            "min": None,
            "max": None,
        }
    return {
        "n": len(arr),
        "total": float(sum(arr)),
        "median": float(statistics.median(arr)),
        "mean": float(statistics.fmean(arr)),
        "min": float(min(arr)),
        "max": float(max(arr)),
    }
