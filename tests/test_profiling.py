from __future__ import annotations

import time
from pathlib import Path

import pytest

from vr_modality_bias.utils.profiling import (
    Timer,
    cuda_peak_bytes,
    dir_size_bytes,
    format_bytes,
    format_seconds,
    reset_cuda_peak,
    summarize_seconds,
)


def test_timer_records_elapsed_seconds_positive():
    with Timer() as t:
        time.sleep(0.02)
    assert t.seconds >= 0.015
    assert t.seconds < 1.0  # generous upper bound for CI/dev


def test_timer_is_reusable_across_contexts():
    t = Timer()
    with t:
        time.sleep(0.01)
    first = t.seconds
    with t:
        time.sleep(0.02)
    second = t.seconds
    assert first > 0
    assert second > first - 0.005


def test_format_bytes_common_magnitudes():
    assert format_bytes(0) == "0.00 B"
    assert format_bytes(512) == "512.00 B"
    assert format_bytes(1024) == "1.00 KB"
    assert format_bytes(1024 * 1024) == "1.00 MB"
    assert format_bytes(1024 ** 3) == "1.00 GB"
    assert format_bytes(int(1.5 * 1024 ** 4)) == "1.50 TB"


def test_format_seconds_common_magnitudes():
    assert format_seconds(0) == "0.00 s"
    assert format_seconds(1.234) == "1.23 s"
    assert format_seconds(59.99) == "59.99 s"
    assert format_seconds(60) == "1m 00.0s"
    assert format_seconds(75) == "1m 15.0s"
    assert format_seconds(3600) == "1h 0m"
    assert format_seconds(7325) == "2h 2m"


def test_format_seconds_handles_none_and_negatives():
    assert format_seconds(None) == "n/a"
    assert format_seconds(-1.5) == "-1.50 s"


def test_dir_size_bytes_sums_recursively(tmp_path: Path):
    (tmp_path / "a.bin").write_bytes(b"x" * 1024)
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.bin").write_bytes(b"y" * 512)
    (sub / "c.bin").write_bytes(b"z" * 256)
    assert dir_size_bytes(tmp_path) == 1024 + 512 + 256


def test_dir_size_bytes_returns_zero_for_missing_path(tmp_path: Path):
    assert dir_size_bytes(tmp_path / "does_not_exist") == 0


def test_dir_size_bytes_handles_single_file(tmp_path: Path):
    path = tmp_path / "lone.bin"
    path.write_bytes(b"x" * 333)
    assert dir_size_bytes(path) == 333


def test_cuda_helpers_are_safe_without_cuda():
    # On a CPU-only machine these must return cleanly.
    reset_cuda_peak()
    assert cuda_peak_bytes() == 0


def test_summarize_seconds_on_known_values():
    stats = summarize_seconds([1.0, 2.0, 3.0, 4.0, 5.0])
    assert stats["n"] == 5
    assert stats["total"] == 15.0
    assert stats["median"] == 3.0
    assert stats["mean"] == 3.0
    assert stats["min"] == 1.0
    assert stats["max"] == 5.0


def test_summarize_seconds_empty_returns_none_central_stats():
    stats = summarize_seconds([])
    assert stats["n"] == 0
    assert stats["total"] == 0.0
    assert stats["median"] is None
    assert stats["mean"] is None
    assert stats["min"] is None
    assert stats["max"] is None


@pytest.mark.parametrize("value", [1, 1.5, 100.25])
def test_format_bytes_accepts_int_and_float(value: float):
    out = format_bytes(value)
    assert isinstance(out, str)
    assert " " in out  # "X.XX <unit>"
