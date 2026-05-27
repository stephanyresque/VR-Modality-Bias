from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from vr_modality_bias.utils.runs import (
    current_run_dir,
    make_run_dir,
    pointer_path,
)


def test_make_run_dir_creates_timestamped_dir_and_pointer(tmp_path: Path):
    ts = datetime(2026, 5, 27, 14, 30, 0)
    run_dir = make_run_dir(tmp_path, "myrun", timestamp=ts)
    assert run_dir.is_dir()
    assert run_dir.name == "myrun_2026-05-27_143000"

    pointer = pointer_path(tmp_path, "myrun")
    assert pointer.is_file()
    assert Path(pointer.read_text(encoding="utf-8").strip()) == run_dir.resolve()


def test_make_run_dir_updates_pointer_on_subsequent_calls(tmp_path: Path):
    first = make_run_dir(tmp_path, "myrun", timestamp=datetime(2026, 1, 1, 0, 0, 0))
    second = make_run_dir(tmp_path, "myrun", timestamp=datetime(2026, 6, 1, 0, 0, 0))
    assert first != second
    assert current_run_dir(tmp_path, "myrun") == second.resolve()


def test_make_run_dir_raises_on_duplicate_timestamp(tmp_path: Path):
    ts = datetime(2026, 5, 27, 14, 30, 0)
    make_run_dir(tmp_path, "myrun", timestamp=ts)
    with pytest.raises(FileExistsError):
        make_run_dir(tmp_path, "myrun", timestamp=ts)


def test_current_run_dir_raises_when_no_pointer(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        current_run_dir(tmp_path, "myrun")


def test_current_run_dir_raises_when_target_was_deleted(tmp_path: Path):
    run_dir = make_run_dir(tmp_path, "myrun", timestamp=datetime(2026, 5, 27))
    import shutil

    shutil.rmtree(run_dir)
    with pytest.raises(FileNotFoundError):
        current_run_dir(tmp_path, "myrun")


def test_pointer_is_isolated_per_run_name(tmp_path: Path):
    a = make_run_dir(tmp_path, "runA", timestamp=datetime(2026, 5, 27, 10))
    b = make_run_dir(tmp_path, "runB", timestamp=datetime(2026, 5, 27, 11))
    assert current_run_dir(tmp_path, "runA") == a.resolve()
    assert current_run_dir(tmp_path, "runB") == b.resolve()
