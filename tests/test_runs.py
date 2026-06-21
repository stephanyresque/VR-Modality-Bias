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


# ================================================================
# Block-4 organized layout: <output_root>/<area>/<model>/<length>/...
# ================================================================


def test_length_from_prompt_key_recognises_three_lengths():
    from vr_modality_bias.utils.runs import length_from_prompt_key

    assert length_from_prompt_key("caption_short") == "short"
    assert length_from_prompt_key("caption_medium") == "medium"
    assert length_from_prompt_key("caption_long") == "long"


def test_length_from_prompt_key_rejects_unknown():
    from vr_modality_bias.utils.runs import length_from_prompt_key
    import pytest

    with pytest.raises(ValueError):
        length_from_prompt_key("caption_xxx")
    with pytest.raises(ValueError):
        length_from_prompt_key("describe")


def test_area_root_builds_nested_path(tmp_path):
    from vr_modality_bias.utils.runs import area_root

    root = area_root(
        tmp_path / "results",
        area="diagnostico",
        model_key="llava-1.5-7b",
        length="long",
    )
    assert root == tmp_path / "results" / "diagnostico" / "llava-1.5-7b" / "long"


def test_area_root_rejects_bad_area_or_length(tmp_path):
    from vr_modality_bias.utils.runs import area_root
    import pytest

    with pytest.raises(ValueError):
        area_root(tmp_path, area="wrong", model_key="m", length="short")
    with pytest.raises(ValueError):
        area_root(tmp_path, area="diagnostico", model_key="m", length="xl")


def test_make_run_dir_under_area_root_round_trip(tmp_path):
    """The organized layout composes cleanly: area_root + make_run_dir + current_run_dir."""
    from vr_modality_bias.utils.runs import area_root, current_run_dir, make_run_dir

    root = area_root(
        tmp_path / "results",
        area="diagnostico",
        model_key="llava-1.5-7b",
        length="short",
    )
    run_dir = make_run_dir(root, "baseline_llava")
    # Run dir lives at <area_root>/<run-name>_<ts>/
    assert run_dir.parent == root
    assert run_dir.name.startswith("baseline_llava_")

    # The pointer file is co-located with the run dir, NOT at output_root.
    # current_run_dir takes the same area_root and finds it via the pointer.
    resolved = current_run_dir(root, "baseline_llava")
    assert resolved.resolve() == run_dir.resolve()
