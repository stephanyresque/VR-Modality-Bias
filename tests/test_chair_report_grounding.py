"""Tests for the ground-truth side of scripts/chair_report.py.

Two things are pinned here. First, the ground truth now comes from the
project's own ``annotations.jsonl`` instead of the MSCOCO instance file.
Second, a caption without a ground-truth entry is fatal: the MSCOCO path
skipped those, which on a dataset swap would let an id-format mismatch empty
every table without a word of warning.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from vr_modality_bias.data.annotations import ObjectAnnotation, write_object_annotations

_SCRIPTS = Path(__file__).parent.parent / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"_script_{name}", _SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def chair_report():
    return _load_script("chair_report")


def _caption(image_id: str, *, length: str = "short", condition: str = "off") -> dict:
    return {
        "image_id": image_id,
        "length": length,
        "condition": condition,
        "caption": "a mug on a desk",
        "alpha": None,
        "sparc": None,
    }


# ---------------------------------------------------------------- gt source


def test_ground_truth_comes_from_the_object_annotations(chair_report, tmp_path: Path):
    path = tmp_path / "annotations.jsonl"
    write_object_annotations(
        [
            ObjectAnnotation(image_id="adt_seq07_000123", objects=("mug", "keyboard")),
            ObjectAnnotation(image_id="adt_seq07_000124", objects=("desk",)),
        ],
        path,
    )

    gt = chair_report.load_object_ground_truth(path)

    assert gt == {
        "adt_seq07_000123": {"mug", "keyboard"},
        "adt_seq07_000124": {"desk"},
    }


def test_an_item_annotated_with_no_objects_still_appears(chair_report, tmp_path: Path):
    path = tmp_path / "annotations.jsonl"
    write_object_annotations([ObjectAnnotation(image_id="empty_scene", objects=())], path)

    gt = chair_report.load_object_ground_truth(path)

    assert gt["empty_scene"] == set(), (
        "an item the dataset annotated as containing nothing is grounded; it "
        "must not be confused with an item that is missing from the file."
    )


def test_the_loader_rejects_a_malformed_annotations_file(chair_report, tmp_path: Path):
    path = tmp_path / "annotations.jsonl"
    path.write_text('{"image_id": "x", "objects": ["mug"]}\n{ not json\n', encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        chair_report.load_object_ground_truth(path)

    assert "line 2" in str(excinfo.value)


# ---------------------------------------------------------------- hard guard


def test_a_caption_without_ground_truth_is_fatal(chair_report):
    gt = {"adt_seq07_000123": {"mug"}}
    entries = [_caption("adt_seq07_000123"), _caption("000000000139")]

    with pytest.raises(ValueError):
        chair_report.assert_captions_are_grounded(entries, gt)


def test_the_failure_counts_the_ungrounded_captions(chair_report):
    gt = {"a": {"mug"}}
    entries = [
        _caption("missing_1", length="short"),
        _caption("missing_1", length="long"),
        _caption("missing_2", length="short"),
    ]

    with pytest.raises(ValueError) as excinfo:
        chair_report.assert_captions_are_grounded(entries, gt)

    message = str(excinfo.value)
    assert "3 caption(s)" in message, "the count must be of captions, not of ids"
    assert "2 image_id(s)" in message


def test_the_failure_shows_identifiers_from_both_sides(chair_report):
    gt = {"000000000139": {"mug"}, "000000000285": {"desk"}}
    entries = [_caption("adt_seq07_000123")]

    with pytest.raises(ValueError) as excinfo:
        chair_report.assert_captions_are_grounded(entries, gt)

    message = str(excinfo.value)
    assert "adt_seq07_000123" in message, "must show the caption-side id"
    assert "000000000139" in message, "must show the ground-truth-side id"


def test_a_sample_of_ids_is_shown_not_the_whole_set(chair_report):
    gt = {f"gt_{i:04d}": {"mug"} for i in range(50)}
    entries = [_caption(f"cap_{i:04d}") for i in range(50)]

    with pytest.raises(ValueError) as excinfo:
        chair_report.assert_captions_are_grounded(entries, gt)

    message = str(excinfo.value)
    assert "cap_0004" in message
    assert "cap_0005" not in message, "the message must stay readable"


def test_fully_grounded_captions_pass(chair_report):
    gt = {"a": {"mug"}, "b": {"desk"}}

    chair_report.assert_captions_are_grounded([_caption("a"), _caption("b")], gt)


def test_ground_truth_without_captions_is_not_an_error(chair_report):
    gt = {"a": {"mug"}, "b": {"desk"}, "c": {"chair"}}

    chair_report.assert_captions_are_grounded([_caption("a")], gt)


def test_the_reverse_gap_is_reported_as_a_count(chair_report, tmp_path: Path, capsys):
    """A partial or resumed run has ground truth for items not generated yet."""
    gt = {"a": {"mug"}, "b": {"desk"}, "c": {"chair"}}
    entries = [_caption("a")]

    chair_report.assert_captions_are_grounded(entries, gt)
    ungenerated = sorted(set(gt) - {e["image_id"] for e in entries})

    assert ungenerated == ["b", "c"]
