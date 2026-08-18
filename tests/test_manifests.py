"""Tests for :mod:`vr_modality_bias.data.manifests`.

The provenance fields (``dataset``, ``scene_id``, ``frame_index``) were added
after the MSCOCO manifests were already on disk, so the invariant that matters
most here is that a record written without them still loads.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vr_modality_bias.data.manifests import (
    ImageRecord,
    read_manifest,
    write_manifest,
)

_REPO_MANIFEST = (
    Path(__file__).parent.parent
    / "data" / "processed" / "mscoco_baseline" / "manifest.jsonl"
)

_LEGACY_LINE = (
    '{"image_id": "000000000139", "file_name": "000000000139.jpg", '
    '"width": 640, "height": 426, "source": "mscoco_baseline"}'
)


def test_legacy_record_without_the_provenance_fields_loads(tmp_path: Path):
    path = tmp_path / "manifest.jsonl"
    path.write_text(_LEGACY_LINE + "\n", encoding="utf-8")

    records = read_manifest(path)

    assert len(records) == 1
    assert records[0].image_id == "000000000139"
    assert records[0].source == "mscoco_baseline"
    assert records[0].dataset is None
    assert records[0].scene_id is None
    assert records[0].frame_index is None


@pytest.mark.skipif(
    not _REPO_MANIFEST.is_file(),
    reason="no manifest committed under data/processed",
)
def test_the_manifest_committed_in_the_repo_still_loads():
    records = read_manifest(_REPO_MANIFEST)

    assert records
    assert all(r.dataset is None for r in records)
    assert all(r.scene_id is None for r in records)
    assert all(r.frame_index is None for r in records)


def test_round_trip_with_the_provenance_fields(tmp_path: Path):
    path = tmp_path / "manifest.jsonl"
    records = [
        ImageRecord(
            image_id="adt_seq07_000123",
            file_name="adt_seq07_000123.jpg",
            width=1408,
            height=1408,
            source="aria_digital_twin",
            dataset="adt",
            scene_id="seq07",
            frame_index=123,
        ),
        ImageRecord(
            image_id="odi_0042",
            file_name="odi_0042.jpg",
            width=4096,
            height=2048,
            source="odi_bench",
            dataset="odi_bench",
        ),
    ]

    assert write_manifest(records, path) == 2
    assert read_manifest(path) == records


def test_an_image_id_with_a_forward_slash_is_rejected():
    with pytest.raises(ValueError, match="path separator"):
        ImageRecord(
            image_id="scene_A/frame_7",
            file_name="frame_7.png",
            width=8,
            height=8,
            source="adt",
        )


def test_an_image_id_with_a_backslash_is_rejected():
    with pytest.raises(ValueError, match="path separator"):
        ImageRecord(
            image_id="scene_A\\frame_7",
            file_name="frame_7.png",
            width=8,
            height=8,
            source="adt",
        )


def test_the_rejection_names_the_offending_identifier():
    with pytest.raises(ValueError, match=r"scene_A/frame_7"):
        ImageRecord("scene_A/frame_7", "f.png", 8, 8, "adt")


def test_a_double_underscore_in_the_image_id_is_allowed():
    """compute_metrics.py splits '{image_id}__{condition}' with rsplit('__', 1),
    so the separator is unambiguous however many '__' the id itself holds."""
    record = ImageRecord("scene__A__frame7", "f.png", 8, 8, "adt")

    assert record.image_id == "scene__A__frame7"
    assert record.image_id.rsplit("__", 1) == ["scene__A", "frame7"]


def test_a_file_name_may_still_contain_a_separator(tmp_path: Path):
    """Only the identity is constrained; the path under images_dir is free."""
    record = ImageRecord("seq07_000123", "seq07/frame_000123.png", 8, 8, "adt")

    assert record.file_name == "seq07/frame_000123.png"


def test_malformed_line_names_the_file_and_the_line(tmp_path: Path):
    path = tmp_path / "manifest.jsonl"
    path.write_text(_LEGACY_LINE + "\n" + "{ not json\n", encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        read_manifest(path)

    message = str(excinfo.value)
    assert str(path) in message
    assert "line 2" in message
