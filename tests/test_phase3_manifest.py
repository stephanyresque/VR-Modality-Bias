"""Tests for the image-discovery side of scripts/phase3_generate.py.

The script used to glob ``*.jpg`` and take the file stem as the item id. It now
reads the manifest, so identity and file name are independent and the
extension is whatever the dataset uses. Generation itself needs a GPU and is
validated on the DGX; only the resolution step is exercised here.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from vr_modality_bias.data.manifests import ImageRecord, write_manifest

_SCRIPTS = Path(__file__).parent.parent / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"_script_{name}", _SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def phase3():
    return _load_script("phase3_generate")


def _stage(tmp_path: Path, records: list[ImageRecord], *, on_disk: list[str] | None = None) -> dict:
    """Write a manifest plus the image files, and return a matching cfg dict."""
    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = tmp_path / "manifest.jsonl"
    write_manifest(records, manifest_path)

    staged = [r.file_name for r in records] if on_disk is None else on_disk
    for file_name in staged:
        path = images_dir / file_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not a real image")

    return {
        "dataset": {
            "manifest_path": str(manifest_path),
            "images_dir": str(images_dir),
        }
    }


def _record(image_id: str, file_name: str) -> ImageRecord:
    return ImageRecord(
        image_id=image_id,
        file_name=file_name,
        width=8,
        height=8,
        source="test",
    )


# ---------------------------------------------------------------- reading


def test_items_come_from_the_manifest(phase3, tmp_path: Path):
    cfg = _stage(tmp_path, [
        _record("adt_seq07_000123", "seq07/frame_000123.png"),
        _record("odi_0042", "odi_0042.jpeg"),
    ])

    items = phase3.resolve_manifest_items(cfg, image_ids=None, limit=10)

    assert items == [
        ("adt_seq07_000123", tmp_path / "images" / "seq07" / "frame_000123.png"),
        ("odi_0042", tmp_path / "images" / "odi_0042.jpeg"),
    ]


def test_the_id_is_the_manifest_key_not_the_file_stem(phase3, tmp_path: Path):
    cfg = _stage(tmp_path, [_record("scene_A__frame_7", "0000123.png")])

    (image_id, image_path) = phase3.resolve_manifest_items(cfg, image_ids=None, limit=1)[0]

    assert image_id == "scene_A__frame_7"
    assert image_path.stem == "0000123", (
        "the stem and the id must be free to disagree; that is the whole "
        "point of reading the manifest."
    )


def test_manifest_order_is_preserved(phase3, tmp_path: Path):
    cfg = _stage(tmp_path, [
        _record("z_last", "z.png"),
        _record("a_first", "a.png"),
    ])

    items = phase3.resolve_manifest_items(cfg, image_ids=None, limit=10)

    assert [i for i, _ in items] == ["z_last", "a_first"], (
        "the old glob sorted lexicographically; the manifest's own order is "
        "what defines the evaluation set now."
    )


# ---------------------------------------------------------------- limit


def test_the_limit_takes_the_first_items(phase3, tmp_path: Path):
    cfg = _stage(tmp_path, [_record(f"item_{i}", f"{i}.png") for i in range(5)])

    items = phase3.resolve_manifest_items(cfg, image_ids=None, limit=2)

    assert [i for i, _ in items] == ["item_0", "item_1"]


def test_a_limit_beyond_the_manifest_returns_everything(phase3, tmp_path: Path):
    cfg = _stage(tmp_path, [_record("a", "a.png"), _record("b", "b.png")])

    items = phase3.resolve_manifest_items(cfg, image_ids=None, limit=99)

    assert len(items) == 2


# ---------------------------------------------------------------- selection


def test_explicit_image_ids_select_and_order_the_items(phase3, tmp_path: Path):
    cfg = _stage(tmp_path, [_record(f"item_{i}", f"{i}.png") for i in range(4)])

    items = phase3.resolve_manifest_items(
        cfg, image_ids=["item_3", "item_0"], limit=1,
    )

    assert [i for i, _ in items] == ["item_3", "item_0"], (
        "explicit ids override the limit and keep the order given"
    )


def test_an_unknown_image_id_is_rejected(phase3, tmp_path: Path):
    cfg = _stage(tmp_path, [_record("a", "a.png")])

    with pytest.raises(ValueError, match="not_in_manifest"):
        phase3.resolve_manifest_items(cfg, image_ids=["not_in_manifest"], limit=1)


# ---------------------------------------------------------------- missing file


def test_a_file_listed_but_absent_from_disk_raises(phase3, tmp_path: Path):
    cfg = _stage(
        tmp_path,
        [_record("present", "present.png"), _record("gone", "gone.png")],
        on_disk=["present.png"],
    )

    with pytest.raises(FileNotFoundError) as excinfo:
        phase3.resolve_manifest_items(cfg, image_ids=None, limit=10)

    message = str(excinfo.value)
    assert "gone" in message, "the message must name the item"
    assert "gone.png" in message, "the message must name the path"


def test_a_missing_file_is_not_silently_skipped(phase3, tmp_path: Path):
    cfg = _stage(
        tmp_path,
        [_record("a", "a.png"), _record("b", "b.png")],
        on_disk=["a.png"],
    )

    with pytest.raises(FileNotFoundError):
        phase3.resolve_manifest_items(cfg, image_ids=None, limit=10)


def test_an_empty_manifest_raises(phase3, tmp_path: Path):
    cfg = _stage(tmp_path, [])

    with pytest.raises(ValueError, match="no items"):
        phase3.resolve_manifest_items(cfg, image_ids=None, limit=10)
