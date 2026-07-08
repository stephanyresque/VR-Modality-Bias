"""Tests for unit_example.py (per-image unit-example folders).

Covers:
    - KL matrix reconstruction from the nested-list row format,
    - error path when ``image_id`` is missing (lists available ids),
    - tolerance to rows without ``image_id``,
    - deep curve length equals ``caption_len``,
    - ``_format_meta`` includes all required fields and handles NaN,
    - ``emit_for_row`` writes all four artefacts when the image exists,
    - ``emit_for_row`` writes 3 artefacts (no image.jpg) when the source
      image is missing.

CPU-only, no model required.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


_SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "unit_example.py"
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("script_08", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sample_row(
    image_id: str = "img_001",
    *,
    n_layers: int = 30,
    caption_len: int = 8,
    residual_ratio: float = 0.42,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    kl = rng.random((n_layers, caption_len), dtype=np.float32)
    return {
        "image_id": image_id,
        "caption_len": caption_len,
        "n_layers": n_layers,
        "hidden_dim": 576,
        "caption_ref": f"Synthetic caption for {image_id}.",
        "kl": kl.tolist(),
        "residual_ratio": residual_ratio,
        "model_id": "HuggingFaceTB/SmolVLM-256M-Instruct",
        "prompt_key": "caption_short",
        "seed_global": 42,
        "noise_seed": 12345,
        "timestamp_iso": "2026-06-01T22:30:00+00:00",
    }


# ---------------------------------------------------------------- helpers


def test_kl_matrix_reconstructs_from_nested_list():
    script = _load_script_module()
    row = {
        "image_id": "img_001",
        "kl": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
    }
    matrix = script._kl_matrix_from_row(row)
    assert matrix.shape == (2, 3)
    assert matrix.dtype == np.float32
    np.testing.assert_allclose(matrix[0], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(matrix[1], [0.4, 0.5, 0.6])


def test_find_row_returns_matching_row():
    script = _load_script_module()
    rows = [{"image_id": "a", "x": 1}, {"image_id": "b", "x": 2}]
    assert script._find_row(rows, "b") == {"image_id": "b", "x": 2}


def test_find_row_raises_with_available_ids_when_missing():
    script = _load_script_module()
    rows = [{"image_id": "a"}, {"image_id": "b"}, {"image_id": "c"}]
    with pytest.raises(KeyError) as exc_info:
        script._find_row(rows, "z")
    msg = str(exc_info.value)
    assert "z" in msg
    assert "a" in msg and "b" in msg and "c" in msg


def test_find_row_tolerates_rows_with_none_image_id():
    script = _load_script_module()
    rows = [{"image_id": "a"}, {"foo": "no_id_here"}, {"image_id": "b"}]
    assert script._find_row(rows, "a") == {"image_id": "a"}
    with pytest.raises(KeyError):
        script._find_row(rows, "z")


def test_deep_curve_length_matches_caption_len():
    from vr_modality_bias.experiment.plots import average_token_curve

    rng = np.random.default_rng(0)
    kl = rng.random((30, 12), dtype=np.float32)
    curve = average_token_curve(kl)
    assert curve.shape == (12,)


# ------------------------------------------------------------- _format_meta


def test_format_meta_contains_required_fields():
    script = _load_script_module()
    row = _sample_row("img_test", residual_ratio=0.5712)
    text = script._format_meta(row, prompt="Describe the image briefly.")
    for needle in (
        "image_id: img_test",
        "caption_len: 8",
        "residual_ratio: 0.571200",
        "Prompt:",
        "Describe the image briefly.",
        "caption_ref:",
        "Synthetic caption for img_test.",
        "model_id: HuggingFaceTB/SmolVLM-256M-Instruct",
    ):
        assert needle in text, f"missing in meta.txt: {needle!r}"


def test_format_meta_handles_nan_residual_ratio():
    script = _load_script_module()
    row = _sample_row("img_nan", residual_ratio=float("nan"))
    text = script._format_meta(row, prompt="p")
    assert "residual_ratio: nan" in text


# ------------------------------------------------------------- emit_for_row


def test_emit_for_row_writes_all_artefacts_when_image_exists(tmp_path: Path):
    script = _load_script_module()

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    # Synthesise a tiny jpg so the copy step has something to consume.
    arr = np.full((20, 30, 3), 200, dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(images_dir / "img_001.jpg")

    out_dir = tmp_path / "plots" / "unit_examples"
    row = _sample_row("img_001")
    target = script.emit_for_row(
        row,
        out_dir=out_dir,
        images_dir=images_dir,
        prompt="describe the image",
        overwrite=False,
    )
    assert target == out_dir / "img_001"
    assert (target / "image.jpg").is_file()
    assert (target / "meta.txt").is_file()
    heatmap = target / "kl_heatmap.png"
    curve = target / "kl_token_curve.png"
    assert heatmap.is_file()
    assert curve.is_file()
    assert heatmap.read_bytes()[:8] == _PNG_MAGIC
    assert curve.read_bytes()[:8] == _PNG_MAGIC

    # meta.txt content sanity
    meta_text = (target / "meta.txt").read_text(encoding="utf-8")
    assert "image_id: img_001" in meta_text
    assert "describe the image" in meta_text


def test_emit_for_row_skips_image_copy_when_source_missing(tmp_path: Path):
    """Missing source jpg: produces meta.txt + heatmap + curve, no image.jpg."""
    script = _load_script_module()

    images_dir = tmp_path / "images"
    images_dir.mkdir()  # empty — no jpg present

    out_dir = tmp_path / "plots" / "unit_examples"
    row = _sample_row("img_missing")
    target = script.emit_for_row(
        row,
        out_dir=out_dir,
        images_dir=images_dir,
        prompt="p",
        overwrite=False,
    )
    assert not (target / "image.jpg").exists()
    assert (target / "meta.txt").is_file()
    assert (target / "kl_heatmap.png").is_file()
    assert (target / "kl_token_curve.png").is_file()


def test_emit_for_row_is_idempotent_without_overwrite(tmp_path: Path):
    """Calling twice without ``overwrite`` keeps the same files."""
    script = _load_script_module()
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    out_dir = tmp_path / "plots" / "unit_examples"
    row = _sample_row("img_idem")

    first = script.emit_for_row(
        row, out_dir=out_dir, images_dir=images_dir, prompt="p"
    )
    heatmap = first / "kl_heatmap.png"
    mtime_first = heatmap.stat().st_mtime

    # Run again — the file should not be rewritten.
    second = script.emit_for_row(
        row, out_dir=out_dir, images_dir=images_dir, prompt="p"
    )
    assert second == first
    assert heatmap.stat().st_mtime == mtime_first


def test_emit_for_row_overwrite_rewrites_meta_with_new_prompt(tmp_path: Path):
    script = _load_script_module()
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    out_dir = tmp_path / "plots" / "unit_examples"
    row = _sample_row("img_over")

    script.emit_for_row(
        row, out_dir=out_dir, images_dir=images_dir, prompt="OLD"
    )
    script.emit_for_row(
        row, out_dir=out_dir, images_dir=images_dir, prompt="NEW", overwrite=True
    )
    meta = (out_dir / "img_over" / "meta.txt").read_text(encoding="utf-8")
    assert "NEW" in meta
    assert "OLD" not in meta
