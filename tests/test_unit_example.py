"""Tests for the unit-example panel (script 08 + the new plot helper).

Covers the four acceptance criteria laid out in the task spec:
    1. The KL matrix is reconstructed correctly from the nested-list row.
    2. A missing ``image_id`` raises with a message that lists the
       available ids.
    3. The deep curve length equals ``caption_len``.
    4. The panel still renders (with placeholder text) when the source
       image is absent.

Also exercises the panel's defensive input validation.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from vr_modality_bias.experiment.plots import (
    average_token_curve,
    plot_unit_example_panel,
)


# Scripts whose filename starts with a digit can't be imported the usual
# way, so we load the module from its file path.
_SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "08_unit_example.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("script_08", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------- script-08 helpers


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
    """Defensive: a row missing ``image_id`` should not break the lookup."""
    script = _load_script_module()
    rows = [{"image_id": "a"}, {"foo": "no_id_here"}, {"image_id": "b"}]
    assert script._find_row(rows, "a") == {"image_id": "a"}
    with pytest.raises(KeyError):
        script._find_row(rows, "z")


# ----------------------------------------------------------- deep-curve length


def test_deep_curve_length_matches_caption_len():
    n_layers, caption_len = 30, 8
    rng = np.random.default_rng(0)
    kl = rng.random((n_layers, caption_len), dtype=np.float32)
    curve = average_token_curve(kl)
    assert curve.shape == (caption_len,)


# -------------------------------------------------- plot_unit_example_panel


def test_unit_example_panel_writes_png_with_image(tmp_path: Path):
    n_layers, caption_len = 30, 6
    rng = np.random.default_rng(1)
    kl = rng.random((n_layers, caption_len), dtype=np.float32)
    curve = average_token_curve(kl)
    image = rng.integers(0, 256, size=(40, 50, 3), dtype=np.uint8)

    out = plot_unit_example_panel(
        path=tmp_path / "unit_example.png",
        image_id="img_test",
        kl_matrix=kl,
        deep_curve=curve,
        prompt="describe the image",
        caption_ref="A test caption with content.",
        residual_ratio=0.42,
        image_array=image,
    )
    assert out.is_file()
    assert out.stat().st_size > 1024
    assert out.read_bytes()[:8] == _PNG_MAGIC


def test_unit_example_panel_works_without_image(tmp_path: Path):
    """Missing image must yield a valid PNG via the placeholder branch."""
    n_layers, caption_len = 6, 4
    kl = np.ones((n_layers, caption_len), dtype=np.float32)
    curve = average_token_curve(kl)

    out = plot_unit_example_panel(
        path=tmp_path / "unit_example_no_img.png",
        image_id="img_missing",
        kl_matrix=kl,
        deep_curve=curve,
        prompt="describe",
        caption_ref="placeholder caption.",
        residual_ratio=0.5,
        image_array=None,
    )
    assert out.is_file()
    assert out.read_bytes()[:8] == _PNG_MAGIC


def test_unit_example_panel_handles_nan_residual_ratio(tmp_path: Path):
    """NaN residual_ratio shows as ``nan`` in the text block, not as a crash."""
    n_layers, caption_len = 3, 3
    kl = np.zeros((n_layers, caption_len), dtype=np.float32)
    curve = average_token_curve(kl)

    out = plot_unit_example_panel(
        path=tmp_path / "unit_example_nan.png",
        image_id="img_nan",
        kl_matrix=kl,
        deep_curve=curve,
        prompt="describe",
        caption_ref="short.",
        residual_ratio=float("nan"),
        image_array=None,
    )
    assert out.is_file()


def test_unit_example_panel_rejects_non_2d_kl(tmp_path: Path):
    with pytest.raises(ValueError):
        plot_unit_example_panel(
            path=tmp_path / "bad.png",
            image_id="x",
            kl_matrix=np.zeros((5,), dtype=np.float32),
            deep_curve=np.zeros((3,), dtype=np.float32),
            prompt="p",
            caption_ref="c",
            residual_ratio=0.0,
        )


def test_unit_example_panel_rejects_non_1d_curve(tmp_path: Path):
    with pytest.raises(ValueError):
        plot_unit_example_panel(
            path=tmp_path / "bad.png",
            image_id="x",
            kl_matrix=np.zeros((3, 4), dtype=np.float32),
            deep_curve=np.zeros((3, 4), dtype=np.float32),
            prompt="p",
            caption_ref="c",
            residual_ratio=0.0,
        )
