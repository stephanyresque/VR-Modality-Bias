"""Tests for scripts/alucination_viz.py."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


_SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "alucination_viz.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("script_11", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ----------------------------------------------------------- script helpers


def test_kl_matrix_reconstructs_from_nested_list():
    script = _load_script_module()
    row = {"kl": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]}
    matrix = script._kl_matrix_from_row(row)
    assert matrix.shape == (2, 3)
    assert matrix.dtype == np.float32
    np.testing.assert_allclose(matrix[1], [0.4, 0.5, 0.6])


def test_find_row_returns_matching():
    script = _load_script_module()
    rows = [{"image_id": "a"}, {"image_id": "b", "x": 1}]
    assert script._find_row(rows, "b") == {"image_id": "b", "x": 1}


def test_find_row_raises_with_available_ids_when_missing():
    script = _load_script_module()
    rows = [{"image_id": "a"}, {"image_id": "b"}, {"image_id": "c"}]
    with pytest.raises(KeyError) as exc_info:
        script._find_row(rows, "z")
    msg = str(exc_info.value)
    assert "z" in msg
    assert "a" in msg and "b" in msg and "c" in msg


def test_find_row_tolerates_rows_without_image_id():
    script = _load_script_module()
    rows = [{"image_id": "a"}, {"foo": "no_id"}, {"image_id": "b"}]
    assert script._find_row(rows, "a") == {"image_id": "a"}
    with pytest.raises(KeyError):
        script._find_row(rows, "z")


# ------------------------------------------------------- moving-average


def test_moving_average_preserves_length_across_windows():
    script = _load_script_module()
    curve = np.linspace(0.0, 1.0, 25)
    for window in (1, 3, 7, 15, 25):
        out = script._moving_average_padded(curve, window=window)
        assert out.shape == curve.shape, f"window={window}"


def test_moving_average_window_one_is_identity():
    script = _load_script_module()
    curve = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    out = script._moving_average_padded(curve, window=1)
    np.testing.assert_allclose(out, curve)


def test_moving_average_constant_curve_unchanged():
    script = _load_script_module()
    curve = np.full(20, 0.7, dtype=np.float64)
    for window in (3, 7, 15):
        out = script._moving_average_padded(curve, window=window)
        np.testing.assert_allclose(out, 0.7, atol=1e-12)


def test_moving_average_handles_window_larger_than_curve():
    """A pathological case for very short captions — must still run."""
    script = _load_script_module()
    curve = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    out = script._moving_average_padded(curve, window=15)
    assert out.shape == curve.shape


# ----------------------------------------------------------- _normalize


def test_normalize_spans_zero_to_one():
    script = _load_script_module()
    curve = np.array([0.1, 0.5, 0.9])
    out = script._normalize(curve)
    assert float(out.min()) == 0.0
    assert float(out.max()) == 1.0
    np.testing.assert_allclose(out, [0.0, 0.5, 1.0])


def test_normalize_constant_curve_returns_zeros():
    script = _load_script_module()
    curve = np.full(5, 0.42, dtype=np.float64)
    out = script._normalize(curve)
    np.testing.assert_array_equal(out, np.zeros(5))


def test_normalize_empty_curve_returns_empty():
    script = _load_script_module()
    out = script._normalize(np.array([], dtype=np.float64))
    assert out.shape == (0,)


# ---------------------------------------------------------- token_kl.csv / json


def test_write_token_kl_csv_has_caption_len_rows(tmp_path: Path):
    script = _load_script_module()
    tokens = ["The", " cat", " sat", " on", " mat"]
    deep_curve = np.array([0.1, 0.5, 0.9, 0.3, 0.2], dtype=np.float64)

    csv_path, json_path = script._write_token_kl(tmp_path, tokens, deep_curve)

    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    assert len(rows) == 5
    assert [r["position"] for r in rows] == ["0", "1", "2", "3", "4"]
    # kl_norm in [0, 1] for every row
    for r in rows:
        kn = float(r["kl_norm"])
        assert 0.0 <= kn <= 1.0


def test_write_token_kl_preserves_leading_whitespace(tmp_path: Path):
    """The whole point of QUOTE_ALL: tokens with leading spaces survive."""
    script = _load_script_module()
    tokens = ["The", " cat", " sat"]
    deep_curve = np.array([0.1, 0.2, 0.3], dtype=np.float64)

    csv_path, _ = script._write_token_kl(tmp_path, tokens, deep_curve)
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        out = list(reader)
    # Crucial: spaces preserved exactly.
    assert out[0]["token"] == "The"
    assert out[1]["token"] == " cat"
    assert out[2]["token"] == " sat"


def test_write_token_kl_handles_special_csv_characters(tmp_path: Path):
    script = _load_script_module()
    tokens = ['"quoted"', " comma,here", "tab\there", ""]
    deep_curve = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)

    csv_path, _ = script._write_token_kl(tmp_path, tokens, deep_curve)
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        out = list(reader)
    assert out[0]["token"] == '"quoted"'
    assert out[1]["token"] == " comma,here"
    assert out[2]["token"] == "tab\there"
    assert out[3]["token"] == ""


def test_write_token_kl_json_mirrors_csv(tmp_path: Path):
    script = _load_script_module()
    tokens = ["A", " B", " C"]
    deep_curve = np.array([0.0, 0.5, 1.0], dtype=np.float64)

    csv_path, json_path = script._write_token_kl(tmp_path, tokens, deep_curve)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(data) == 3
    assert [d["token"] for d in data] == tokens
    np.testing.assert_allclose([d["kl_deep"] for d in data], [0.0, 0.5, 1.0])
    np.testing.assert_allclose([d["kl_norm"] for d in data], [0.0, 0.5, 1.0])


def test_write_token_kl_rejects_length_mismatch(tmp_path: Path):
    script = _load_script_module()
    with pytest.raises(ValueError):
        script._write_token_kl(tmp_path, ["a", "b"], np.array([0.1, 0.2, 0.3]))


def test_write_token_kl_kl_norm_max_is_one_min_is_zero(tmp_path: Path):
    """The token at min KL must be 0.0 and at max KL must be 1.0 exactly."""
    script = _load_script_module()
    tokens = ["lo", "mid1", "hi", "mid2", "mid3"]
    deep_curve = np.array([0.10, 0.30, 0.90, 0.55, 0.45], dtype=np.float64)
    csv_path, _ = script._write_token_kl(tmp_path, tokens, deep_curve)
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    # min is at position 0, max at position 2
    assert float(rows[0]["kl_norm"]) == 0.0
    assert float(rows[2]["kl_norm"]) == 1.0
    assert float(rows[1]["kl_norm"]) > 0.0
    assert float(rows[1]["kl_norm"]) < 1.0


def test_write_token_kl_creates_parent_dir(tmp_path: Path):
    """Output directory is created on demand."""
    script = _load_script_module()
    nested = tmp_path / "deeper" / "still_deeper"
    csv_path, json_path = script._write_token_kl(
        nested, ["x"], np.array([0.5], dtype=np.float64)
    )
    assert csv_path.is_file()
    assert json_path.is_file()
