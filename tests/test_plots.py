from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vr_modality_bias.experiment.plots import (
    average_matrices,
    average_token_curve,
    pad_to_max_caption_len,
    plot_heatmap,
    plot_token_curve,
)

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_pad_to_max_caption_len_pads_with_nan():
    a = np.full((3, 5), 1.0, dtype=np.float32)
    b = np.full((3, 8), 2.0, dtype=np.float32)
    stacked = pad_to_max_caption_len([a, b])
    assert stacked.shape == (2, 3, 8)
    # Image 0: first 5 cols equal 1, last 3 cols are NaN.
    assert np.all(stacked[0, :, :5] == 1.0)
    assert np.all(np.isnan(stacked[0, :, 5:]))
    # Image 1: full 8 cols equal 2.
    assert np.all(stacked[1] == 2.0)


def test_pad_to_max_caption_len_rejects_empty_list():
    with pytest.raises(ValueError):
        pad_to_max_caption_len([])


def test_pad_to_max_caption_len_rejects_inconsistent_n_layers():
    a = np.zeros((3, 5), dtype=np.float32)
    b = np.zeros((4, 5), dtype=np.float32)
    with pytest.raises(ValueError):
        pad_to_max_caption_len([a, b])


def test_average_matrices_nan_aware():
    """Position 5..7 only present in matrix B → mean equals B's value, not NaN."""
    a = np.full((2, 5), 1.0, dtype=np.float32)
    b = np.full((2, 8), 3.0, dtype=np.float32)
    mean = average_matrices([a, b])
    assert mean.shape == (2, 8)
    np.testing.assert_allclose(mean[:, :5], 2.0)  # (1 + 3) / 2 across both images
    np.testing.assert_allclose(mean[:, 5:], 3.0)  # only B contributes


def test_average_token_curve_uses_deep_block_only():
    n_layers, T = 30, 10
    mat = np.zeros((n_layers, T), dtype=np.float64)
    # Shallow layers carry mass that must NOT be averaged in.
    mat[:20, :] = 99.0
    # Deep block (indices 20..29): mass = 1.0 throughout.
    mat[20:, :] = 1.0
    curve = average_token_curve(mat)
    assert curve.shape == (T,)
    np.testing.assert_allclose(curve, 1.0)


def test_plot_heatmap_writes_png_file(tmp_path: Path):
    mat = np.linspace(0.0, 1.0, num=30 * 12, dtype=np.float32).reshape(30, 12)
    path = tmp_path / "kl_heatmap.png"
    out = plot_heatmap(
        mat,
        path=path,
        title="Test heatmap",
        cbar_label="value",
    )
    assert out == path
    assert path.is_file()
    assert path.stat().st_size > 1024
    assert path.read_bytes()[:8] == _PNG_MAGIC


def test_plot_token_curve_writes_png_file(tmp_path: Path):
    curve = np.array([0.1, 0.2, 0.4, 0.7, 0.5, 0.3, 0.2, 0.1], dtype=np.float32)
    path = tmp_path / "kl_token_curve.png"
    out = plot_token_curve(
        curve,
        path=path,
        title="Test curve",
        y_label="metric",
    )
    assert out == path
    assert path.is_file()
    assert path.stat().st_size > 512
    assert path.read_bytes()[:8] == _PNG_MAGIC


def test_plot_heatmap_handles_nan_cells(tmp_path: Path):
    """Cells filled with NaN must render without crashing."""
    mat = np.full((10, 20), np.nan, dtype=np.float64)
    mat[:, :10] = 0.5
    path = tmp_path / "with_nans.png"
    plot_heatmap(mat, path=path, title="NaN cells", cbar_label="v")
    assert path.is_file()


def test_plot_heatmap_rejects_non_2d_input(tmp_path: Path):
    with pytest.raises(ValueError):
        plot_heatmap(
            np.zeros((3,), dtype=np.float32),
            path=tmp_path / "x.png",
            title="bad",
            cbar_label="v",
        )


def test_plot_token_curve_rejects_non_1d_input(tmp_path: Path):
    with pytest.raises(ValueError):
        plot_token_curve(
            np.zeros((3, 4), dtype=np.float32),
            path=tmp_path / "x.png",
            title="bad",
            y_label="v",
        )
