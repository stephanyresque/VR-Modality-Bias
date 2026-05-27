"""Plotting helpers for the four ``plots/*.png`` artefacts of Phase 5.

Pure functions that take NumPy arrays and write PNG files; no I/O against
``metrics.parquet`` here — that lives in ``scripts/07_make_plots.py``.

Why the matplotlib import is gated:
    The Agg backend is selected at module load so headless DGX / CI
    environments without a display can render figures. Doing this here, in
    the plotting module, keeps the rest of the package importable without
    pulling matplotlib (e.g. for unit tests that don't touch plots).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402  — Agg backend must be selected first
import numpy as np  # noqa: E402

from vr_modality_bias.metrics.residual import deep_block  # noqa: E402

__all__ = [
    "average_matrices",
    "average_token_curve",
    "pad_to_max_caption_len",
    "plot_heatmap",
    "plot_token_curve",
]


def pad_to_max_caption_len(matrices: list[np.ndarray]) -> np.ndarray:
    """Stack ``(n_layers, T_i)`` matrices into ``(n_images, n_layers, max_T)``.

    Shorter sequences are right-padded with NaN so ``np.nanmean`` averages
    only the images that actually have a token at that position.

    Raises:
        ValueError: if ``matrices`` is empty or has inconsistent ``n_layers``.
    """
    if not matrices:
        raise ValueError("Cannot pad an empty list of matrices.")
    n_layers = matrices[0].shape[0]
    for i, m in enumerate(matrices):
        if m.ndim != 2:
            raise ValueError(f"matrix {i} is not 2-D (shape={m.shape}).")
        if m.shape[0] != n_layers:
            raise ValueError(
                f"matrix {i} has n_layers={m.shape[0]}, "
                f"expected {n_layers} (first matrix)."
            )
    max_T = max(m.shape[1] for m in matrices)
    stacked = np.full((len(matrices), n_layers, max_T), np.nan, dtype=np.float64)
    for i, m in enumerate(matrices):
        T = m.shape[1]
        stacked[i, :, :T] = m
    return stacked


def average_matrices(matrices: list[np.ndarray]) -> np.ndarray:
    """NaN-aware mean over images. Returns ``(n_layers, max_T)`` ``float64``."""
    stacked = pad_to_max_caption_len(matrices)
    with np.errstate(all="ignore"):
        return np.nanmean(stacked, axis=0)


def average_token_curve(mean_matrix: np.ndarray) -> np.ndarray:
    """Deep-block-averaged token curve from a per-layer mean matrix.

    Averages rows in ``[l0, l1)`` where ``(l0, l1) = deep_block(n_layers)``.
    Per EXPERIMENT.md §4.5 Metric 3 the deep block is the last third.
    """
    if mean_matrix.ndim != 2:
        raise ValueError(f"mean_matrix must be 2-D; got {mean_matrix.shape}")
    n_layers, _ = mean_matrix.shape
    l0, l1 = deep_block(n_layers)
    with np.errstate(all="ignore"):
        return np.nanmean(mean_matrix[l0:l1, :], axis=0)


def plot_heatmap(
    matrix: np.ndarray,
    *,
    path: Path,
    title: str,
    cbar_label: str,
    cmap: str = "viridis",
    dpi: int = 150,
) -> Path:
    """Render a ``layer × token`` heatmap and save it to ``path`` (PNG)."""
    if matrix.ndim != 2:
        raise ValueError(f"matrix must be 2-D; got {matrix.shape}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=dpi)
    im = ax.imshow(matrix, aspect="auto", origin="lower", cmap=cmap)
    ax.set_xlabel("Token index (caption position)")
    ax.set_ylabel("Layer index")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_token_curve(
    curve: np.ndarray,
    *,
    path: Path,
    title: str,
    y_label: str,
    dpi: int = 150,
) -> Path:
    """Render a 1-D ``deep-mean vs. token`` curve and save it to ``path`` (PNG)."""
    if curve.ndim != 1:
        raise ValueError(f"curve must be 1-D; got {curve.shape}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.0, 4.0), dpi=dpi)
    x = np.arange(curve.shape[0])
    ax.plot(x, curve, marker="o", markersize=3, linewidth=1.5)
    ax.set_xlabel("Token index")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
