"""Plotting helpers for the four ``plots/*.png`` artefacts of Phase 5."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  
import numpy as np  

from vr_modality_bias.metrics.residual import deep_block  

__all__ = [
    "average_matrices",
    "average_token_curve",
    "pad_to_max_caption_len",
    "plot_heatmap",
    "plot_token_curve",
    "plot_unit_example_panel",
]


def pad_to_max_caption_len(matrices: list[np.ndarray]) -> np.ndarray:
    """Stack ``(n_layers, T_i)`` matrices into ``(n_images, n_layers, max_T)``"""
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
    """Deep-block-averaged token curve from a per-layer mean matrix"""
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


def plot_unit_example_panel(
    *,
    path: Path,
    image_id: str,
    kl_matrix: np.ndarray,
    deep_curve: np.ndarray,
    prompt: str,
    caption_ref: str,
    residual_ratio: float,
    image_array: np.ndarray | None = None,
    cmap: str = "viridis",
    cbar_label: str = "KL divergence (nats)",
    dpi: int = 150,
) -> Path:
    
    if kl_matrix.ndim != 2:
        raise ValueError(f"kl_matrix must be 2-D; got {kl_matrix.shape}")
    if deep_curve.ndim != 1:
        raise ValueError(f"deep_curve must be 1-D; got {deep_curve.shape}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10.0, 12.0), dpi=dpi)
    gs = fig.add_gridspec(
        3, 2, height_ratios=[1.0, 1.2, 0.8], width_ratios=[1.0, 1.4]
    )

    ax_img = fig.add_subplot(gs[0, 0])
    if image_array is not None:
        ax_img.imshow(np.asarray(image_array))
    else:
        ax_img.text(
            0.5,
            0.5,
            f"(image {image_id} not available)",
            ha="center",
            va="center",
            transform=ax_img.transAxes,
            fontsize=11,
            color="gray",
            style="italic",
        )
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    ax_img.set_title(f"image_id = {image_id}")

    ax_text = fig.add_subplot(gs[0, 1])
    ax_text.axis("off")
    rr_str = f"{residual_ratio:.4f}" if np.isfinite(residual_ratio) else "nan"
    context = (
        f"Prompt:\n{prompt}\n\n"
        f"caption_ref:\n{caption_ref}\n\n"
        f"residual_ratio = {rr_str}"
    )
    ax_text.text(
        0.0,
        1.0,
        context,
        ha="left",
        va="top",
        transform=ax_text.transAxes,
        fontsize=10,
        wrap=True,
        family="monospace",
    )

    ax_hm = fig.add_subplot(gs[1, :])
    im = ax_hm.imshow(kl_matrix, aspect="auto", origin="lower", cmap=cmap)
    ax_hm.set_xlabel("Token index (caption position)")
    ax_hm.set_ylabel("Layer index")
    ax_hm.set_title(f"KL(B || A) per (layer, token) — single image {image_id}")
    cbar = fig.colorbar(im, ax=ax_hm)
    cbar.set_label(cbar_label)

    ax_cv = fig.add_subplot(gs[2, :])
    x = np.arange(deep_curve.shape[0])
    ax_cv.plot(x, deep_curve, marker="o", markersize=3, linewidth=1.5)
    ax_cv.set_xlabel("Token index")
    ax_cv.set_ylabel("Mean KL (deep block, last third)")
    ax_cv.set_title("Deep-block mean KL vs. token — single image")
    ax_cv.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
