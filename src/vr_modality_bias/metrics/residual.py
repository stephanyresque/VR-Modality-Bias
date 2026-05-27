"""Residual drift ratio — Metric 3 (principal) of EXPERIMENT.md §4.5.

The scalar reported per image. Computed by averaging the per-layer KL across
the *deep block* (last third of layers), then taking the fraction of the
resulting curve's mass that lies at positions ``t >= t0``.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["deep_block", "residual_drift_ratio"]


def deep_block(n_layers: int) -> tuple[int, int]:
    """Return the half-open layer range ``[l0, l1)`` defining the deep block.

    Following EXPERIMENT.md §4.5 Metric 3::

        l0 = floor(2 * n_layers / 3)
        l1 = n_layers

    The choice of "last third" follows the empirical observation that the
    predictive signal concentrates in late layers (VISTA, Li et al. 2025).
    """
    if n_layers <= 0:
        raise ValueError(f"n_layers must be positive, got {n_layers}.")
    l0 = (2 * n_layers) // 3
    return l0, n_layers


def residual_drift_ratio(
    kl_matrix: np.ndarray,
    t0: int = 5,
) -> float:
    """Residual drift ratio of the deep-averaged KL curve.

    Computes the deep-block-averaged KL curve over tokens, then returns the
    fraction of its total mass that falls at positions ``t >= t0``::

        residual_ratio = sum_{t >= t0} KL_deep(t) / sum_{t} KL_deep(t)

    Args:
        kl_matrix: ``(n_layers, caption_len)`` array, typically the output
            of :func:`vr_modality_bias.metrics.kl.compute_kl_matrix`.
        t0: First token included in the tail. Defaults to 5 per §4.5.

    Returns:
        The ratio as a Python float, or ``NaN`` if the total mass is zero
        / non-finite, or if ``caption_len <= t0``.
    """
    if kl_matrix.ndim != 2:
        raise ValueError(
            f"kl_matrix must be 2-D (n_layers, caption_len); got shape {kl_matrix.shape}"
        )

    n_layers, caption_len = kl_matrix.shape
    if caption_len <= t0:
        return math.nan

    l0, l1 = deep_block(n_layers)
    deep_curve = kl_matrix[l0:l1, :].astype(np.float64).mean(axis=0)

    total_mass = float(deep_curve.sum())
    if not math.isfinite(total_mass) or total_mass == 0.0:
        return math.nan

    tail_mass = float(deep_curve[t0:].sum())
    return tail_mass / total_mass
