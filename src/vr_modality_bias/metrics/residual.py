"""Residual drift ratio"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["deep_block", "residual_drift_ratio"]


def deep_block(n_layers: int) -> tuple[int, int]:
    
    if n_layers <= 0:
        raise ValueError(f"n_layers must be positive, got {n_layers}.")
    l0 = (2 * n_layers) // 3
    return l0, n_layers


def residual_drift_ratio(
    kl_matrix: np.ndarray,
    t0: int = 5,
) -> float:
    
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
