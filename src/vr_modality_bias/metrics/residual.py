"""Residual drift ratio + length-invariant complement."""

from __future__ import annotations

import math

import numpy as np

__all__ = ["deep_block", "head_tail_ratio", "residual_drift_ratio"]


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


def head_tail_ratio(
    kl_matrix: np.ndarray,
    t0: int = 5,
) -> float:
    """Length-invariant attenuation indicator: ``mean(tail) / mean(head)``.

    Complements :func:`residual_drift_ratio`. While ``residual_drift_ratio``
    saturates towards ``1`` on long captions even for a perfectly flat
    curve (because ``(T - t0) / T → 1``), this ratio compares the *average*
    deep-KL after ``t0`` against the average before, and is therefore
    independent of caption length when the curve shape is fixed.

    Interpretation:
        - ``~1.0`` — flat curve, signal sustained, no attenuation.
        - ``< 1`` — tail weaker than head → modality bias / drift.
        - ``> 1`` — tail stronger than head (rare).

    Returns:
        ``mean(deep_curve[t0:]) / mean(deep_curve[:t0])`` as a Python float,
        or ``NaN`` when ``caption_len <= t0``, when either segment is empty,
        or when the head mean is non-positive (head can't anchor the ratio).
    """
    if kl_matrix.ndim != 2:
        raise ValueError(
            f"kl_matrix must be 2-D (n_layers, caption_len); got shape {kl_matrix.shape}"
        )

    n_layers, caption_len = kl_matrix.shape
    if caption_len <= t0 or t0 <= 0:
        return math.nan

    l0, l1 = deep_block(n_layers)
    deep_curve = kl_matrix[l0:l1, :].astype(np.float64).mean(axis=0)

    head = deep_curve[:t0]
    tail = deep_curve[t0:]
    if head.size == 0 or tail.size == 0:
        return math.nan

    head_mean = float(head.mean())
    tail_mean = float(tail.mean())
    if not math.isfinite(head_mean) or head_mean <= 0.0:
        return math.nan
    if not math.isfinite(tail_mean):
        return math.nan

    return tail_mean / head_mean
