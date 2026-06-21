"""Deep-block KL summaries: ``share_tail`` (current) + legacy metrics."""

from __future__ import annotations

import math

import numpy as np

__all__ = ["deep_block", "share_tail", "head_tail_ratio", "residual_drift_ratio"]


def deep_block(n_layers: int) -> tuple[int, int]:
    """Return ``(lo, hi)`` for the deep-third layer slice (used as a half-open range)."""
    if n_layers <= 0:
        raise ValueError(f"n_layers must be positive, got {n_layers}.")
    l0 = (2 * n_layers) // 3
    return l0, n_layers


def share_tail(kl_matrix: np.ndarray) -> float:
    """Fraction of deep-block KL mass that sits in the tail half of the caption.

    The OFFICIAL post-Block-3 attenuation summary. Computed as::

        deep_curve = mean over deep_block layers of KL per token
        tail_start = caption_len // 2          # floor split, see below
        share_tail = sum(deep_curve[tail_start:]) / sum(deep_curve)

    Properties (all covered by ``tests/test_residual.py``):

    * **Bounded** ``[0, 1]`` by construction (tail is a subset of the
      total; both sums are non-negative whenever the deep curve is).
    * **Invariant under positive multiplicative scaling**: if SPARC
      amplifies ``deep_curve`` by any ``k > 0``, ``share_tail`` does not
      change because ``k`` cancels in num/denom. This is the central
      property that motivated replacing ``head_tail_ratio``.
    * **Length-robust**: the same curve shape gives the same ``share_tail``
      regardless of whether the caption has 10 or 200 tokens (the metric
      depends on the *shape*, not the absolute length).
    * **Neutral reference at 0.5**: a perfectly flat deep curve splits
      mass equally between halves → ``share_tail = 0.5``. Below 0.5 means
      the influence drops in the tail (modality-bias signature); above
      0.5 means it concentrates or sustains in the tail.

    Choice of split (``tail_start = caption_len // 2``):
        Using ``floor`` instead of ``ceil`` makes the formula symmetric
        for even captions (exactly half-and-half) and assigns the middle
        token to the TAIL for odd captions (so the tail is the larger
        half by one element when the caption is odd). The asymmetry is
        ``1 / caption_len`` and vanishes for captions ≥ ~20 tokens.

    Returns:
        ``float`` in ``[0, 1]``, or ``NaN`` if ``caption_len < 2`` or
        ``sum(deep_curve) <= 0`` (degenerate cases — caller decides what
        to do; no implicit zero or fallback).
    """
    if kl_matrix.ndim != 2:
        raise ValueError(
            f"kl_matrix must be 2-D (n_layers, caption_len); got shape {kl_matrix.shape}"
        )

    n_layers, caption_len = kl_matrix.shape
    if caption_len < 2:
        return math.nan

    l0, l1 = deep_block(n_layers)
    deep_curve = kl_matrix[l0:l1, :].astype(np.float64).mean(axis=0)

    total = float(deep_curve.sum())
    if not math.isfinite(total) or total <= 0.0:
        return math.nan

    tail_start = caption_len // 2
    tail = float(deep_curve[tail_start:].sum())
    # Numerical safety: clamp to [0, 1]. A non-negative deep_curve with a
    # positive total can in principle give exactly 0.0 or 1.0 (all mass at
    # one extreme); float rounding can otherwise nudge a hair outside.
    value = tail / total
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def residual_drift_ratio(
    kl_matrix: np.ndarray,
    t0: int = 5,
) -> float:
    """Legacy length-saturating tail-mass ratio. Use :func:`share_tail` instead."""
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
    """**DEPRECATED — use :func:`share_tail`.** ``mean(tail) / mean(head)``.

    .. deprecated:: Block-3 of the LLaVA migration
        This ratio is unbounded above and inflates to 50+/NaN under SPARC
        (which amplifies the magnitude of the deep KL curve). The
        replacement, :func:`share_tail`, is bounded ``[0, 1]`` and
        invariant under any positive scaling — both properties this ratio
        lacks. Kept here only so orphan scripts (scripts/12, 15, 16, ...)
        can still import the symbol without breaking. **Do not use as a
        headline metric in new analyses.**

    Interpretation when it was the headline metric:
        - ``~1.0`` — flat curve, signal sustained.
        - ``< 1``  — tail weaker than head → modality bias.
        - ``> 1``  — tail stronger than head (rare).

    Returns:
        ``mean(deep_curve[t0:]) / mean(deep_curve[:t0])`` as a Python
        float, or ``NaN`` when ``caption_len <= t0``, when either segment
        is empty, or when the head mean is non-positive.
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
