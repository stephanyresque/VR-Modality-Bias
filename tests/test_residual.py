from __future__ import annotations

import math

import numpy as np

from vr_modality_bias.metrics.residual import deep_block, residual_drift_ratio


def _kl_matrix_with_curve(curve: np.ndarray, n_layers: int = 6) -> np.ndarray:
    """Tile ``curve`` (shape ``(T,)``) into a ``(n_layers, T)`` KL matrix.

    Because every layer carries the same curve, the deep-block mean equals
    the curve itself — that lets the tests target ``residual_drift_ratio``
    independently of the deep-block reduction.
    """
    return np.tile(curve.astype(np.float32), (n_layers, 1))


def test_constant_curve_yields_t_minus_t0_over_t():
    t0 = 5
    T = 20
    curve = np.full(T, 0.7, dtype=np.float32)
    kl = _kl_matrix_with_curve(curve)
    expected = (T - t0) / T
    assert math.isclose(residual_drift_ratio(kl, t0=t0), expected, rel_tol=1e-6)


def test_curve_concentrated_before_t0_yields_zero():
    t0 = 5
    T = 20
    curve = np.zeros(T, dtype=np.float32)
    curve[:t0] = 1.0
    kl = _kl_matrix_with_curve(curve)
    assert residual_drift_ratio(kl, t0=t0) == 0.0


def test_curve_concentrated_after_t0_yields_one():
    t0 = 5
    T = 20
    curve = np.zeros(T, dtype=np.float32)
    curve[t0:] = 1.0
    kl = _kl_matrix_with_curve(curve)
    assert residual_drift_ratio(kl, t0=t0) == 1.0


def test_total_mass_zero_returns_nan():
    kl = np.zeros((6, 20), dtype=np.float32)
    result = residual_drift_ratio(kl, t0=5)
    assert math.isnan(result)


def test_caption_len_less_than_or_equal_to_t0_returns_nan():
    # caption_len == t0
    kl = np.ones((6, 5), dtype=np.float32)
    assert math.isnan(residual_drift_ratio(kl, t0=5))
    # caption_len < t0
    kl_short = np.ones((6, 3), dtype=np.float32)
    assert math.isnan(residual_drift_ratio(kl_short, t0=5))


def test_non_finite_total_mass_returns_nan():
    kl = np.array([[np.inf, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]] * 6)
    assert math.isnan(residual_drift_ratio(kl, t0=5))


def test_deep_block_returns_last_third():
    # 30 layers → l0 = floor(60/3) = 20, l1 = 30
    assert deep_block(30) == (20, 30)
    # 31 layers → l0 = floor(62/3) = 20, l1 = 31
    assert deep_block(31) == (20, 31)
    # 32 layers → l0 = floor(64/3) = 21, l1 = 32
    assert deep_block(32) == (21, 32)
    # 3 layers → l0 = floor(6/3) = 2, l1 = 3
    assert deep_block(3) == (2, 3)


def test_deep_block_rejects_non_positive_n_layers():
    try:
        deep_block(0)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_residual_uses_only_deep_block_layers():
    """Layers outside the deep block must not contribute to the curve."""
    n_layers = 30
    T = 20
    t0 = 5
    l0, _ = deep_block(n_layers)

    # Curve A: nonzero only outside the deep block (these layers must be ignored).
    kl = np.zeros((n_layers, T), dtype=np.float32)
    kl[:l0, :] = 1.0  # all "shallow" layers carry mass
    # All deep-block layers are zero, so deep-curve is identically zero -> NaN.
    assert math.isnan(residual_drift_ratio(kl, t0=t0))

    # Curve B: nonzero only inside the deep block, concentrated after t0.
    kl = np.zeros((n_layers, T), dtype=np.float32)
    kl[l0:, t0:] = 1.0
    assert residual_drift_ratio(kl, t0=t0) == 1.0
