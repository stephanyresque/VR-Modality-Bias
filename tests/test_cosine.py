from __future__ import annotations

import numpy as np
import torch

from vr_modality_bias.metrics.cosine import compute_cosine_distance_matrix


def _broadcast_vector_to_hidden(
    vec: torch.Tensor,
    *,
    n_layers: int,
    seq_len: int,
) -> torch.Tensor:
    """Repeat ``vec`` (shape ``(D,)``) across ``(n_layers, seq_len)``."""
    return vec.view(1, 1, -1).expand(n_layers, seq_len, -1).contiguous()


def test_cos_dist_is_zero_for_identical_vectors():
    n_layers, seq_len, dim = 3, 8, 5
    caption_start, caption_len = 2, 4
    vec = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    h = _broadcast_vector_to_hidden(vec, n_layers=n_layers, seq_len=seq_len)
    cos = compute_cosine_distance_matrix(h, h, caption_start, caption_len)
    assert cos.shape == (n_layers, caption_len)
    np.testing.assert_allclose(cos, 0.0, atol=1e-6)


def test_cos_dist_is_two_for_anti_parallel_vectors():
    n_layers, seq_len, dim = 2, 6, 4
    caption_start, caption_len = 1, 3
    vec = torch.tensor([1.0, -1.0, 1.0, -1.0])
    h_a = _broadcast_vector_to_hidden(vec, n_layers=n_layers, seq_len=seq_len)
    h_b = _broadcast_vector_to_hidden(-vec, n_layers=n_layers, seq_len=seq_len)
    cos = compute_cosine_distance_matrix(h_a, h_b, caption_start, caption_len)
    np.testing.assert_allclose(cos, 2.0, atol=1e-6)


def test_cos_dist_is_one_for_orthogonal_vectors():
    n_layers, seq_len, dim = 2, 6, 4
    caption_start, caption_len = 1, 3
    vec_a = torch.tensor([1.0, 0.0, 0.0, 0.0])
    vec_b = torch.tensor([0.0, 1.0, 0.0, 0.0])
    h_a = _broadcast_vector_to_hidden(vec_a, n_layers=n_layers, seq_len=seq_len)
    h_b = _broadcast_vector_to_hidden(vec_b, n_layers=n_layers, seq_len=seq_len)
    cos = compute_cosine_distance_matrix(h_a, h_b, caption_start, caption_len)
    np.testing.assert_allclose(cos, 1.0, atol=1e-6)


def test_cos_dist_is_clamped_to_non_negative_for_tiny_numerical_overshoot():
    """When ``cos_sim`` is just above 1 due to fp16 noise, the result must be 0."""
    n_layers, seq_len, dim = 2, 4, 3
    caption_start, caption_len = 1, 2
    h = torch.randn(n_layers, seq_len, dim, dtype=torch.float16)
    cos = compute_cosine_distance_matrix(h, h, caption_start, caption_len)
    assert (cos >= 0.0).all()
    np.testing.assert_allclose(cos, 0.0, atol=1e-3)


def test_cos_dist_has_expected_shape_and_dtype():
    n_layers, seq_len, dim = 4, 9, 7
    caption_start, caption_len = 3, 5
    h_a = torch.randn(n_layers, seq_len, dim, dtype=torch.float32)
    h_b = torch.randn(n_layers, seq_len, dim, dtype=torch.float32)
    cos = compute_cosine_distance_matrix(h_a, h_b, caption_start, caption_len)
    assert cos.shape == (n_layers, caption_len)
    assert cos.dtype == np.float32


def test_cos_dist_rejects_shape_mismatch():
    h_a = torch.zeros(2, 5, 4)
    h_b = torch.zeros(2, 5, 8)
    try:
        compute_cosine_distance_matrix(h_a, h_b, caption_start=1, caption_len=2)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
