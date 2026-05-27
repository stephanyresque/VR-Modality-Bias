from __future__ import annotations

import numpy as np
import torch

from vr_modality_bias.metrics.kl import compute_kl_matrix


def _make_hidden(n_layers: int, seq_len: int, vocab: int, *, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n_layers, seq_len, vocab, generator=g, dtype=torch.float32)


def test_kl_matrix_is_zero_when_a_equals_b():
    n_layers, seq_len, vocab = 3, 10, 16
    caption_start, caption_len = 4, 5
    hidden = _make_hidden(n_layers, seq_len, vocab, seed=0)
    kl = compute_kl_matrix(
        torch.nn.Identity(),
        hidden,
        hidden,
        caption_start=caption_start,
        caption_len=caption_len,
    )
    assert kl.shape == (n_layers, caption_len)
    np.testing.assert_allclose(kl, 0.0, atol=1e-5)


def test_kl_matrix_is_strictly_positive_when_distributions_differ_strongly():
    """Construct ``A`` and ``B`` so their softmaxes peak on disjoint vocab regions."""
    n_layers, seq_len, vocab = 2, 8, 32
    caption_start, caption_len = 3, 4

    h_a = torch.zeros(n_layers, seq_len, vocab, dtype=torch.float32)
    h_b = torch.zeros_like(h_a)
    # Peak A on the first half, B on the second half — distinct argmax per token.
    half = vocab // 2
    h_a[..., :half] = 20.0
    h_b[..., half:] = 20.0

    kl = compute_kl_matrix(
        torch.nn.Identity(),
        h_a,
        h_b,
        caption_start=caption_start,
        caption_len=caption_len,
    )
    assert (kl > 1.0).all(), f"expected strongly positive KL, got {kl}"


def test_kl_matrix_is_non_negative_and_finite_for_random_inputs():
    n_layers, seq_len, vocab = 4, 12, 64
    caption_start, caption_len = 5, 6
    h_a = _make_hidden(n_layers, seq_len, vocab, seed=1)
    h_b = _make_hidden(n_layers, seq_len, vocab, seed=2)
    kl = compute_kl_matrix(
        torch.nn.Identity(),
        h_a,
        h_b,
        caption_start=caption_start,
        caption_len=caption_len,
    )
    assert kl.shape == (n_layers, caption_len)
    assert np.isfinite(kl).all()
    assert (kl >= 0.0).all()


def test_kl_matrix_uses_real_linear_lm_head():
    """A learned-style linear head shouldn't break the pipeline."""
    n_layers, seq_len, hidden_dim, vocab = 3, 8, 12, 24
    caption_start, caption_len = 4, 3
    g = torch.Generator().manual_seed(7)
    head = torch.nn.Linear(hidden_dim, vocab, bias=False)
    with torch.no_grad():
        head.weight.copy_(
            torch.randn(vocab, hidden_dim, generator=g, dtype=torch.float32)
        )

    h_a = torch.randn(
        n_layers, seq_len, hidden_dim, generator=g, dtype=torch.float32
    )
    h_b = torch.randn(
        n_layers, seq_len, hidden_dim, generator=g, dtype=torch.float32
    )

    kl = compute_kl_matrix(
        head,
        h_a,
        h_b,
        caption_start=caption_start,
        caption_len=caption_len,
    )
    assert kl.shape == (n_layers, caption_len)
    assert np.isfinite(kl).all()
    assert (kl >= 0.0).all()


def test_kl_matrix_rejects_shape_mismatch():
    h_a = torch.zeros(2, 5, 8)
    h_b = torch.zeros(2, 5, 16)
    try:
        compute_kl_matrix(torch.nn.Identity(), h_a, h_b, caption_start=1, caption_len=2)
    except ValueError as e:
        assert "shape" in str(e).lower()
    else:
        raise AssertionError("expected ValueError")


def test_kl_matrix_rejects_caption_start_zero():
    """caption_start must be >= 1 because the predictive position is t-1."""
    hidden = _make_hidden(2, 5, 8, seed=0)
    try:
        compute_kl_matrix(
            torch.nn.Identity(), hidden, hidden, caption_start=0, caption_len=2
        )
    except ValueError as e:
        assert "predictive" in str(e).lower()
    else:
        raise AssertionError("expected ValueError")


def test_kl_matrix_handles_fp16_inputs():
    """Hidden states arrive as fp16 from the model wrapper; intermediates are fp32."""
    n_layers, seq_len, vocab = 2, 6, 16
    caption_start, caption_len = 2, 3
    h_a = _make_hidden(n_layers, seq_len, vocab, seed=3).to(torch.float16)
    h_b = _make_hidden(n_layers, seq_len, vocab, seed=4).to(torch.float16)
    kl = compute_kl_matrix(
        torch.nn.Identity(),
        h_a,
        h_b,
        caption_start=caption_start,
        caption_len=caption_len,
    )
    assert kl.dtype == np.float32
    assert np.isfinite(kl).all()
    assert (kl >= 0.0).all()
