"""KL divergence per (layer, token) — Metric 1 of EXPERIMENT.md §4.5.

For each layer ``l`` and each caption position ``t``, project the *predictive*
hidden state ``h_{t-1}`` through the model's ``lm_head`` to obtain logits,
softmax to probabilities, and compute the divergence

.. math::

    D_{KL}(P^{l,t}_B \\,\\|\\, P^{l,t}_A)
       = \\sum_v P^{l,t}_B(v) \\log \\frac{P^{l,t}_B(v)}{P^{l,t}_A(v)}.

The full softmax is replaced by the top-K + ``other-mass`` approximation
described in §4.5 (``K = 50`` by default). All intermediate computation is
performed in fp32 regardless of the input dtype. Results are clamped at
``>= 0`` (a tiny negative value can arise when ``P_B`` is essentially zero
on the union and we lose precision in the other-mass term).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["compute_kl_matrix"]

_DEFAULT_EPS = 1e-12


def _infer_head_device_and_dtype(
    lm_head: torch.nn.Module,
) -> tuple[torch.device | None, torch.dtype | None]:
    """Return ``(device, dtype)`` of ``lm_head``'s first parameter, or ``(None, None)``.

    ``nn.Identity`` (used in tests) has no parameters; the caller falls back
    to the hidden-states tensor's device and to fp32.
    """
    try:
        param = next(lm_head.parameters())
    except StopIteration:
        return None, None
    return param.device, param.dtype


def compute_kl_matrix(
    lm_head: torch.nn.Module,
    hidden_states_A: torch.Tensor,
    hidden_states_B: torch.Tensor,
    caption_start: int,
    caption_len: int,
    top_k: int = 50,
    device: torch.device | None = None,
    eps: float = _DEFAULT_EPS,
) -> np.ndarray:
    """KL(B || A) per (layer, token) via top-K + other-mass.

    Args:
        lm_head: The model's language-modelling head (e.g.
            ``model.lm_head``), used to project hidden states into vocab
            logits. Must be a callable :class:`torch.nn.Module` that maps
            ``(..., hidden_dim) -> (..., vocab_size)``.
        hidden_states_A: Tensor of shape ``(n_layers, seq_len, hidden_dim)``
            from condition A (real image).
        hidden_states_B: Tensor of shape ``(n_layers, seq_len, hidden_dim)``
            from condition B (noise image). Must match ``hidden_states_A``.
        caption_start: Index of the first caption token in the full input
            sequence (so the predictive position for token ``caption_start``
            is ``caption_start - 1``).
        caption_len: Number of caption tokens; the output has
            ``caption_len`` columns.
        top_k: Vocabulary size of the top-K union used for the bulk-mass
            estimate. Defaults to 50 per §4.5.
        device: Compute device. If ``None``, uses the ``lm_head``'s device.
        eps: Small constant added before logs and divisions for numerical
            stability. Defaults to ``1e-12`` per §4.5.

    Returns:
        A ``(n_layers, caption_len)`` ``float32`` array.
    """
    if hidden_states_A.shape != hidden_states_B.shape:
        raise ValueError(
            f"hidden_states_A.shape={tuple(hidden_states_A.shape)} != "
            f"hidden_states_B.shape={tuple(hidden_states_B.shape)}"
        )
    if hidden_states_A.dim() != 3:
        raise ValueError(
            f"hidden_states must be 3-D (n_layers, seq_len, hidden_dim); "
            f"got shape {tuple(hidden_states_A.shape)}"
        )

    n_layers, seq_len, _ = hidden_states_A.shape
    pos_start = caption_start - 1
    pos_end = caption_start + caption_len - 1
    if pos_start < 0:
        raise ValueError(
            f"caption_start={caption_start} yields negative predictive position. "
            "Need caption_start >= 1."
        )
    if pos_end > seq_len:
        raise ValueError(
            f"caption_start + caption_len - 1 = {pos_end} exceeds "
            f"seq_len={seq_len}."
        )

    head_device, head_dtype = _infer_head_device_and_dtype(lm_head)
    if device is None:
        device = head_device if head_device is not None else hidden_states_A.device
    # Project in the lm_head's native dtype (fp16/bf16 on GPU, fp32 on CPU);
    # the softmax and KL accumulation are always promoted to fp32 below.
    project_dtype = head_dtype if head_dtype is not None else torch.float32

    kl_matrix = np.empty((n_layers, caption_len), dtype=np.float32)

    for layer in range(n_layers):
        h_a = (
            hidden_states_A[layer, pos_start:pos_end, :]
            .to(device=device, dtype=project_dtype)
            .contiguous()
        )
        h_b = (
            hidden_states_B[layer, pos_start:pos_end, :]
            .to(device=device, dtype=project_dtype)
            .contiguous()
        )

        with torch.no_grad():
            logits_a = lm_head(h_a).float()  # (T, V) — promote to fp32
            logits_b = lm_head(h_b).float()

        vocab_size = logits_a.shape[-1]
        k = min(top_k, vocab_size)

        p_a = F.softmax(logits_a, dim=-1)
        p_b = F.softmax(logits_b, dim=-1)

        # Indices forming the union top-K(A) ∪ top-K(B).
        _, idx_a = torch.topk(logits_a, k=k, dim=-1)
        _, idx_b = torch.topk(logits_b, k=k, dim=-1)
        mask = torch.zeros_like(p_a, dtype=torch.bool)
        mask.scatter_(1, idx_a, True)
        mask.scatter_(1, idx_b, True)

        # Bulk term: contribution from the union only.
        ratio = p_b / p_a.clamp(min=eps)
        log_ratio = torch.log(ratio.clamp(min=eps))
        bulk = (p_b * log_ratio).masked_fill(~mask, 0.0).sum(dim=-1)

        # Other-mass term: residual probability outside the union.
        mass_a = (p_a * mask).sum(dim=-1)
        mass_b = (p_b * mask).sum(dim=-1)
        other_a = (1.0 - mass_a).clamp(min=0.0)
        other_b = (1.0 - mass_b).clamp(min=0.0)
        tail = other_b * torch.log((other_b + eps) / (other_a + eps))

        kl_row = (bulk + tail).clamp(min=0.0)
        kl_matrix[layer] = kl_row.detach().cpu().numpy().astype(np.float32)

    return kl_matrix
