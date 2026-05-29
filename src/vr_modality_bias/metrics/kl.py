"""KL divergence per (layer, token)"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["compute_kl_matrix"]

_DEFAULT_EPS = 1e-12


def _infer_head_device_and_dtype(
    lm_head: torch.nn.Module,
) -> tuple[torch.device | None, torch.dtype | None]:
    
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
    """KL(B || A) per (layer, token) via top-K + other-mass"""
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
            logits_a = lm_head(h_a).float() 
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
