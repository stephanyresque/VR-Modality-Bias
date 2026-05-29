"""Cosine distance per (layer, token)"""

from __future__ import annotations

import numpy as np
import torch

__all__ = ["compute_cosine_distance_matrix"]


def compute_cosine_distance_matrix(
    hidden_states_A: torch.Tensor,
    hidden_states_B: torch.Tensor,
    caption_start: int,
    caption_len: int,
    eps: float = 1e-8,
) -> np.ndarray:
    """Cosine distance between A and B at every (layer, predictive position)"""
    if hidden_states_A.shape != hidden_states_B.shape:
        raise ValueError(
            f"hidden_states_A.shape={tuple(hidden_states_A.shape)} != "
            f"hidden_states_B.shape={tuple(hidden_states_B.shape)}"
        )
    if hidden_states_A.dim() != 3:
        raise ValueError(
            f"hidden_states must be 3-D; got {tuple(hidden_states_A.shape)}"
        )

    _, seq_len, _ = hidden_states_A.shape
    pos_start = caption_start - 1
    pos_end = caption_start + caption_len - 1
    if pos_start < 0:
        raise ValueError(
            f"caption_start={caption_start} yields negative predictive position."
        )
    if pos_end > seq_len:
        raise ValueError(
            f"caption_start + caption_len - 1 = {pos_end} exceeds seq_len={seq_len}."
        )

    h_a = hidden_states_A[:, pos_start:pos_end, :].to(torch.float32)
    h_b = hidden_states_B[:, pos_start:pos_end, :].to(torch.float32)

    dot = (h_a * h_b).sum(dim=-1)
    norm_a = torch.linalg.vector_norm(h_a, dim=-1).clamp(min=eps)
    norm_b = torch.linalg.vector_norm(h_b, dim=-1).clamp(min=eps)
    cos_sim = dot / (norm_a * norm_b)
    cos_dist = (1.0 - cos_sim).clamp(min=0.0)

    return cos_dist.detach().cpu().numpy().astype(np.float32)
