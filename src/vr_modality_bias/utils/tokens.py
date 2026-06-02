"""Per-token decoding utilities."""

from __future__ import annotations

from typing import Any

import torch

__all__ = ["decode_caption_tokens"]


def decode_caption_tokens(
    model: Any,
    input_ids: torch.Tensor,
    caption_start: int,
    *,
    skip_special_tokens: bool = False,
) -> list[str]:
    
    processor = getattr(model, "_processor", None)
    if processor is None:
        raise RuntimeError(
            "model has no _processor — load the model first or use a wrapper "
            "that exposes a HuggingFace processor."
        )
    tokenizer = getattr(processor, "tokenizer", None) or processor
    if not hasattr(tokenizer, "decode"):
        raise RuntimeError(
            f"tokenizer {type(tokenizer).__name__} has no .decode method."
        )

    if input_ids.dim() != 1:
        raise ValueError(
            f"input_ids must be 1-D; got shape {tuple(input_ids.shape)}."
        )
    if caption_start < 0 or caption_start > input_ids.shape[0]:
        raise ValueError(
            f"caption_start={caption_start} out of range for "
            f"input_ids of length {input_ids.shape[0]}."
        )

    out: list[str] = []
    for token_id in input_ids[caption_start:].tolist():
        out.append(
            tokenizer.decode([int(token_id)], skip_special_tokens=skip_special_tokens)
        )
    return out
