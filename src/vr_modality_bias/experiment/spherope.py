from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Iterator

from pyprojroot import here

try:
    from vr_modality_bias.utils.attn import decoder_of
    from vr_modality_bias.utils.spherope import SpheRoPETextRotary, SpheRoPEVisionRotary
except ModuleNotFoundError:
    sys.path.insert(0, str(here()))

    from src.vr_modality_bias.utils.attn import decoder_of
    from src.vr_modality_bias.utils.spherope import SpheRoPETextRotary, SpheRoPEVisionRotary

__all__ = ["SPHEROPE_MODES", "enable_spherope", "vision_tower_of"]

SPHEROPE_MODES: tuple[str, ...] = ("off", "vit", "llm", "both")


def vision_tower_of(model):
    inner = getattr(model, "model", model)
    for attr in ("visual", "vision_tower", "vision_model"):
        candidate = getattr(inner, attr, None)
        if candidate is not None:
            return candidate
    raise AttributeError(
        f"Could not find the vision tower on {type(model).__name__}."
    )


@contextmanager
def enable_spherope(
    model_wrapper,
    *,
    mode: str,
    image_positions,
    pad_cols_vit: int = 0,
    pad_cols_llm: int = 0,
) -> Iterator[dict]:
    if mode not in SPHEROPE_MODES:
        raise ValueError(f"Unknown spherope mode {mode!r}. Known: {SPHEROPE_MODES}.")
    handles: dict = {"mode": mode, "vision": None, "text": None}
    if mode == "off":
        yield handles
        return

    model = model_wrapper._model
    vision = vision_tower_of(model)
    decoder = decoder_of(model)
    if not hasattr(vision, "rotary_pos_emb") or not hasattr(decoder, "rotary_emb"):
        raise ValueError(
            f"{type(model).__name__} has no 2D rotary embedding to replace: SpheRoPE "
            "only applies to families whose vision tower and decoder use RoPE "
            "(Qwen2.5-VL)."
        )
    mrope_section = decoder.config.rope_parameters["mrope_section"]

    original_vision = vision.rotary_pos_emb
    original_text = decoder.rotary_emb
    try:
        if mode in ("vit", "both"):
            handles["vision"] = SpheRoPEVisionRotary(original_vision, pad_cols=pad_cols_vit)
            vision.rotary_pos_emb = handles["vision"]
        if mode in ("llm", "both"):
            handles["text"] = SpheRoPETextRotary(original_text, mrope_section)
            handles["text"].arm(image_positions, pad_cols=pad_cols_llm)
            decoder.rotary_emb = handles["text"]
        yield handles
    finally:
        vision.rotary_pos_emb = original_vision
        decoder.rotary_emb = original_text
