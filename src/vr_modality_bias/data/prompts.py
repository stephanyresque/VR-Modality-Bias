"""Prompt registry.

Only ``caption_short`` is exercised by the baseline (EXPERIMENT.md §4.7).
The remaining keys are registered but left inactive (``None``) for the full
Stage 2 experiment — see EXPERIMENT.md §9 (non-objectives) and §4.7.
"""

from __future__ import annotations

__all__ = ["PROMPTS", "get_prompt"]

CAPTION_SHORT: str = (
    "Describe the image in one single sentence. Be objective and concise. "
    "Mention only the main subject and the most important context. Do not "
    "add extra details, opinions, or multiple sentences. Output exactly one "
    "sentence."
)

# Inactive in the baseline. Kept here so the prompt key set is stable for
# downstream code; populate before enabling the corresponding task.
CAPTION_MEDIUM: str | None = None  # TODO: deferred (Stage 2 full)
CAPTION_LONG: str | None = None  # TODO: deferred (Stage 2 full)
VQA_COUNT: str | None = None  # TODO: deferred (Stage 2 full)
VQA_SPATIAL: str | None = None  # TODO: deferred (Stage 2 full)
VQA_RECOGNITION: str | None = None  # TODO: deferred (Stage 2 full)


PROMPTS: dict[str, str | None] = {
    "caption_short": CAPTION_SHORT,
    "caption_medium": CAPTION_MEDIUM,
    "caption_long": CAPTION_LONG,
    "vqa_count": VQA_COUNT,
    "vqa_spatial": VQA_SPATIAL,
    "vqa_recognition": VQA_RECOGNITION,
}


def get_prompt(key: str) -> str:
    """Return the prompt text for ``key`` or raise if missing/inactive."""
    if key not in PROMPTS:
        raise KeyError(
            f"Unknown prompt key {key!r}. Known keys: {sorted(PROMPTS)}"
        )
    value = PROMPTS[key]
    if value is None:
        raise NotImplementedError(
            f"Prompt {key!r} is registered but inactive in the baseline. "
            "See EXPERIMENT.md §9."
        )
    return value
