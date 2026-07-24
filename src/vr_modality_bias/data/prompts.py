"""Prompt registry."""

from __future__ import annotations

__all__ = ["PROMPTS", "get_prompt"]

CAPTION_SHORT: str = (
    "Describe the image in one single sentence. Be objective and concise. "
    "Mention only the main subject and the most important context. Do not "
    "add extra details, opinions, or multiple sentences. Output exactly one "
    "sentence."
)

CAPTION_MEDIUM: str = (
    "Describe the image in three to five sentences. Be objective and "
    "specific. Mention the main subject, relevant objects, setting, and "
    "visible actions or relationships. Do not add opinions or information "
    "that cannot be inferred from the image."
)

CAPTION_LONG: str = (
    "Describe the image in a long, detailed paragraph. Be thorough and "
    "cover the main subject, the setting, all visible objects, actions, "
    "spatial relationships, colors, and contextual details. Aim for a "
    "rich, complete description of the scene."
)


PROMPTS: dict[str, str] = {
    "caption_short": CAPTION_SHORT,
    "caption_medium": CAPTION_MEDIUM,
    "caption_long": CAPTION_LONG,
}


def get_prompt(key: str) -> str:
    """Return the prompt text for ``key`` or raise if missing."""
    if key not in PROMPTS:
        raise KeyError(
            f"Unknown prompt key {key!r}. Known keys: {sorted(PROMPTS)}"
        )
    return PROMPTS[key]
