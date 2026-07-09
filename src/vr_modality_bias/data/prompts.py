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
VQA_COUNT: str = (
    "Answer the counting question using only the image. Return a concise "
    "answer with the number and, when helpful, the counted object. If the "
    "quantity is not visible, answer that it cannot be determined."
)
VQA_SPATIAL: str = (
    "Answer the spatial question using only the image. Be concise and refer "
    "to visible positions, directions, or relationships between objects. If "
    "the relationship is not visible, answer that it cannot be determined."
)
VQA_RECOGNITION: str = (
    "Answer the recognition question using only the image. Identify the "
    "visible object, person, place, attribute, or action as directly as "
    "possible. If it is not visible, answer that it cannot be determined."
)
# Unlike every other entry, this one is a FORMAT TEMPLATE, not a ready prompt:
# it carries a ``{object}`` placeholder. ``scripts/build_pope.py`` renders it
# once per question and stores the result in pope_questions.jsonl, so the
# generation script never formats it again. Wording is the POPE protocol's
# (Li et al., EMNLP 2023); do not paraphrase, the benchmark numbers depend on it.
VQA_POPE: str = "Is there a {object} in the image? Please answer yes or no."


PROMPTS: dict[str, str] = {
    "caption_short": CAPTION_SHORT,
    "caption_medium": CAPTION_MEDIUM,
    "caption_long": CAPTION_LONG,
    "vqa_count": VQA_COUNT,
    "vqa_spatial": VQA_SPATIAL,
    "vqa_recognition": VQA_RECOGNITION,
    "vqa_pope": VQA_POPE,
}


def get_prompt(key: str) -> str:
    """Return the prompt text for ``key`` or raise if missing."""
    if key not in PROMPTS:
        raise KeyError(
            f"Unknown prompt key {key!r}. Known keys: {sorted(PROMPTS)}"
        )
    return PROMPTS[key]
