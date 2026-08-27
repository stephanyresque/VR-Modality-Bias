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


VQA_COMPOSED: str = "Answer the question about the image.\n\n{question}"


# The judge is text-only by design, not to save compute. What is under test is
# whether the equirectangular projection breaks the model being evaluated; a
# judge that looked at the same panorama would be exposed to the same defect,
# and the measurement would be circular. Never add the image to this prompt.
JUDGE_COMPOSED: str = """You are grading one sub-question of a composite question that was put to a \
vision-language model about a photograph.

You cannot see the photograph. Do not try to imagine it, and do not use any \
knowledge of your own about what such a scene usually contains. Your only job \
is to compare the generated answer against a reference answer written by a \
human annotator who did see the photograph.

Treat the reference answer as correct by definition. You are never asked \
whether the reference is right. You are asked only whether the generated \
answer is consistent with it.

FULL QUESTION PUT TO THE MODEL (context only, do not grade all of it):
{composed_question}

THE ONE SUB-QUESTION YOU ARE GRADING:
{sub_question}

HUMAN REFERENCE ANSWER TO THAT SUB-QUESTION:
{reference_answer}

THE FULL ANSWER THE MODEL GENERATED:
{generated_answer}

Choose exactly one verdict:

  "correct"        the generated answer says something about this sub-question \
that agrees with the reference.
  "incorrect"      the generated answer says something about this sub-question \
that contradicts the reference.
  "not_addressed"  the generated answer says nothing about this sub-question.

Rules:
- The generated answer responds to the whole composite question. Read all of \
it, but grade only the sub-question above.
- Silence is "not_addressed", never "incorrect". Only a claim that actually \
contradicts the reference is "incorrect".
- A hedged or partial statement that agrees with the reference is "correct". \
Wording need not match; judge the claim, not the phrasing.
- For a count, the number must match the reference to be "correct". For a \
direction, the named direction must match.
- For an angle, the numeric value must match the reference. For text read \
from the scene, the transcription must match the reference apart from case \
and punctuation. For a yes/no reference, the generated claim must take the \
same side.
- If the answer is degenerate or repetitive but still contains a claim about \
this sub-question, grade that claim normally.

Reply with one JSON object and nothing else. Do not explain your reasoning \
outside the JSON. The "evidence" field is a short quote from the generated \
answer, or an empty string when nothing addressed the sub-question:

{{"verdict": "correct", "evidence": "there are three chairs"}}"""


PROMPTS: dict[str, str] = {
    "caption_short": CAPTION_SHORT,
    "caption_medium": CAPTION_MEDIUM,
    "caption_long": CAPTION_LONG,
    "vqa_composed": VQA_COMPOSED,
    "judge_composed": JUDGE_COMPOSED,
}


def get_prompt(key: str) -> str:
    """Return the prompt text for ``key`` or raise if missing."""
    if key not in PROMPTS:
        raise KeyError(
            f"Unknown prompt key {key!r}. Known keys: {sorted(PROMPTS)}"
        )
    return PROMPTS[key]
