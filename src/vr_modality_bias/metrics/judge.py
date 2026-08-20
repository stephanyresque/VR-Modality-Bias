from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass

from vr_modality_bias.data.prompts import get_prompt
from vr_modality_bias.metrics.report import classify_degeneration

__all__ = [
    "ALL_LABELS",
    "JUDGE_PROMPT_KEY",
    "VERDICTS",
    "VERDICT_CORRECT",
    "VERDICT_INCORRECT",
    "VERDICT_INVALID",
    "VERDICT_NOT_ADDRESSED",
    "JudgeVerdict",
    "build_judge_prompt",
    "compute_judge_aggregate",
    "group_by_arm",
    "parse_verdict",
]

VERDICT_CORRECT = "correct"
VERDICT_INCORRECT = "incorrect"
VERDICT_NOT_ADDRESSED = "not_addressed"
VERDICT_INVALID = "invalid"

# The three are reported separately and always. Dropping not_addressed, or
# folding it into incorrect, would make an arm that answers less look better,
# which is the failure mode this whole track exists to detect. invalid is a
# fourth, disjoint label: a judge that would not answer must never become a
# silent correct.
VERDICTS: tuple[str, ...] = (
    VERDICT_CORRECT,
    VERDICT_INCORRECT,
    VERDICT_NOT_ADDRESSED,
)
ALL_LABELS: tuple[str, ...] = VERDICTS + (VERDICT_INVALID,)

JUDGE_PROMPT_KEY = "judge_composed"

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN = re.compile(r"<think>.*\Z", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```(?:json)?", re.IGNORECASE)


@dataclass(frozen=True)
class JudgeVerdict:
    verdict: str
    evidence: str = ""
    raw: str = ""
    reason: str = ""

    def __post_init__(self) -> None:
        if self.verdict not in ALL_LABELS:
            raise ValueError(
                f"Unknown verdict {self.verdict!r}. Valid: {list(ALL_LABELS)}."
            )


def build_judge_prompt(
    *,
    composed_question: str,
    sub_question: str,
    reference_answer,
    generated_answer: str,
) -> str:
    return get_prompt(JUDGE_PROMPT_KEY).format(
        composed_question=str(composed_question).strip(),
        sub_question=str(sub_question).strip(),
        reference_answer=str(reference_answer).strip(),
        generated_answer=str(generated_answer).strip(),
    )


def _strip_scaffolding(text: str) -> str:
    stripped = _THINK_BLOCK.sub(" ", text)
    stripped = _THINK_OPEN.sub(" ", stripped)
    return _FENCE.sub(" ", stripped).strip()


def _extract_object(text: str) -> dict | None:
    for candidate in _json_candidates(text):
        try:
            loaded = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(loaded, dict):
            return loaded
    return None


def _json_candidates(text: str):
    yield text
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last > first:
        yield text[first:last + 1]
        nested = text.find("}", first)
        if nested != -1 and nested != last:
            yield text[first:nested + 1]


def parse_verdict(text) -> JudgeVerdict:
    raw = "" if text is None else str(text)
    cleaned = _strip_scaffolding(raw)
    if not cleaned:
        return JudgeVerdict(VERDICT_INVALID, raw=raw, reason="empty response")

    payload = _extract_object(cleaned)
    if payload is None:
        return JudgeVerdict(VERDICT_INVALID, raw=raw, reason="no JSON object found")

    if "verdict" not in payload:
        return JudgeVerdict(VERDICT_INVALID, raw=raw, reason="no 'verdict' key")

    label = payload["verdict"]
    if not isinstance(label, str) or label.strip().lower() not in VERDICTS:
        return JudgeVerdict(
            VERDICT_INVALID, raw=raw, reason=f"verdict not one of {list(VERDICTS)}"
        )

    evidence = payload.get("evidence", "")
    return JudgeVerdict(
        verdict=label.strip().lower(),
        evidence="" if evidence is None else str(evidence),
        raw=raw,
    )


def group_by_arm(items: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        groups[item["condition_label"]].append(item)
    return dict(groups)


def _empty_counts() -> dict[str, int]:
    return {label: 0 for label in ALL_LABELS}


def _rates(counts: dict[str, int]) -> dict[str, float | int]:
    # The denominator carries invalid too, so the four rates sum to one and a
    # judge that fails often cannot inflate rate_correct by shrinking the base.
    total = sum(counts.values())
    out: dict[str, float | int] = {"n_subquestions": total}
    for label in ALL_LABELS:
        out[f"n_{label}"] = counts[label]
        out[f"rate_{label}"] = (counts[label] / total) if total else float("nan")
    return out


def compute_judge_aggregate(items: list[dict]) -> dict:
    by_type: dict[str, dict[str, int]] = {}
    overall = _empty_counts()
    n_items = len(items)
    n_all_correct = 0
    n_degenerate = 0
    total_words = 0

    for item in items:
        verdicts = item["verdicts"]
        for verdict in verdicts:
            label = verdict["verdict"]
            if label not in ALL_LABELS:
                raise ValueError(
                    f"Unknown verdict {label!r} in the aggregate input. "
                    f"Valid: {list(ALL_LABELS)}."
                )
            counts = by_type.setdefault(verdict["component_type"], _empty_counts())
            counts[label] += 1
            overall[label] += 1

        if verdicts and all(v["verdict"] == VERDICT_CORRECT for v in verdicts):
            n_all_correct += 1

        answer = str(item.get("answer", ""))
        if classify_degeneration(answer)[0]:
            n_degenerate += 1
        total_words += len(answer.split())

    return {
        "n_items": n_items,
        "n_all_correct": n_all_correct,
        "rate_all_correct": (n_all_correct / n_items) if n_items else float("nan"),
        "mean_answer_words": (total_words / n_items) if n_items else float("nan"),
        "n_degenerate": n_degenerate,
        "rate_degenerate": (n_degenerate / n_items) if n_items else float("nan"),
        "overall": _rates(overall),
        "by_type": {name: _rates(counts) for name, counts in sorted(by_type.items())},
    }
