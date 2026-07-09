"""POPE (Li et al., EMNLP 2023) answer normalisation and binary metrics:
accuracy, precision, recall, F1 and yes-ratio, with "yes" as the positive class.
Shared by scripts/pope_generate.py and scripts/pope_report.py so the two can
never disagree on what a model answer means.
"""

from __future__ import annotations

import re

__all__ = [
    "ANSWER_INVALID",
    "ANSWER_NO",
    "ANSWER_YES",
    "compute_pope_metrics",
    "normalize_answer",
]


ANSWER_YES = "yes"
ANSWER_NO = "no"
ANSWER_INVALID = "invalid"

_FIRST_WORD = re.compile(r"[A-Za-z]+")


def normalize_answer(text: str | None) -> str:
    """Map a generated answer to ``"yes"``, ``"no"`` or ``"invalid"``.

    The rule is the first alphabetic word, lowercased. Anything that is not
    exactly ``yes`` or ``no`` is ``invalid``.

    This is stricter than the official POPE evaluator, which folds any answer
    containing "no"/"not" into "no" and *everything else* into "yes" -- so a
    refusal or a degenerate loop silently scores as a "yes". Here those land in
    ``invalid`` and are reported apart, never counted as a prediction. That
    matters for us precisely because SPARC's failure mode is degeneration.
    """
    match = _FIRST_WORD.search(text or "")
    if match is None:
        return ANSWER_INVALID
    word = match.group(0).lower()
    if word in (ANSWER_YES, ANSWER_NO):
        return word
    return ANSWER_INVALID


def compute_pope_metrics(records: list[dict]) -> dict:
    """Aggregate ``{"expected": yes|no, "answer": yes|no|invalid}`` records.

    "yes" is the positive class, as in the POPE paper: a true positive is the
    model saying an object is present when it is.

        accuracy  = (TP + TN) / n_valid
        precision = TP / (TP + FP)
        recall    = TP / (TP + FN)
        f1        = harmonic mean of precision and recall
        yes_ratio = (TP + FP) / n_valid      (the model's answer bias)

    Invalid answers are excluded from the confusion matrix and surfaced as
    ``n_invalid`` / ``pct_invalid``. Denominators here are ``n_valid``, so a run
    with many invalid answers shows a healthy accuracy next to a loud
    ``pct_invalid``; read the two together.

    Zero-handling mirrors :func:`metrics.chair.compute_chair_aggregate`:
    no valid answers -> every rate is NaN; a zero denominator inside an
    otherwise non-empty set -> 0.0.
    """
    n_total = len(records)
    valid = [r for r in records if r["answer"] in (ANSWER_YES, ANSWER_NO)]
    n_valid = len(valid)
    n_invalid = n_total - n_valid
    pct_invalid = (100.0 * n_invalid / n_total) if n_total else float("nan")

    if n_valid == 0:
        return {
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "yes_ratio": float("nan"),
            "n_total": n_total,
            "n_valid": 0,
            "n_invalid": n_invalid,
            "pct_invalid": pct_invalid,
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
        }

    tp = sum(1 for r in valid if r["expected"] == ANSWER_YES and r["answer"] == ANSWER_YES)
    fp = sum(1 for r in valid if r["expected"] == ANSWER_NO and r["answer"] == ANSWER_YES)
    tn = sum(1 for r in valid if r["expected"] == ANSWER_NO and r["answer"] == ANSWER_NO)
    fn = sum(1 for r in valid if r["expected"] == ANSWER_YES and r["answer"] == ANSWER_NO)

    accuracy = (tp + tn) / n_valid
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    yes_ratio = (tp + fp) / n_valid

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "yes_ratio": yes_ratio,
        "n_total": n_total,
        "n_valid": n_valid,
        "n_invalid": n_invalid,
        "pct_invalid": pct_invalid,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }
