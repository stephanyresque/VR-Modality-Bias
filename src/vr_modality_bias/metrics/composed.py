from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vr_modality_bias.data.vocabulary import Vocabulary

__all__ = [
    "VERDICTS",
    "VERDICT_CORRECT",
    "VERDICT_INCORRECT",
    "VERDICT_INDETERMINATE",
    "ComponentVerdict",
    "compute_composed_aggregate",
    "verify_component",
    "verify_count",
    "verify_direction",
    "verify_existence",
]

VERDICT_CORRECT = "correct"
VERDICT_INCORRECT = "incorrect"
VERDICT_INDETERMINATE = "indeterminate"
VERDICTS: tuple[str, ...] = (
    VERDICT_CORRECT,
    VERDICT_INCORRECT,
    VERDICT_INDETERMINATE,
)

_NEGATION_CUES: frozenset[str] = frozenset({
    "no", "not", "never", "none", "nothing", "neither", "nor", "without",
    "absent", "lack", "lacks", "lacking", "missing", "cannot", "unable",
    "isn't", "aren't", "wasn't", "weren't", "don't", "doesn't", "didn't",
    "can't", "couldn't", "won't", "wouldn't", "hasn't", "haven't",
})

_AFFIRMATION_ONLY_CUES: frozenset[str] = frozenset({
    "no", "none", "nothing", "neither", "without", "absent", "missing",
})

_NUMBER_WORDS: dict[str, int] = {
    "zero": 0, "none": 0, "no": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
}

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_CLAUSE_BOUNDARY = re.compile(
    r"[,;:]|\bbut\b|\bhowever\b|\balthough\b|\bthough\b|\bwhereas\b|\bwhile\b"
)
_DIGITS = re.compile(r"\b\d+\b")
_WORD = re.compile(r"[a-z0-9']+(?:-[a-z0-9']+)*")


@dataclass(frozen=True)
class ComponentVerdict:
    verdict: str
    evidence: str

    def __post_init__(self) -> None:
        if self.verdict not in VERDICTS:
            raise ValueError(
                f"Unknown verdict {self.verdict!r}. Valid: {list(VERDICTS)}."
            )


def _padded(text: str) -> str:
    lowered = re.sub(r"[^\w\s'-]", " ", text.lower())
    return f" {re.sub(r'\s+', ' ', lowered).strip()} "


def _split_clauses(answer: str) -> list[str]:
    clauses: list[str] = []
    for sentence in _SENTENCE_BOUNDARY.split(answer):
        for clause in _CLAUSE_BOUNDARY.split(sentence):
            stripped = clause.strip()
            if stripped:
                clauses.append(stripped)
    return clauses


def _target_forms(target: str) -> tuple[str, ...]:
    base = target.strip().lower()
    forms = {base}
    if base.endswith("s"):
        forms.add(base[:-1])
        if base.endswith("es"):
            forms.add(base[:-2])
    else:
        forms.add(base + "s")
        if base.endswith(("s", "x", "z", "ch", "sh")):
            forms.add(base + "es")
        if base.endswith("y") and len(base) > 1 and base[-2] not in "aeiou":
            forms.add(base[:-1] + "ies")
    return tuple(sorted(forms))


def _mentions_target(clause: str, forms: tuple[str, ...]) -> bool:
    padded = _padded(clause)
    return any(f" {form} " in padded for form in forms)


def _clauses_mentioning(answer: str, target: str) -> list[str]:
    forms = _target_forms(target)
    return [c for c in _split_clauses(answer) if _mentions_target(c, forms)]


def _sentences_mentioning(answer: str, target: str) -> list[str]:
    forms = _target_forms(target)
    sentences = [s.strip() for s in _SENTENCE_BOUNDARY.split(answer) if s.strip()]
    return [s for s in sentences if _mentions_target(s, forms)]


def _is_negated(clause: str) -> bool:
    words = _WORD.findall(clause.lower())
    return any(word in _NEGATION_CUES for word in words)


def verify_existence(answer: str, target: str, expected) -> ComponentVerdict:
    wanted = str(expected).strip().lower()
    if wanted not in {"yes", "no", "true", "false"}:
        raise ValueError(
            f"existence component expects a yes/no annotation, got {expected!r} "
            f"for target {target!r}."
        )
    wanted_present = wanted in {"yes", "true"}

    clauses = _clauses_mentioning(answer, target)
    if not clauses:
        standalone = _standalone_polarity(answer)
        if standalone is None:
            return ComponentVerdict(VERDICT_INDETERMINATE, "")
        found_present, evidence = standalone
        return ComponentVerdict(
            VERDICT_CORRECT if found_present == wanted_present else VERDICT_INCORRECT,
            evidence,
        )

    polarities = {not _is_negated(c) for c in clauses}
    if len(polarities) != 1:
        return ComponentVerdict(VERDICT_INDETERMINATE, " | ".join(clauses))

    found_present = polarities.pop()
    evidence = next(
        (c for c in clauses if (not _is_negated(c)) == found_present), clauses[0]
    )
    return ComponentVerdict(
        VERDICT_CORRECT if found_present == wanted_present else VERDICT_INCORRECT,
        evidence,
    )


def _standalone_polarity(answer: str) -> tuple[bool, str] | None:
    clauses = _split_clauses(answer)
    if not clauses:
        return None
    first = clauses[0]
    words = _WORD.findall(first.lower())
    if not words:
        return None
    if words[0] == "yes":
        return True, first
    if words[0] in _AFFIRMATION_ONLY_CUES:
        return False, first
    return None


def _numbers_in(clause: str) -> list[int]:
    found: list[int] = []
    lowered = clause.lower()
    for match in _DIGITS.finditer(lowered):
        found.append(int(match.group(0)))
    for word in _WORD.findall(lowered):
        if word in _NUMBER_WORDS:
            found.append(_NUMBER_WORDS[word])
    return found


def verify_count(answer: str, target: str, expected) -> ComponentVerdict:
    try:
        wanted = int(str(expected).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"count component expects a numeric annotation, got {expected!r} "
            f"for target {target!r}."
        ) from exc

    clauses = _clauses_mentioning(answer, target)
    if not clauses:
        return ComponentVerdict(VERDICT_INDETERMINATE, "")

    candidates: dict[int, str] = {}
    for clause in clauses:
        for value in _numbers_in(clause):
            candidates.setdefault(value, clause)

    if len(candidates) != 1:
        return ComponentVerdict(VERDICT_INDETERMINATE, " | ".join(clauses))

    found, evidence = next(iter(candidates.items()))
    return ComponentVerdict(
        VERDICT_CORRECT if found == wanted else VERDICT_INCORRECT, evidence
    )


def _directions_in(clause: str, directions: Vocabulary) -> set[str]:
    padded = _padded(clause)
    found: set[str] = set()
    for term, canonical in directions.synonym_to_category.items():
        if f" {term} " in padded:
            found.add(canonical)
    return found


def verify_direction(
    answer: str, target: str, expected, directions: Vocabulary
) -> ComponentVerdict:
    wanted_term = str(expected).strip().lower()
    wanted = directions.synonym_to_category.get(wanted_term)
    if wanted is None:
        raise ValueError(
            f"direction component annotates {expected!r} for target {target!r}, "
            f"which is not in the direction table {directions.name!r}. Known: "
            f"{list(directions.categories)}."
        )

    # Sentence scope, not clause scope: an orientation phrase routinely hangs
    # off its noun across a comma ("three chairs, to the left of the table"),
    # unlike a quantity, which sits against it. Widening can only turn a missed
    # match into a found one or into a second candidate, and a second candidate
    # resolves to indeterminate rather than to a guess.
    clauses = _sentences_mentioning(answer, target)
    if not clauses:
        return ComponentVerdict(VERDICT_INDETERMINATE, "")

    found: dict[str, str] = {}
    for clause in clauses:
        for canonical in _directions_in(clause, directions):
            found.setdefault(canonical, clause)

    if len(found) != 1:
        return ComponentVerdict(VERDICT_INDETERMINATE, " | ".join(clauses))

    canonical, evidence = next(iter(found.items()))
    return ComponentVerdict(
        VERDICT_CORRECT if canonical == wanted else VERDICT_INCORRECT, evidence
    )


def verify_component(
    answer: str, component, directions: Vocabulary
) -> ComponentVerdict:
    if component.component_type == "existence":
        return verify_existence(answer, component.target, component.answer)
    if component.component_type == "count":
        return verify_count(answer, component.target, component.answer)
    if component.component_type == "direction":
        return verify_direction(answer, component.target, component.answer, directions)
    raise ValueError(f"Unknown component_type {component.component_type!r}.")


def _empty_counts() -> dict[str, int]:
    return {verdict: 0 for verdict in VERDICTS}


def _rates(counts: dict[str, int]) -> dict[str, float | int]:
    total = sum(counts.values())
    out: dict[str, float | int] = {"n_components": total}
    for verdict in VERDICTS:
        out[f"n_{verdict}"] = counts[verdict]
        out[f"rate_{verdict}"] = (counts[verdict] / total) if total else float("nan")
    return out


def compute_composed_aggregate(items: list[dict]) -> dict:
    by_type: dict[str, dict[str, int]] = {}
    overall = _empty_counts()
    n_items = len(items)
    n_all_correct = 0
    n_degenerate = 0
    total_words = 0

    for item in items:
        verdicts = [c["verdict"] for c in item["components"]]
        for component in item["components"]:
            counts = by_type.setdefault(component["component_type"], _empty_counts())
            counts[component["verdict"]] += 1
            overall[component["verdict"]] += 1
        if verdicts and all(v == VERDICT_CORRECT for v in verdicts):
            n_all_correct += 1
        if item.get("is_degenerate"):
            n_degenerate += 1
        total_words += len(str(item.get("answer", "")).split())

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
