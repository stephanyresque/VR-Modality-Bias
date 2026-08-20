from __future__ import annotations

import re
import sys

__all__ = [
    "BIGRAM_MIN_COUNT",
    "BIGRAM_MIN_WORDS",
    "BIGRAM_SHARE",
    "CONSECUTIVE_REPEAT_TRIGGER",
    "DEGENERATION_REASONS",
    "MIN_WORDS",
    "alpha_from_entries",
    "assert_entries_are_grounded",
    "classify_degeneration",
    "condition_label",
    "condition_sort_key",
    "derive_model_id",
    "label_from_sparc",
    "print_table",
    "section",
]

MIN_WORDS = 3
CONSECUTIVE_REPEAT_TRIGGER = 3
BIGRAM_MIN_WORDS = 6
BIGRAM_MIN_COUNT = 3
BIGRAM_SHARE = 0.30

DEGENERATION_REASONS: tuple[str, ...] = (
    "empty",
    "too_short",
    "word_repetition",
    "bigram_repetition",
)

_SAMPLE_IDS = 5
_PUNCTUATION = re.compile(r"[^\w\s'-]")


def _normalise_token(word: str) -> str:
    return _PUNCTUATION.sub("", word.lower()).strip()


def classify_degeneration(text: str) -> tuple[bool, str]:
    stripped = text.strip()
    if not stripped:
        return (True, "empty")

    words = stripped.split()
    if len(words) < MIN_WORDS:
        return (True, "too_short")

    tokens = [_normalise_token(word) for word in words]

    run = 1
    for i in range(1, len(tokens)):
        if tokens[i] and tokens[i] == tokens[i - 1]:
            run += 1
            if run >= CONSECUTIVE_REPEAT_TRIGGER:
                return (True, "word_repetition")
        else:
            run = 1

    if len(tokens) >= BIGRAM_MIN_WORDS:
        bigrams = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
        counts: dict[tuple[str, str], int] = {}
        for bigram in bigrams:
            counts[bigram] = counts.get(bigram, 0) + 1
        most = max(counts.values())
        if most >= max(BIGRAM_MIN_COUNT, int(len(bigrams) * BIGRAM_SHARE)):
            return (True, "bigram_repetition")

    return (False, "")


def _entry_key(entry: dict, id_fields: tuple[str, ...]):
    values = tuple(str(entry.get(field)) for field in id_fields)
    return values[0] if len(values) == 1 else values


def _key_noun(id_fields: tuple[str, ...]) -> str:
    if len(id_fields) == 1:
        return f"{id_fields[0]}(s)"
    return "(" + ", ".join(id_fields) + ") pair(s)"


def assert_entries_are_grounded(
    entries: list[dict],
    ground_truth,
    *,
    id_fields: tuple[str, ...],
    entry_noun: str,
) -> None:
    # Fatal, never a warning. On a dataset swap the generated ids and the
    # annotated ids can disagree -- a different zero-padding, a stem versus a
    # full file name -- and every entry is then silently skipped, so the report
    # prints empty tables without a word. That is the likeliest failure of a
    # dataset swap and the hardest to notice, so it stops the run right here.
    missing = [e for e in entries if _entry_key(e, id_fields) not in ground_truth]
    if not missing:
        return

    missing_keys = sorted({_entry_key(e, id_fields) for e in missing})
    known_keys = sorted(ground_truth)
    noun = _key_noun(id_fields)
    raise ValueError(
        f"{len(missing)} {entry_noun}(s) spanning {len(missing_keys)} {noun} "
        f"have no entry in the ground truth ({len(known_keys)} {noun} "
        f"available). Compare the two id formats:\n"
        f"  {entry_noun} {noun} with no ground truth: {missing_keys[:_SAMPLE_IDS]}\n"
        f"  ground-truth {noun}: {known_keys[:_SAMPLE_IDS]}"
    )


def label_from_sparc(sparc: dict) -> str:
    layer = f"L{int(sparc['selected_layer'])}"
    if sparc.get("conserve"):
        return (f"on adaptive+qcond+conserve rho={float(sparc['rho']):g} "
                f"s={float(sparc['sink_frac']):g} {layer}")
    if sparc.get("qcond"):
        return f"on adaptive+qcond q={float(sparc['qtop_frac']):g} {layer}"
    if sparc.get("adaptive"):
        return (f"on adaptive lam={float(sparc['lam']):g} "
                f"ceil={float(sparc['ceiling']):g} {layer}")
    return f"on sparc a={float(sparc['alpha']):g} {layer}"


def condition_label(entry: dict) -> str:
    if entry["condition"] == "off":
        return "off"
    sparc = entry.get("sparc")
    if sparc is None:
        return f"on α={float(entry.get('alpha', 0)):g}"
    return label_from_sparc(sparc)


def _num_after(label: str, token: str) -> float:
    match = re.search(re.escape(token) + r"([-+eE\d.]+)", label)
    return float(match.group(1)) if match else 0.0


def condition_sort_key(label: str) -> tuple[int, float, float]:
    if label == "off":
        return (0, 0.0, 0.0)
    layer = _num_after(label, "L")
    if label.startswith("on α="):
        return (1, _num_after(label, "α="), 0.0)
    if label.startswith("on sparc"):
        return (1, _num_after(label, "a="), layer)
    if label.startswith("on adaptive+qcond+conserve"):
        return (4, 0.0, layer)
    if label.startswith("on adaptive+qcond"):
        return (3, 0.0, layer)
    if label.startswith("on adaptive"):
        return (2, _num_after(label, "lam="), layer)
    return (5, 0.0, layer)


def alpha_from_entries(entries: list[dict]) -> float | None:
    for entry in entries:
        if entry.get("condition") == "off":
            continue
        sparc = entry.get("sparc")
        value = sparc.get("alpha") if isinstance(sparc, dict) else entry.get("alpha")
        if value is not None:
            return float(value)
    return None


def derive_model_id(entries: list[dict]) -> str:
    ids = {str(e.get("model_id")) for e in entries if e.get("model_id")}
    if not ids:
        return "unknown"
    ids_list = sorted(ids)
    if len(ids_list) > 1:
        print(
            f"  WARN: entries carry multiple model_ids: {ids_list}; "
            f"using {ids_list[0]!r}.",
            file=sys.stderr,
        )
    return ids_list[0]


def print_table(
    headers: list[str], rows: list[list], aligns: list[str] | None = None
) -> None:
    if not rows:
        print("  (no rows)")
        return
    if aligns is None:
        aligns = ["<"] * len(headers)
    str_rows = [[str(c) for c in r] for r in rows]
    widths = [
        max(len(h), *(len(r[i]) for r in str_rows))
        for i, h in enumerate(headers)
    ]
    sep = "  "
    print("  " + sep.join(f"{h:<{w}}" for h, w in zip(headers, widths)))
    print("  " + sep.join("-" * w for w in widths))
    for r in str_rows:
        print("  " + sep.join(f"{c:{a}{w}}" for c, w, a in zip(r, widths, aligns)))


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)
