#!/usr/bin/env python
"""Composed-answer report: verifies each annotated component inside the free-text
answer, aggregates per arm, and prints an auditable sample.

Verification is deterministic and written by us. There is no judge model, no API
call and no checkpoint anywhere in this path. Every component gets one of three
verdicts -- correct, incorrect, indeterminate -- and the three are reported side
by side, never folded into two: an arm that "wins" by staying silent would look
like an improvement under a two-verdict report.

Run: python scripts/composed_report.py --run-dir results/runs/composed \
         --questions <questions.jsonl> --direction-terms configs/direction_terms.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from pyprojroot import here

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chair_report import (  # noqa: E402
    _alpha_from_entries,
    _condition_label,
    _condition_sort_key,
    _derive_model_id,
    _print_table,
    _section,
    classify_degeneration,
)

try:
    from vr_modality_bias.data.annotations import read_question_annotations
    from vr_modality_bias.data.vocabulary import load_vocabulary
    from vr_modality_bias.metrics.composed import (
        VERDICT_CORRECT,
        VERDICT_INCORRECT,
        VERDICTS,
        compute_composed_aggregate,
        verify_component,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(here()))

    from src.vr_modality_bias.data.annotations import read_question_annotations
    from src.vr_modality_bias.data.vocabulary import load_vocabulary
    from src.vr_modality_bias.metrics.composed import (
        VERDICT_CORRECT,
        VERDICT_INCORRECT,
        VERDICTS,
        compute_composed_aggregate,
        verify_component,
    )


_SAMPLE_IDS = 5


def load_answers(jsonl_path: Path) -> list[dict]:
    entries: list[dict] = []
    if not jsonl_path.exists():
        return entries
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def index_questions(path: Path) -> dict[tuple[str, str], object]:
    return {
        (record.image_id, record.question_id): record
        for record in read_question_annotations(path)
    }


def assert_answers_are_grounded(
    entries: list[dict], questions: dict[tuple[str, str], object]
) -> None:
    ungrounded = [
        e for e in entries
        if (str(e.get("image_id")), str(e.get("question_id"))) not in questions
    ]
    if not ungrounded:
        return
    answer_keys = sorted(
        {(str(e.get("image_id")), str(e.get("question_id"))) for e in ungrounded}
    )
    question_keys = sorted(questions)
    raise ValueError(
        f"{len(ungrounded)} answer(s) spanning {len(answer_keys)} "
        f"(image_id, question_id) pair(s) have no annotated question "
        f"({len(question_keys)} pair(s) available). Compare the two id formats:\n"
        f"  answer pair(s) with no question: {answer_keys[:_SAMPLE_IDS]}\n"
        f"  annotated question pair(s)     : {question_keys[:_SAMPLE_IDS]}"
    )


def verify_entries(
    entries: list[dict],
    questions: dict[tuple[str, str], object],
    directions,
) -> list[dict]:
    verified: list[dict] = []
    for entry in entries:
        key = (str(entry["image_id"]), str(entry["question_id"]))
        question = questions[key]
        answer = str(entry.get("answer", ""))
        is_degenerate, degeneration_reason = classify_degeneration(answer)

        components: list[dict] = []
        for component in question.components:
            outcome = verify_component(answer, component, directions)
            verdict = outcome.verdict
            evidence = outcome.evidence
            if is_degenerate and verdict == VERDICT_CORRECT:
                # A degenerate answer can hit the right token by accident. That
                # is a failure of the model, not an inability of ours to decide,
                # so it is incorrect rather than indeterminate.
                verdict = VERDICT_INCORRECT
                evidence = f"[degenerate: {degeneration_reason}] {evidence}"
            components.append({
                "component_type": component.component_type,
                "target": component.target,
                "expected": component.answer,
                "verdict": verdict,
                "evidence": evidence,
            })

        verified.append({
            "image_id": entry["image_id"],
            "question_id": entry["question_id"],
            "question_text": question.question_text,
            "condition_label": _condition_label(entry),
            "answer": answer,
            "is_degenerate": is_degenerate,
            "degeneration_reason": degeneration_reason,
            "components": components,
            "entry": entry,
        })
    return verified


def group_by_arm(verified: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in verified:
        groups[item["condition_label"]].append(item)
    return groups


def report_by_component_type(groups: dict[str, list[dict]]) -> None:
    _section("1. VERDICTS BY COMPONENT TYPE -- per arm")
    print("  The unit is the COMPONENT, not the item: items carry different")
    print("  numbers of components, so averaging per item would over-weight the")
    print("  short ones.")
    print("  %indet is not a rounding artefact -- read it next to %correct. An arm")
    print("  that answers less can raise %correct while raising %indet too.")
    print()
    headers = ["arm", "component", "n", "%correct", "%incorrect", "%indet"]
    aligns = ["<", "<", ">", ">", ">", ">"]

    rows: list[list] = []
    for label in sorted(groups, key=_condition_sort_key):
        agg = compute_composed_aggregate(groups[label])
        for component_type, stats in agg["by_type"].items():
            rows.append([
                label, component_type, stats["n_components"],
                f"{100 * stats['rate_correct']:.1f}%",
                f"{100 * stats['rate_incorrect']:.1f}%",
                f"{100 * stats['rate_indeterminate']:.1f}%",
            ])
    _print_table(headers, rows, aligns)


def report_overall(groups: dict[str, list[dict]]) -> None:
    _section("2. OVERALL -- per arm, all components pooled")
    print("  all_correct = fraction of ITEMS whose every component is correct.")
    print("  words       = mean answer length; this is what shows whether the")
    print("                composed question produced a naturally long answer.")
    print()
    headers = [
        "arm", "items", "components", "%correct", "%incorrect", "%indet",
        "all_correct", "words", "%degen",
    ]
    aligns = ["<"] + [">"] * 8

    rows: list[list] = []
    for label in sorted(groups, key=_condition_sort_key):
        agg = compute_composed_aggregate(groups[label])
        overall = agg["overall"]
        rows.append([
            label, agg["n_items"], overall["n_components"],
            f"{100 * overall['rate_correct']:.1f}%",
            f"{100 * overall['rate_incorrect']:.1f}%",
            f"{100 * overall['rate_indeterminate']:.1f}%",
            f"{100 * agg['rate_all_correct']:.1f}%",
            f"{agg['mean_answer_words']:.1f}",
            f"{100 * agg['rate_degenerate']:.1f}%",
        ])
    _print_table(headers, rows, aligns)


def report_audit_sample(
    verified: list[dict], *, n_samples: int, image_ids: list[str] | None
) -> None:
    _section(f"3. AUDIT SAMPLE -- {n_samples} item(s) with the evidence per verdict")
    print("  Text matching is fragile and the only way to know whether it works")
    print("  is to read it. Without this section there is no way to tell 'the")
    print("  method got worse' from 'the verifier is wrong'.")
    print()

    if image_ids:
        chosen = [i for i in verified if str(i["image_id"]) in set(image_ids)]
    else:
        chosen = verified
    if not chosen:
        print("  (no items to sample)")
        return

    # Prefer items that exercise the verifier: anything not fully correct comes
    # first, so the sample shows the decisions worth checking.
    def _interest(item: dict) -> tuple[int, str, str]:
        verdicts = [c["verdict"] for c in item["components"]]
        clean = all(v == VERDICT_CORRECT for v in verdicts)
        return (1 if clean else 0, str(item["image_id"]), str(item["question_id"]))

    for item in sorted(chosen, key=_interest)[:n_samples]:
        print(f"  ── {item['image_id']} / {item['question_id']}  [{item['condition_label']}]")
        print(f"     Q: {item['question_text']}")
        print(f"     A: {item['answer']}")
        if item["is_degenerate"]:
            print(f"     ! degenerate ({item['degeneration_reason']}) "
                  f"-- no component may count as correct")
        for component in item["components"]:
            print(
                f"       [{component['verdict']:<13}] "
                f"{component['component_type']:<9} "
                f"target={component['target']!r} expected={component['expected']!r}"
            )
            evidence = component["evidence"] or "(nothing in the answer matched)"
            print(f"                       evidence: {evidence}")
        print()


def collect_rows(groups: dict[str, list[dict]], *, model_id: str) -> list[dict]:
    rows: list[dict] = []
    for label in sorted(groups, key=_condition_sort_key):
        items = groups[label]
        agg = compute_composed_aggregate(items)
        overall = agg["overall"]
        row = {
            "model_id": model_id,
            "condition": "off" if label == "off" else "on",
            "condition_label": label,
            "alpha": _alpha_from_entries([i["entry"] for i in items]),
            "n_items": agg["n_items"],
            "n_components": overall["n_components"],
            "rate_correct": overall["rate_correct"],
            "rate_incorrect": overall["rate_incorrect"],
            "rate_indeterminate": overall["rate_indeterminate"],
            "n_all_correct": agg["n_all_correct"],
            "rate_all_correct": agg["rate_all_correct"],
            "mean_answer_words": agg["mean_answer_words"],
            "n_degenerate": agg["n_degenerate"],
            "rate_degenerate": agg["rate_degenerate"],
        }
        for component_type, stats in agg["by_type"].items():
            row[f"n_components_{component_type}"] = stats["n_components"]
            for verdict in VERDICTS:
                row[f"rate_{verdict}_{component_type}"] = stats[f"rate_{verdict}"]
        rows.append(row)
    return rows


def write_composed_results(rows: list[dict], run_dir: Path) -> tuple[Path, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    json_path = run_dir / "composed_results.json"
    csv_path = run_dir / "composed_results.csv"

    json_path.write_text(
        json.dumps(
            {
                "generated_iso": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
                "n_rows": len(rows),
                "rows": rows,
            },
            indent=2, ensure_ascii=False, default=str,
        ) + "\n",
        encoding="utf-8",
    )

    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row[k]) for k in columns})

    return json_path, csv_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
        help="Directory containing answers.jsonl (= <output-root>/<run-name>).")
    parser.add_argument("--questions", type=Path, required=True,
        help="JSON Lines question-annotation file holding the components.")
    parser.add_argument("--direction-terms", type=Path, required=True,
        help="JSON table of equivalent orientation terms (same shape as a "
             "vocabulary file: name, categories, synonyms). No default: the "
             "table is meant to be tuned against the real data, so which one "
             "was used is part of the result.")
    parser.add_argument("--sample", type=int, default=5,
        help="How many items to print in the audit section.")
    parser.add_argument("--sample-image-ids", type=str, nargs="*", default=None,
        help="Restrict the audit sample to these image ids.")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    answers_path = args.run_dir / "answers.jsonl"
    for path, what in (
        (args.run_dir, "run-dir"),
        (answers_path, "answers.jsonl"),
        (args.questions, "questions"),
        (args.direction_terms, "direction-terms"),
    ):
        if not path.exists():
            print(f"ERROR: {what} {path} not found.", file=sys.stderr)
            return 1

    directions = load_vocabulary(args.direction_terms)
    entries = load_answers(answers_path)
    questions = index_questions(args.questions)

    print("=" * 78)
    print("COMPOSED-ANSWER REPORT")
    print("=" * 78)
    print(f"  generated       : {datetime.now(tz=timezone.utc).isoformat(timespec='seconds')}")
    print(f"  run_dir         : {args.run_dir}")
    print(f"  answers         : {answers_path}")
    print(f"  questions       : {args.questions}")
    print(f"  direction terms : {args.direction_terms} "
          f"({directions.name}, {len(directions.categories)} directions)")
    print(f"  verification    : deterministic, no judge model")
    print()
    print(f"  loaded answers  : {len(entries)}")
    print(f"  annotated pairs : {len(questions)}")

    if not entries:
        print("ERROR: answers.jsonl is empty.", file=sys.stderr)
        return 1

    try:
        assert_answers_are_grounded(entries, questions)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    answered = {(str(e["image_id"]), str(e["question_id"])) for e in entries}
    unanswered = sorted(set(questions) - answered)
    if unanswered:
        print(f"  questions with no answer yet : {len(unanswered)} "
              f"(partial or resumed run; not an error)")

    verified = verify_entries(entries, questions, directions)
    groups = group_by_arm(verified)

    report_by_component_type(groups)
    report_overall(groups)
    report_audit_sample(
        verified, n_samples=args.sample, image_ids=args.sample_image_ids,
    )

    model_id = _derive_model_id([i["entry"] for i in verified])
    rows = collect_rows(groups, model_id=model_id)
    json_path, csv_path = write_composed_results(rows, args.run_dir)
    print()
    print(f"  persisted JSON : {json_path}")
    print(f"  persisted CSV  : {csv_path}")
    print(f"  model_id       : {model_id}")

    print()
    print("=" * 78)
    print("END OF REPORT")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
