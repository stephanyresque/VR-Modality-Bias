#!/usr/bin/env python
"""POPE report over a pope_generate run: accuracy / precision / recall / F1 /
yes-ratio per (condition, strategy), plus the invalid-answer rate.

Run: python scripts/pope_report.py --run-dir results/runs/pope
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

try:
    from vr_modality_bias.metrics.pope import compute_pope_metrics, normalize_answer
except ModuleNotFoundError:
    sys.path.insert(0, str(here()))

    from src.vr_modality_bias.metrics.pope import compute_pope_metrics, normalize_answer


CONDITIONS_ORDER = ("baseline", "sparc", "adaptive")
STRATEGIES_ORDER = ("random", "popular", "adversarial")
# Pseudo-strategy: the three strategies pooled. Reported alongside them because
# the POPE headline number in the literature is usually the per-strategy one.
ALL_STRATEGIES = "all"


def _load_answers(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def _print_table(headers: list[str], rows: list[list], aligns: list[str] | None = None) -> None:
    if not rows:
        print("  (no rows)")
        return
    if aligns is None:
        aligns = ["<"] * len(headers)
    str_rows = [[str(c) for c in r] for r in rows]
    widths = [max(len(h), *(len(r[i]) for r in str_rows)) for i, h in enumerate(headers)]
    sep = "  "
    print("  " + sep.join(f"{h:<{w}}" for h, w in zip(headers, widths, strict=True)))
    print("  " + sep.join("-" * w for w in widths))
    for r in str_rows:
        print("  " + sep.join(
            f"{c:{a}{w}}" for c, w, a in zip(r, widths, aligns, strict=True)
        ))


def _renormalise(entries: list[dict]) -> list[dict]:
    """Recompute ``answer`` from ``answer_raw``.

    pope_generate.py already stored a normalised answer, but re-deriving it here
    means a change to the rule can be applied to an existing run without
    regenerating, and it catches a file written by an older normaliser.
    """
    out = []
    for e in entries:
        out.append({**e, "answer": normalize_answer(e.get("answer_raw", ""))})
    return out


def collect_pope_rows(entries: list[dict], *, model_id: str) -> list[dict]:
    """One row per (condition, strategy), plus a pooled ``all`` row per condition."""
    by_cell: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for e in entries:
        by_cell[(str(e["condition"]), str(e["strategy"]))].append(e)

    conditions = sorted(
        {str(e["condition"]) for e in entries},
        key=lambda c: (CONDITIONS_ORDER.index(c) if c in CONDITIONS_ORDER else 99, c),
    )

    rows: list[dict] = []
    for condition in conditions:
        strategies = sorted(
            {s for (c, s) in by_cell if c == condition},
            key=lambda s: (STRATEGIES_ORDER.index(s) if s in STRATEGIES_ORDER else 99, s),
        )
        for strategy in [*strategies, ALL_STRATEGIES]:
            if strategy == ALL_STRATEGIES:
                cells = [e for (c, _s), v in by_cell.items() if c == condition for e in v]
            else:
                cells = by_cell[(condition, strategy)]
            if not cells:
                continue
            metrics = compute_pope_metrics(cells)
            rows.append({
                "model_id": model_id,
                "condition": condition,
                "strategy": strategy,
                **metrics,
            })
    return rows


def _fmt(value) -> str:
    if value is None:
        return "--"
    if isinstance(value, float):
        return "nan" if value != value else f"{value:.4f}"
    return str(value)


def report_metrics(rows: list[dict]) -> None:
    _section("1. POPE: accuracy / precision / recall / F1 / yes-ratio")
    print("  Positive class = 'yes' (the object IS in the image).")
    print("  yes_ratio near 0.5 means a balanced answerer; near 1.0 means the")
    print("  model says yes to everything, which is the hallucination failure mode.")
    print("  Rates are computed over VALID answers only; see the invalid table.")
    print()
    headers = ["condition", "strategy", "n_valid", "accuracy", "precision",
               "recall", "f1", "yes_ratio"]
    aligns = ["<", "<", ">", ">", ">", ">", ">", ">"]
    table = [
        [r["condition"], r["strategy"], r["n_valid"],
         _fmt(r["accuracy"]), _fmt(r["precision"]), _fmt(r["recall"]),
         _fmt(r["f1"]), _fmt(r["yes_ratio"])]
        for r in rows
    ]
    _print_table(headers, table, aligns)


def report_confusion(rows: list[dict]) -> None:
    _section("2. CONFUSION MATRIX per (condition, strategy)")
    print("  TP: said yes, object present.   FP: said yes, object absent (hallucination).")
    print("  TN: said no, object absent.     FN: said no, object present (omission).")
    print()
    headers = ["condition", "strategy", "TP", "FP", "TN", "FN"]
    aligns = ["<", "<", ">", ">", ">", ">"]
    table = [
        [r["condition"], r["strategy"], r["tp"], r["fp"], r["tn"], r["fn"]]
        for r in rows
    ]
    _print_table(headers, table, aligns)


def report_invalid(rows: list[dict]) -> None:
    _section("3. INVALID ANSWERS: excluded from the rates above, never silently")
    print("  An answer is invalid when its first alphabetic word is neither")
    print("  'yes' nor 'no'. A high rate here invalidates the row above it.")
    print()
    headers = ["condition", "strategy", "n_total", "n_valid", "n_invalid", "%invalid"]
    aligns = ["<", "<", ">", ">", ">", ">"]
    table = [
        [r["condition"], r["strategy"], r["n_total"], r["n_valid"], r["n_invalid"],
         "--" if r["pct_invalid"] != r["pct_invalid"] else f"{r['pct_invalid']:.1f}%"]
        for r in rows
    ]
    _print_table(headers, table, aligns)


def write_pope_results(rows: list[dict], run_dir: Path) -> tuple[Path, Path]:
    """Write ``pope_results.{json,csv}`` under ``run_dir``. Returns both paths."""
    run_dir.mkdir(parents=True, exist_ok=True)
    json_path = run_dir / "pope_results.json"
    csv_path = run_dir / "pope_results.csv"

    json_path.write_text(
        json.dumps({
            "generated_iso": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
            "n_rows": len(rows),
            "rows": rows,
        }, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )

    columns = (
        "model_id", "condition", "strategy",
        "n_total", "n_valid", "n_invalid", "pct_invalid",
        "accuracy", "precision", "recall", "f1", "yes_ratio",
        "tp", "fp", "tn", "fn",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row[k]) for k in columns})

    return json_path, csv_path


def _derive_model_id(entries: list[dict]) -> str:
    ids = sorted({str(e.get("model_id")) for e in entries if e.get("model_id")})
    if not ids:
        return "unknown"
    if len(ids) > 1:
        print(f"  WARN: pope_answers.jsonl carries multiple model_ids: {ids}; "
              f"using {ids[0]!r}.", file=sys.stderr)
    return ids[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
        help="Directory containing pope_answers.jsonl.")
    args = parser.parse_args()

    answers_path = args.run_dir / "pope_answers.jsonl"
    if not answers_path.exists():
        print(f"ERROR: {answers_path} not found.", file=sys.stderr)
        return 1

    entries = _load_answers(answers_path)
    if not entries:
        print(f"ERROR: {answers_path} is empty.", file=sys.stderr)
        return 1
    entries = _renormalise(entries)

    print("=" * 78)
    print("POPE REPORT")
    print("=" * 78)
    print(f"  generated : {datetime.now(tz=timezone.utc).isoformat(timespec='seconds')}")
    print(f"  run_dir   : {args.run_dir}")
    print(f"  answers   : {len(entries)}")

    model_id = _derive_model_id(entries)
    rows = collect_pope_rows(entries, model_id=model_id)

    report_metrics(rows)
    report_confusion(rows)
    report_invalid(rows)

    json_path, csv_path = write_pope_results(rows, args.run_dir)
    print()
    print(f"  persisted JSON : {json_path}")
    print(f"  persisted CSV  : {csv_path}")
    print(f"  model_id       : {model_id}")
    print(f"  rows           : {len(rows)}")
    print()
    print("=" * 78)
    print("END OF REPORT")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
