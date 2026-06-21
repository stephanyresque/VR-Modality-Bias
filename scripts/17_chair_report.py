#!/usr/bin/env python
"""Phase 3 — CHAIR report, stdout only."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from pyprojroot import here

try:
    from vr_modality_bias.data.coco_annotations import (
        DEFAULT_TARGET_DIR as DEFAULT_ANNOTATIONS_DIR,
        ensure_coco_annotations,
    )
    from vr_modality_bias.metrics.chair import (
        chair_per_caption,
        compute_chair_aggregate,
        load_ground_truth_objects,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(here()))
    from src.vr_modality_bias.data.coco_annotations import (
        DEFAULT_TARGET_DIR as DEFAULT_ANNOTATIONS_DIR,
        ensure_coco_annotations,
    )
    from src.vr_modality_bias.metrics.chair import (
        chair_per_caption,
        compute_chair_aggregate,
        load_ground_truth_objects,
    )


LENGTHS_ORDER = ("short", "medium", "long")


# ---------------------------------------------------------------- IO


def _load_captions(jsonl_path: Path) -> list[dict]:
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


# ---------------------------------------------------------------- degeneracy


def classify_degeneration(caption: str) -> tuple[bool, str]:
    """Heuristic: is the caption degenerate, and if so, why?

    Returns (is_degenerate, reason). Reasons:
        empty           — only whitespace
        too_short       — fewer than 3 word tokens
        word_repetition — a single word repeats 4+ times consecutively
                          (catches "its its its its")
        bigram_repetition — same 2-gram repeats heavily (catches "This The
                          This The" patterns)
    Non-degenerate → (False, "").
    """
    stripped = caption.strip()
    if not stripped:
        return (True, "empty")
    words = stripped.split()
    if len(words) < 3:
        return (True, "too_short")
    # Consecutive word repetition.
    max_consec = 1
    cur = 1
    for i in range(1, len(words)):
        if words[i].lower() == words[i - 1].lower():
            cur += 1
            max_consec = max(max_consec, cur)
        else:
            cur = 1
    if max_consec >= 4:
        return (True, "word_repetition")
    # Bigram repetition (catches "This The This The ...").
    if len(words) >= 6:
        bigrams = [(words[i].lower(), words[i + 1].lower())
                   for i in range(len(words) - 1)]
        counts: dict[tuple, int] = {}
        for bg in bigrams:
            counts[bg] = counts.get(bg, 0) + 1
        most = max(counts.values())
        if most >= max(3, int(len(bigrams) * 0.30)):
            return (True, "bigram_repetition")
    return (False, "")


# ---------------------------------------------------------------- print


def _print_table(headers: list[str], rows: list[list], aligns: list[str] | None = None) -> None:
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


def _section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def _condition_label(entry: dict) -> str:
    """Human-readable condition string for tables."""
    if entry["condition"] == "off":
        return "off"
    return f"on α={float(entry.get('alpha', 0)):.1f}"


def _condition_sort_key(label: str) -> tuple[int, float]:
    if label == "off":
        return (0, 0.0)
    # "on α=1.1" → extract the float
    try:
        return (1, float(label.split("=")[1]))
    except (IndexError, ValueError):
        return (1, 0.0)


# ---------------------------------------------------------------- sections


def report_chair_by_length(entries: list[dict], gt: dict[str, set[str]]) -> None:
    _section("1. CHAIR BY LENGTH — baseline vs SPARC")
    print("  CHAIR_s = fraction of captions with ≥1 hallucinated object (lower is better).")
    print("  CHAIR_i = fraction of OBJECT MENTIONS that are hallucinated (lower is better).")
    print()
    headers = ["length", "condition", "n", "CHAIR_s", "CHAIR_i", "total_mentioned", "total_hallucinated"]
    aligns =  ["<",      "<",         ">", ">",       ">",       ">",               ">"]

    # Group entries by (length, condition_label).
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for e in entries:
        # Skip captions for images with no GT annotations — they can't
        # ground the hallucination metric.
        if e["image_id"] not in gt:
            continue
        groups[(e["length"], _condition_label(e))].append(e)

    rows_out = []
    for length in LENGTHS_ORDER:
        labels_in_length = sorted(
            {key[1] for key in groups if key[0] == length},
            key=_condition_sort_key,
        )
        for label in labels_in_length:
            cells = groups.get((length, label), [])
            if not cells:
                continue
            per_caption = [
                chair_per_caption(c["caption"], gt[c["image_id"]])
                for c in cells
            ]
            agg = compute_chair_aggregate(per_caption)
            rows_out.append([
                length, label, agg["n_captions"],
                f"{agg['chair_s']:.4f}", f"{agg['chair_i']:.4f}",
                agg["total_mentioned"], agg["total_hallucinated"],
            ])
    _print_table(headers, rows_out, aligns)


def report_degeneration(entries: list[dict]) -> None:
    _section("2. DEGENERATION RATE — captions empty / too short / repetitive")
    print("  Per (length, condition): % of degenerate captions and breakdown by reason.")
    print("  (Reasons: empty | too_short | word_repetition | bigram_repetition)")
    print()
    headers = ["length", "condition", "n", "%degen", "empty", "too_short", "word_rep", "bigram_rep"]
    aligns =  ["<",      "<",         ">", ">",      ">",     ">",         ">",        ">"]

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for e in entries:
        groups[(e["length"], _condition_label(e))].append(e)

    rows_out = []
    for length in LENGTHS_ORDER:
        labels_in_length = sorted(
            {key[1] for key in groups if key[0] == length},
            key=_condition_sort_key,
        )
        for label in labels_in_length:
            cells = groups.get((length, label), [])
            if not cells:
                continue
            n_total = len(cells)
            counts = {"empty": 0, "too_short": 0,
                      "word_repetition": 0, "bigram_repetition": 0}
            n_degen = 0
            for c in cells:
                is_d, reason = classify_degeneration(c["caption"])
                if is_d:
                    n_degen += 1
                    counts[reason] = counts.get(reason, 0) + 1
            rows_out.append([
                length, label, n_total,
                f"{100 * n_degen / n_total:.1f}%",
                counts["empty"], counts["too_short"],
                counts["word_repetition"], counts["bigram_repetition"],
            ])
    _print_table(headers, rows_out, aligns)


def report_pair_samples(
    entries: list[dict],
    gt: dict[str, set[str]],
    *,
    n_samples: int,
    image_ids: list[str] | None,
) -> None:
    _section(f"3. PAIR SAMPLES — {n_samples} image(s), OFF vs SPARC side by side")
    print("  For visual inspection / picking panel examples. Annotated with")
    print("  ground-truth objects and the model's mentioned/hallucinated sets.")
    print()

    # Index entries by (image_id, length, condition).
    by_key: dict[tuple[str, str, str], dict] = {}
    for e in entries:
        by_key[(e["image_id"], e["length"], e["condition"])] = e

    # Pick image ids — caller override, or the first N image_ids that have
    # both OFF and ON in at least one length.
    if image_ids:
        chosen_ids = image_ids[:n_samples]
    else:
        all_image_ids = sorted({e["image_id"] for e in entries})
        chosen_ids = []
        for img_id in all_image_ids:
            for length in LENGTHS_ORDER:
                off = by_key.get((img_id, length, "off"))
                on = by_key.get((img_id, length, "on"))
                if off and on:
                    chosen_ids.append(img_id)
                    break
            if len(chosen_ids) >= n_samples:
                break

    if not chosen_ids:
        print("  (no images with both OFF and ON found)")
        return

    for img_id in chosen_ids:
        gt_objs = sorted(gt.get(img_id, set()))
        print(f"  ── image_id {img_id}  GT objects: {', '.join(gt_objs) if gt_objs else '(none)'}")
        for length in LENGTHS_ORDER:
            off = by_key.get((img_id, length, "off"))
            on = by_key.get((img_id, length, "on"))
            if not (off or on):
                continue
            print(f"    [{length}]")
            if off:
                cp = chair_per_caption(off["caption"], gt.get(img_id, set()))
                print(f"      OFF             : {off['caption']}")
                print(f"        mentioned    : {sorted(cp['mentioned'])}")
                print(f"        hallucinated : {sorted(cp['hallucinated'])}")
            if on:
                alpha = on.get("alpha")
                a_str = f"α={alpha:.1f}" if alpha is not None else ""
                cp = chair_per_caption(on["caption"], gt.get(img_id, set()))
                print(f"      ON {a_str:<8s} : {on['caption']}")
                print(f"        mentioned    : {sorted(cp['mentioned'])}")
                print(f"        hallucinated : {sorted(cp['hallucinated'])}")
            print()


# ---------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
        help="Directory containing captions.jsonl (= results/runs/<name>).")
    parser.add_argument("--annotations-dir", type=Path,
        default=DEFAULT_ANNOTATIONS_DIR,
        help="Where instances_val2017.json lives.")
    parser.add_argument("--auto-download", action="store_true",
        help="If instances_val2017.json is missing, download it (no manual step).")
    parser.add_argument("--pair-samples", type=int, default=3,
        help="How many images to show side by side in section 3.")
    parser.add_argument("--pair-image-ids", type=str, nargs="*", default=None,
        help="Specific image IDs for the pair samples (overrides auto-pick).")
    args = parser.parse_args()

    if not args.run_dir.exists():
        print(f"ERROR: run-dir {args.run_dir} does not exist.", file=sys.stderr)
        return 1

    captions_path = args.run_dir / "captions.jsonl"
    if not captions_path.exists():
        print(f"ERROR: {captions_path} not found.", file=sys.stderr)
        return 1

    instances_path = args.annotations_dir / "instances_val2017.json"
    if not instances_path.exists():
        if args.auto_download:
            print(f"  {instances_path} not found; downloading...")
            ensure_coco_annotations(args.annotations_dir)
        else:
            print(f"ERROR: {instances_path} not found.", file=sys.stderr)
            print("Run: python -m vr_modality_bias.data.coco_annotations", file=sys.stderr)
            print("  or re-run this script with --auto-download.", file=sys.stderr)
            return 1

    print("=" * 78)
    print("PHASE 3 CHAIR REPORT")
    print("=" * 78)
    print(f"  generated       : {datetime.now(tz=timezone.utc).isoformat(timespec='seconds')}")
    print(f"  run_dir         : {args.run_dir}")
    print(f"  captions        : {captions_path}")
    print(f"  annotations     : {instances_path}")
    print()

    entries = _load_captions(captions_path)
    gt = load_ground_truth_objects(instances_path)
    print(f"  loaded captions : {len(entries)}")
    print(f"  GT images       : {len(gt)}")

    if not entries:
        print("ERROR: captions.jsonl is empty.", file=sys.stderr)
        return 1

    report_chair_by_length(entries, gt)
    report_degeneration(entries)
    report_pair_samples(
        entries, gt,
        n_samples=args.pair_samples,
        image_ids=args.pair_image_ids,
    )

    print()
    print("=" * 78)
    print("END OF REPORT")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
