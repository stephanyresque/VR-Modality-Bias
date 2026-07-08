#!/usr/bin/env python
"""CHAIR report over a phase3_generate run: CHAIR / precision-recall tables,
degeneration rate and OFF-vs-ON pair samples (stdout + chair_results.{json,csv}).

Run: make chair-report  (or: python scripts/chair_report.py --run-dir results/runs/phase3 --auto-download)
"""

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
        load_reference_caption_objects,
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
        load_reference_caption_objects,
    )


LENGTHS_ORDER = ("short", "medium", "long")


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
    """Human-readable condition string for tables.

    Uses ``:g`` for the alpha so 1.05 shows as ``"1.05"`` (not the
    truncated ``"1.1"`` that ``:.1f`` would produce) while 1.1 stays
    ``"1.1"`` (no trailing zero).
    """
    if entry["condition"] == "off":
        return "off"
    return f"on α={float(entry.get('alpha', 0)):g}"


def _condition_sort_key(label: str) -> tuple[int, float]:
    if label == "off":
        return (0, 0.0)
    # "on α=1.1" → extract the float
    try:
        return (1, float(label.split("=")[1]))
    except (IndexError, ValueError):
        return (1, 0.0)


def report_chair_by_length(entries: list[dict], gt: dict[str, set[str]]) -> None:
    _section("1. CHAIR BY LENGTH — baseline vs SPARC")
    print("  CHAIR_s = fraction of captions with ≥1 hallucinated object (lower is better).")
    print("  CHAIR_i = fraction of OBJECT MENTIONS that are hallucinated (lower is better).")
    print()
    headers = ["length", "condition", "n", "CHAIR_s", "CHAIR_i", "total_mentioned", "total_hallucinated"]
    aligns =  ["<",      "<",         ">", ">",       ">",       ">",               ">"]

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


def report_precision_recall_by_length(
    entries: list[dict],
    gt_instances: dict[str, set[str]],
    *,
    gt_captions: dict[str, set[str]] | None = None,
) -> None:
    """Print precision / recall / F1 per (length, condition).

    Always shows precision (= 1 - CHAIR_i, instances-GT) and
    recall/F1 against the instances GT. When ``gt_captions`` is given,
    appends two more columns for the captions-GT recall and F1.

    A separate section from ``report_chair_by_length`` because the
    combined width wouldn't fit comfortably; this one is wider and
    paper-table-shaped.
    """
    _section("1b. PRECISION / RECALL / F1 -- per (length, condition)")
    print("  precision  = 1 - CHAIR_i        (always against instances GT)")
    print("  recall_X   = #correct / #GT_X   (X = instances or captions)")
    print("  f1_X       = harmonic mean of precision and recall_X")
    if gt_captions is None:
        print("  GT = instances only (--recall-gt instances)")
    else:
        print("  GT = both instances (CHAIR + recall_inst) and captions (recall_capt)")
    print()
    if gt_captions is None:
        headers = ["length", "condition", "n", "precision", "recall_inst", "f1_inst"]
    else:
        headers = ["length", "condition", "n",
                   "precision", "recall_inst", "f1_inst",
                   "recall_capt", "f1_capt"]
    aligns = [">"] * len(headers)
    aligns[0] = aligns[1] = "<"

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for e in entries:
        if e["image_id"] not in gt_instances:
            continue
        groups[(e["length"], _condition_label(e))].append(e)

    rows_out: list[list[str]] = []
    for length in LENGTHS_ORDER:
        labels = sorted({k[1] for k in groups if k[0] == length},
                        key=_condition_sort_key)
        for label in labels:
            cells = groups.get((length, label), [])
            if not cells:
                continue
            per_inst = [chair_per_caption(c["caption"], gt_instances[c["image_id"]])
                        for c in cells]
            agg_inst = compute_chair_aggregate(per_inst)
            row = [length, label, agg_inst["n_captions"],
                   f"{agg_inst['precision']:.4f}",
                   f"{agg_inst['recall']:.4f}",
                   f"{agg_inst['f1']:.4f}"]
            if gt_captions is not None:
                per_capt = [chair_per_caption(c["caption"], gt_captions.get(c["image_id"], set()))
                            for c in cells]
                agg_capt = compute_chair_aggregate(per_capt)
                row.extend([f"{agg_capt['recall']:.4f}", f"{agg_capt['f1']:.4f}"])
            rows_out.append(row)
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


# Same numbers the report_* functions PRINT, but collected into structured
# rows and written to ``chair_results.{json,csv}`` so the paper table can
# be assembled from the file without re-running the report. One row per
# (model_id, length, condition_label).


def _derive_model_id(entries: list[dict]) -> str:
    """Take ``model_id`` from the first entry that carries it.

    Bloco 1 spec: one captions.jsonl == one model family. If entries
    disagree, we still return the first — but log a warning to stderr so
    a stitched multi-family file gets noticed.
    """
    ids = {str(e.get("model_id")) for e in entries if e.get("model_id")}
    if not ids:
        return "unknown"
    ids_list = sorted(ids)
    if len(ids_list) > 1:
        print(
            f"  WARN: captions.jsonl carries multiple model_ids: {ids_list}; "
            f"using {ids_list[0]!r}.",
            file=sys.stderr,
        )
    return ids_list[0]


def collect_chair_rows(
    entries: list[dict],
    gt_instances: dict[str, set[str]],
    *,
    model_id: str,
    gt_captions: dict[str, set[str]] | None = None,
) -> list[dict]:
    """One row per (model_id, length, condition_label) with CHAIR + P/R/F1 + degen.

    Always computes CHAIR_s, CHAIR_i, and recall/F1 against ``gt_instances``
    (preserves the historical CHAIR setup; precision = 1 - CHAIR_i).
    When ``gt_captions`` is provided, also computes recall/F1 against the
    captions-GT (the SPARC paper's recall definition), populating the
    ``recall_captions`` / ``f1_captions`` columns alongside the instances
    versions.

    NaN-handling: numeric stats are ``None`` (not NaN) when no captions
    contribute — keeps the JSON serialisable and the CSV clean.
    """
    # CHAIR groups (skips images with no instances-GT annotation, since
    # that's the gating set for "can we ground at all").
    chair_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for e in entries:
        if e["image_id"] not in gt_instances:
            continue
        chair_groups[(e["length"], _condition_label(e))].append(e)

    # Degeneration groups (uses all captions — no GT filter).
    degen_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for e in entries:
        degen_groups[(e["length"], _condition_label(e))].append(e)

    rows: list[dict] = []
    for length in LENGTHS_ORDER:
        labels = sorted(
            {key[1] for key in (set(chair_groups) | set(degen_groups))
             if key[0] == length},
            key=_condition_sort_key,
        )
        for label in labels:
            key = (length, label)

            # --- CHAIR aggregate against GT-A (instances) ---
            chair_cells = chair_groups.get(key, [])
            if chair_cells:
                per_caption_inst = [
                    chair_per_caption(c["caption"], gt_instances[c["image_id"]])
                    for c in chair_cells
                ]
                agg_inst = compute_chair_aggregate(per_caption_inst)
                chair_s = agg_inst["chair_s"]
                chair_i = agg_inst["chair_i"]
                precision = agg_inst["precision"]
                recall_instances = agg_inst["recall"]
                f1_instances = agg_inst["f1"]
                n_chair = agg_inst["n_captions"]
                total_m = agg_inst["total_mentioned"]
                total_h = agg_inst["total_hallucinated"]
                total_c_inst = agg_inst["total_correct"]
                total_gt_inst = agg_inst["total_ground_truth"]
            else:
                chair_s = chair_i = precision = None
                recall_instances = f1_instances = None
                n_chair = total_m = total_h = total_c_inst = total_gt_inst = 0

            # --- GT-B aggregate (captions GT) only when provided ---
            recall_captions = f1_captions = None
            total_c_capt = total_gt_capt = 0
            if gt_captions is not None and chair_cells:
                per_caption_capt = [
                    chair_per_caption(c["caption"], gt_captions.get(c["image_id"], set()))
                    for c in chair_cells
                ]
                agg_capt = compute_chair_aggregate(per_caption_capt)
                recall_captions = agg_capt["recall"]
                f1_captions = agg_capt["f1"]
                total_c_capt = agg_capt["total_correct"]
                total_gt_capt = agg_capt["total_ground_truth"]

            # --- degeneration aggregate ---
            d_cells = degen_groups.get(key, [])
            n_total = len(d_cells)
            counts = {"empty": 0, "too_short": 0,
                      "word_repetition": 0, "bigram_repetition": 0}
            n_degen = 0
            for c in d_cells:
                is_d, reason = classify_degeneration(c["caption"])
                if is_d:
                    n_degen += 1
                    counts[reason] = counts.get(reason, 0) + 1
            pct_degen = (100.0 * n_degen / n_total) if n_total else None

            # --- pick alpha for ON rows (None for OFF) ---
            alpha = None
            if label.startswith("on"):
                try:
                    alpha = float(label.split("=")[1])
                except (IndexError, ValueError):
                    alpha = None
            condition = "off" if label == "off" else "on"

            rows.append({
                "model_id": model_id,
                "length": length,
                "condition": condition,
                "condition_label": label,
                "alpha": alpha,
                "n_captions": n_chair,
                "chair_s": chair_s,
                "chair_i": chair_i,
                # NEW: precision = 1 - chair_i (computed against instances).
                "precision": precision,
                # NEW: recall / F1 against GT-A (instances).
                "recall_instances": recall_instances,
                "f1_instances": f1_instances,
                # NEW: recall / F1 against GT-B (captions); None when
                # gt_captions wasn't provided.
                "recall_captions": recall_captions,
                "f1_captions": f1_captions,
                "total_mentioned": total_m,
                "total_hallucinated": total_h,
                "total_correct_instances": total_c_inst,
                "total_ground_truth_instances": total_gt_inst,
                "total_correct_captions": total_c_capt,
                "total_ground_truth_captions": total_gt_capt,
                "n_total_for_degen": n_total,
                "n_degen": n_degen,
                "pct_degen": pct_degen,
                "n_empty": counts["empty"],
                "n_too_short": counts["too_short"],
                "n_word_repetition": counts["word_repetition"],
                "n_bigram_repetition": counts["bigram_repetition"],
            })
    return rows


def write_chair_results(rows: list[dict], run_dir: Path) -> tuple[Path, Path]:
    """Write ``chair_results.{json,csv}`` under ``run_dir``. Returns both paths.

    JSON: structured (one record per row, NaN as null).
    CSV : flat table for paper / pandas / spreadsheet.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    json_path = run_dir / "chair_results.json"
    csv_path = run_dir / "chair_results.csv"

    # JSON — convert float NaN/None uniformly to null via json.dumps defaults.
    payload = {
        "generated_iso": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        "n_rows": len(rows),
        "rows": rows,
    }
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )

    # CSV -- fixed column order so the paper table downstream is stable.
    # precision/recall/f1 columns added per the recall-GT block.
    import csv as _csv
    columns = (
        "model_id", "length", "condition", "condition_label", "alpha",
        "n_captions", "chair_s", "chair_i",
        "precision", "recall_instances", "f1_instances",
        "recall_captions", "f1_captions",
        "total_mentioned", "total_hallucinated",
        "total_correct_instances", "total_ground_truth_instances",
        "total_correct_captions", "total_ground_truth_captions",
        "n_total_for_degen", "n_degen", "pct_degen",
        "n_empty", "n_too_short", "n_word_repetition", "n_bigram_repetition",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = _csv.DictWriter(fh, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row[k]) for k in columns})

    return json_path, csv_path


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
    parser.add_argument(
        "--recall-gt", choices=("instances", "captions", "both"), default="both",
        help="Which ground-truth definition to use for recall + F1. "
             "'instances' = the detection-annotation GT used historically by "
             "CHAIR (strict). 'captions' = objects mentioned in the human "
             "reference captions (the SPARC paper's definition, more aligned). "
             "'both' (default) computes and persists both, side by side. "
             "CHAIR_s/CHAIR_i and precision always use the instances GT.",
    )
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

    # GT-B (reference-caption objects) for the SPARC paper's recall
    # definition. Same zip as instances_val2017.json, so --auto-download
    # already grabbed it. Load lazily, only when needed.
    gt_captions = None
    if args.recall_gt in ("captions", "both"):
        captions_gt_path = args.annotations_dir / "captions_val2017.json"
        if not captions_gt_path.exists():
            if args.auto_download:
                print(f"  {captions_gt_path} not found; downloading...")
                ensure_coco_annotations(args.annotations_dir)
            else:
                print(f"ERROR: {captions_gt_path} not found.", file=sys.stderr)
                print(
                    "  Re-run with --auto-download, or restrict to "
                    "--recall-gt instances.", file=sys.stderr,
                )
                return 1
        gt_captions = load_reference_caption_objects(captions_gt_path)
        print(f"  caption-GT path : {captions_gt_path}")
        print(f"  caption-GT imgs : {len(gt_captions)}")

    print(f"  recall-gt mode  : {args.recall_gt}")

    if not entries:
        print("ERROR: captions.jsonl is empty.", file=sys.stderr)
        return 1

    report_chair_by_length(entries, gt)
    report_precision_recall_by_length(entries, gt, gt_captions=gt_captions)
    report_degeneration(entries)
    report_pair_samples(
        entries, gt,
        n_samples=args.pair_samples,
        image_ids=args.pair_image_ids,
    )

    # Persist the same numbers in structured form, now extended with
    # precision/recall/F1 (against the GT(s) the user asked for).
    model_id = _derive_model_id(entries)
    chair_rows = collect_chair_rows(
        entries, gt, model_id=model_id,
        gt_captions=gt_captions if args.recall_gt in ("captions", "both") else None,
    )
    json_path, csv_path = write_chair_results(chair_rows, args.run_dir)
    print()
    print(f"  persisted JSON : {json_path}")
    print(f"  persisted CSV  : {csv_path}")
    print(f"  model_id       : {model_id}")
    print(f"  rows           : {len(chair_rows)}")

    print()
    print("=" * 78)
    print("END OF REPORT")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
