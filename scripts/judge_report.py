#!/usr/bin/env python
# Run: python scripts/judge_report.py --run-dir results/runs/arm1_sparc_q \
#          --questions data/processed/odibench/questions.jsonl [--dry-run]

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from pyprojroot import here

try:
    from vr_modality_bias.data.annotations import read_question_annotations
    from vr_modality_bias.metrics.judge import (
        ALL_LABELS,
        VERDICT_CORRECT,
        VERDICT_INVALID,
        build_judge_prompt,
        compute_judge_aggregate,
        group_by_arm,
        parse_verdict,
    )
    from vr_modality_bias.metrics.report import (
        assert_entries_are_grounded,
        classify_degeneration,
        condition_label,
        condition_sort_key,
        derive_model_id,
        print_table,
        section,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(here()))

    from src.vr_modality_bias.data.annotations import read_question_annotations
    from src.vr_modality_bias.metrics.judge import (
        ALL_LABELS,
        VERDICT_CORRECT,
        VERDICT_INVALID,
        build_judge_prompt,
        compute_judge_aggregate,
        group_by_arm,
        parse_verdict,
    )
    from src.vr_modality_bias.metrics.report import (
        assert_entries_are_grounded,
        classify_degeneration,
        condition_label,
        condition_sort_key,
        derive_model_id,
        print_table,
        section,
    )


DEFAULT_JUDGE_MODEL = "Qwen/Qwen3-32B"
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_SEED = 0
ANSWERS_NAME = "answers.jsonl"
VERDICTS_NAME = "verdicts.jsonl"


def _iso_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def _append(jsonl_path: Path, entry: dict) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    entries: list[dict] = []
    if not path.exists():
        return entries
    with path.open("r", encoding="utf-8") as fh:
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


def verdict_key(
    image_id, question_id, condition, component_index
) -> tuple[str, str, str, int]:
    return (str(image_id), str(question_id), str(condition), int(component_index))


def read_done(path: Path) -> set[tuple[str, str, str, int]]:
    done: set[tuple[str, str, str, int]] = set()
    for entry in _read_jsonl(path):
        try:
            done.add(
                verdict_key(
                    entry["image_id"], entry["question_id"],
                    entry["condition"], entry["component_index"],
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return done


def limit_entries(entries: list[dict], limit: int | None) -> list[dict]:
    if limit is None:
        return entries
    pairs = sorted({(str(e["image_id"]), str(e["question_id"])) for e in entries})
    keep = set(pairs[:limit])
    return [
        e for e in entries
        if (str(e["image_id"]), str(e["question_id"])) in keep
    ]


def build_tasks(entries: list[dict], questions: dict) -> list[dict]:
    tasks: list[dict] = []
    for entry in entries:
        key = (str(entry["image_id"]), str(entry["question_id"]))
        question = questions[key]
        answer = str(entry.get("answer", ""))
        for index, component in enumerate(question.components):
            tasks.append({
                "image_id": entry["image_id"],
                "question_id": entry["question_id"],
                "condition": entry["condition"],
                "condition_label": condition_label(entry),
                "component_index": index,
                "component_type": component.component_type,
                "sub_question": component.question,
                "reference_answer": component.answer,
                "answer": answer,
                "prompt": build_judge_prompt(
                    composed_question=question.question_text,
                    sub_question=component.question,
                    reference_answer=component.answer,
                    generated_answer=answer,
                ),
            })
    return tasks


def run_judging(
    tasks: list[dict],
    judge_fn,
    verdicts_path: Path,
    *,
    done: set,
    judge_meta: dict,
    progress_every: int = 25,
) -> tuple[int, int]:
    n_judged = 0
    n_skipped = 0
    for task in tasks:
        key = verdict_key(
            task["image_id"], task["question_id"],
            task["condition"], task["component_index"],
        )
        if key in done:
            n_skipped += 1
            continue

        outcome = parse_verdict(judge_fn(task["prompt"]))
        _append(verdicts_path, {
            "image_id": task["image_id"],
            "question_id": task["question_id"],
            "condition": task["condition"],
            "condition_label": task["condition_label"],
            "component_index": task["component_index"],
            "component_type": task["component_type"],
            "sub_question": task["sub_question"],
            "reference_answer": task["reference_answer"],
            "verdict": outcome.verdict,
            "evidence": outcome.evidence,
            "invalid_reason": outcome.reason,
            "judge_raw": outcome.raw if outcome.verdict == VERDICT_INVALID else "",
            **judge_meta,
            "timestamp_iso": _iso_now(),
        })
        done.add(key)
        n_judged += 1
        if progress_every and n_judged % progress_every == 0:
            print(f"  judged {n_judged} (skipped {n_skipped})", file=sys.stderr)
    return n_judged, n_skipped


def collect_items(entries: list[dict], records: list[dict]) -> list[dict]:
    by_cell: dict[tuple[str, str, str], list[dict]] = {}
    for record in records:
        cell = (
            str(record["image_id"]), str(record["question_id"]),
            str(record["condition"]),
        )
        by_cell.setdefault(cell, []).append(record)

    items: list[dict] = []
    for entry in entries:
        cell = (
            str(entry["image_id"]), str(entry["question_id"]),
            str(entry["condition"]),
        )
        verdicts = sorted(
            by_cell.get(cell, []), key=lambda r: int(r["component_index"])
        )
        if not verdicts:
            continue
        answer = str(entry.get("answer", ""))
        is_degenerate, reason = classify_degeneration(answer)
        items.append({
            "image_id": entry["image_id"],
            "question_id": entry["question_id"],
            "condition": entry["condition"],
            "condition_label": condition_label(entry),
            "answer": answer,
            "is_degenerate": is_degenerate,
            "degeneration_reason": reason,
            "verdicts": verdicts,
            "entry": entry,
        })
    return items


# ---------------------------------------------------------------- report


def report_by_component_type(groups: dict[str, list[dict]]) -> None:
    section("1. VERDICTS BY COMPONENT TYPE -- per arm")
    print("  The unit is the SUB-QUESTION, not the item.")
    print("  The three verdicts are never collapsed: an arm that answers less")
    print("  raises %not_addr while raising %correct, and both have to be visible.")
    print("  %invalid is the judge failing to answer; it is never folded into")
    print("  the other three, and it is inside the denominator.")
    print()
    headers = ["arm", "component", "n", "%correct", "%incorrect", "%not_addr", "%invalid"]
    aligns = ["<", "<", ">", ">", ">", ">", ">"]

    rows: list[list] = []
    for label in sorted(groups, key=condition_sort_key):
        agg = compute_judge_aggregate(groups[label])
        for component_type, stats in agg["by_type"].items():
            rows.append([
                label, component_type, stats["n_subquestions"],
                f"{100 * stats['rate_correct']:.1f}%",
                f"{100 * stats['rate_incorrect']:.1f}%",
                f"{100 * stats['rate_not_addressed']:.1f}%",
                f"{100 * stats['rate_invalid']:.1f}%",
            ])
    print_table(headers, rows, aligns)


def report_overall(groups: dict[str, list[dict]]) -> None:
    section("2. OVERALL -- per arm, all sub-questions pooled")
    print("  all_correct = fraction of ITEMS whose every sub-question is correct.")
    print("  words and %degen are deterministic and cost no judge call. They are")
    print("  the counterweight: an arm cannot win by answering shorter without")
    print("  it showing up here.")
    print()
    headers = [
        "arm", "items", "subq", "%correct", "%incorrect", "%not_addr",
        "%invalid", "all_correct", "words", "%degen",
    ]
    aligns = ["<"] + [">"] * 9

    rows: list[list] = []
    for label in sorted(groups, key=condition_sort_key):
        agg = compute_judge_aggregate(groups[label])
        overall = agg["overall"]
        rows.append([
            label, agg["n_items"], overall["n_subquestions"],
            f"{100 * overall['rate_correct']:.1f}%",
            f"{100 * overall['rate_incorrect']:.1f}%",
            f"{100 * overall['rate_not_addressed']:.1f}%",
            f"{100 * overall['rate_invalid']:.1f}%",
            f"{100 * agg['rate_all_correct']:.1f}%",
            f"{agg['mean_answer_words']:.1f}",
            f"{100 * agg['rate_degenerate']:.1f}%",
        ])
    print_table(headers, rows, aligns)


def report_audit_sample(items: list[dict], *, n_samples: int) -> None:
    section(f"3. AUDIT SAMPLE -- {n_samples} item(s) with the judge's evidence")
    print("  A judge is as fragile as string matching was, and the only way to")
    print("  know whether it works is to read it. Anything not fully correct")
    print("  sorts first, so the sample shows the decisions worth checking.")
    print()

    def _interest(item: dict) -> tuple[int, str, str]:
        verdicts = [v["verdict"] for v in item["verdicts"]]
        clean = all(v == VERDICT_CORRECT for v in verdicts)
        return (1 if clean else 0, str(item["image_id"]), str(item["question_id"]))

    if not items:
        print("  (no items to sample)")
        return

    for item in sorted(items, key=_interest)[:n_samples]:
        print(f"  ── {item['image_id']} / {item['question_id']}  "
              f"[{item['condition_label']}]")
        print(f"     A: {item['answer']}")
        if item["is_degenerate"]:
            print(f"     ! degenerate ({item['degeneration_reason']})")
        for verdict in item["verdicts"]:
            print(
                f"       [{verdict['verdict']:<13}] "
                f"{verdict['component_type']:<9} "
                f"ref={verdict['reference_answer']!r}"
            )
            print(f"                       Q: {verdict['sub_question']}")
            evidence = verdict.get("evidence") or "(nothing in the answer matched)"
            print(f"                       evidence: {evidence}")
            if verdict["verdict"] == VERDICT_INVALID:
                print(f"                       invalid: {verdict.get('invalid_reason')}")
        print()


def collect_rows(groups: dict[str, list[dict]], *, model_id: str, judge_meta: dict):
    rows: list[dict] = []
    for label in sorted(groups, key=condition_sort_key):
        agg = compute_judge_aggregate(groups[label])
        overall = agg["overall"]
        row = {
            "model_id": model_id,
            "judge_model": judge_meta.get("judge_model"),
            "judge_revision": judge_meta.get("judge_revision"),
            "judge_seed": judge_meta.get("judge_seed"),
            "condition": "off" if label == "off" else "on",
            "condition_label": label,
            "n_items": agg["n_items"],
            "n_subquestions": overall["n_subquestions"],
            "n_all_correct": agg["n_all_correct"],
            "rate_all_correct": agg["rate_all_correct"],
            "mean_answer_words": agg["mean_answer_words"],
            "n_degenerate": agg["n_degenerate"],
            "rate_degenerate": agg["rate_degenerate"],
        }
        for label_name in ALL_LABELS:
            row[f"n_{label_name}"] = overall[f"n_{label_name}"]
            row[f"rate_{label_name}"] = overall[f"rate_{label_name}"]
        for component_type, stats in agg["by_type"].items():
            row[f"n_subquestions_{component_type}"] = stats["n_subquestions"]
            for label_name in ALL_LABELS:
                row[f"rate_{label_name}_{component_type}"] = stats[f"rate_{label_name}"]
        rows.append(row)
    return rows


def write_results(rows: list[dict], run_dir: Path, judge_meta: dict):
    run_dir.mkdir(parents=True, exist_ok=True)
    json_path = run_dir / "judge_results.json"
    csv_path = run_dir / "judge_results.csv"

    json_path.write_text(
        json.dumps(
            {
                "generated_iso": _iso_now(),
                "judge": judge_meta,
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


# ---------------------------------------------------------------- the judge


def _assert_not_offloaded(model) -> None:
    # device_map="auto" silently spills to CPU or disk when VRAM is short, which
    # turns a one-hour job into a thirty-hour one with no error anywhere. Fail
    # instead, naming the modules that did not fit.
    device_map = getattr(model, "hf_device_map", None)
    if not device_map:
        return
    stranded = sorted(
        name for name, device in device_map.items()
        if str(device) in {"cpu", "disk"} or device is None
    )
    if stranded:
        raise RuntimeError(
            f"{len(stranded)} module(s) of the judge were placed off the GPU "
            f"({stranded[:5]}). Generation would be unusably slow. Use a "
            f"smaller judge or free VRAM; this is not something to wait out."
        )


def build_hf_judge(
    *, model_id: str, revision: str | None, seed: int, max_new_tokens: int
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    torch.manual_seed(seed)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        quantization_config=quantization,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    _assert_not_offloaded(model)

    def _render(prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        # Qwen3 emits <think>...</think> by default, which a strict JSON parser
        # would reject on every single call. Templates without the switch just
        # ignore it, and older ones raise, hence the fallback.
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

    def judge_fn(prompt: str) -> str:
        text = _render(prompt)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[-1]:]
        return tokenizer.decode(generated, skip_special_tokens=True)

    return judge_fn


# ---------------------------------------------------------------- cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score a composed-question run with an open text-only judge."
    )
    parser.add_argument("--run-dir", type=Path, required=True,
        help=f"Directory holding {ANSWERS_NAME} (= <output-root>/<run-name>).")
    parser.add_argument("--questions", type=Path, required=True,
        help="JSON Lines question annotations, carrying each sub-question and "
             "its human reference answer.")
    parser.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL,
        help="Judge model id. Text-only on purpose: a judge that saw the "
             "panorama would share the defect under test.")
    parser.add_argument("--judge-revision", type=str, default=None,
        help="Judge revision (commit or tag), recorded in the output.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
        help="Torch seed, recorded in the output. Decoding is greedy anyway.")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--limit", type=int, default=None,
        help="Judge only the first N (image, question) pairs, all arms.")
    parser.add_argument("--sample", type=int, default=5,
        help="How many items to print in the audit section.")
    parser.add_argument("--dry-run", action="store_true",
        help="Build the prompts, print a few, load no model and write nothing.")
    parser.add_argument("--dry-run-prompts", type=int, default=3,
        help="How many prompts --dry-run prints.")
    return parser


def main(argv=None, judge_factory=build_hf_judge) -> int:
    args = build_parser().parse_args(argv)

    answers_path = args.run_dir / ANSWERS_NAME
    for path, what in (
        (args.run_dir, "run-dir"),
        (answers_path, ANSWERS_NAME),
        (args.questions, "questions"),
    ):
        if not path.exists():
            print(f"ERROR: {what} {path} not found.", file=sys.stderr)
            return 1

    entries = _read_jsonl(answers_path)
    if not entries:
        print(f"ERROR: {answers_path} is empty.", file=sys.stderr)
        return 1
    questions = index_questions(args.questions)

    try:
        assert_entries_are_grounded(
            entries, questions,
            id_fields=("image_id", "question_id"), entry_noun="answer",
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    entries = limit_entries(entries, args.limit)
    tasks = build_tasks(entries, questions)

    print("=" * 78)
    print("COMPOSED-QUESTION JUDGE REPORT")
    print("=" * 78)
    print(f"  generated       : {_iso_now()}")
    print(f"  run_dir         : {args.run_dir}")
    print(f"  answers         : {answers_path} ({len(entries)} cell(s))")
    print(f"  questions       : {args.questions} ({len(questions)} pair(s))")
    print(f"  sub-questions   : {len(tasks)}")
    print(f"  judge model     : {args.judge_model}")
    print(f"  judge revision  : {args.judge_revision or '(default)'}")
    print(f"  judge sees      : text only, never the image")
    print(f"  decoding        : greedy, seed {args.seed}")

    if args.dry_run:
        section(f"DRY RUN -- {args.dry_run_prompts} prompt(s), no model loaded")
        for task in tasks[:args.dry_run_prompts]:
            print(f"── {task['image_id']} / {task['question_id']} "
                  f"[{task['condition_label']}] component {task['component_index']} "
                  f"({task['component_type']})")
            print("-" * 78)
            print(task["prompt"])
            print("-" * 78)
            print()
        print(f"  {len(tasks)} prompt(s) would be sent to {args.judge_model}.")
        print("  Nothing was written.")
        return 0

    verdicts_path = args.run_dir / VERDICTS_NAME
    done = read_done(verdicts_path)
    print(f"  resume state    : {len(done)} verdict(s) already in {verdicts_path}")

    judge_meta = {
        "judge_model": args.judge_model,
        "judge_revision": args.judge_revision,
        "judge_seed": args.seed,
    }
    judge_fn = judge_factory(
        model_id=args.judge_model,
        revision=args.judge_revision,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
    )

    n_judged, n_skipped = run_judging(
        tasks, judge_fn, verdicts_path, done=done, judge_meta=judge_meta,
    )
    print(f"  judged          : {n_judged}   skipped (resume): {n_skipped}")

    records = _read_jsonl(verdicts_path)
    items = collect_items(entries, records)
    groups = group_by_arm(items)

    report_by_component_type(groups)
    report_overall(groups)
    report_audit_sample(items, n_samples=args.sample)

    model_id = derive_model_id([i["entry"] for i in items])
    rows = collect_rows(groups, model_id=model_id, judge_meta=judge_meta)
    json_path, csv_path = write_results(rows, args.run_dir, judge_meta)

    n_invalid = sum(1 for r in records if r.get("verdict") == VERDICT_INVALID)
    print()
    print(f"  verdicts       : {verdicts_path} ({len(records)} line(s))")
    print(f"  persisted JSON : {json_path}")
    print(f"  persisted CSV  : {csv_path}")
    print(f"  evaluated model: {model_id}")
    if n_invalid:
        print(f"  WARN: {n_invalid} verdict(s) came back invalid; they are counted "
              f"separately and never folded into the other three.")

    print()
    print("=" * 78)
    print("END OF REPORT")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
