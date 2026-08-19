#!/usr/bin/env python
"""ODI-Bench ingestion, in two passes because the component target is curation.

ODI-Bench records carry a question and an answer and nothing else; the
``target`` a component talks about is not in the data and cannot be extracted by
rule (the first existence question is "Is there anyone in the room?", whose
target is "anyone" -- not an object, and it would match nothing in a
description). So:

    --mode emit   scan the raw data, apply the filters, write a CSV with one
                  row per candidate component and an empty ``target`` column
    --mode build  read the filled CSV and write manifest.jsonl + questions.jsonl
                  (the default)

Run: python scripts/ingest_odibench.py --mode emit --raw data/raw/odibench \\
         --direction-terms configs/direction_terms.json --out targets.csv
     python scripts/ingest_odibench.py --raw data/raw/odibench \\
         --targets targets.csv --out data/processed/odibench
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from pyprojroot import here

try:
    from vr_modality_bias.data.annotations import (
        QuestionAnnotation,
        QuestionComponent,
        write_question_annotations,
    )
    from vr_modality_bias.data.manifests import ImageRecord, write_manifest
    from vr_modality_bias.data.vocabulary import load_vocabulary
except ModuleNotFoundError:
    sys.path.insert(0, str(here()))

    from src.vr_modality_bias.data.annotations import (
        QuestionAnnotation,
        QuestionComponent,
        write_question_annotations,
    )
    from src.vr_modality_bias.data.manifests import ImageRecord, write_manifest
    from src.vr_modality_bias.data.vocabulary import load_vocabulary


DATASET_NAME = "odibench"
IMAGE_PREFIX = "indoor"

# QA file -> component type. Two different files both annotate orientation.
QA_FILES: dict[str, str] = {
    "existence.jsonl": "existence",
    "counting.jsonl": "count",
    "view_orientation.jsonl": "direction",
    "relative.jsonl": "direction",
}

# Fixed order for the composed statement, so two runs over the same data
# produce the same question text.
TYPE_ORDER: tuple[str, ...] = ("existence", "count", "direction")

# The raw record's field names are the one thing here that was not verified
# against the real file (the raw data lives on the DGX, not in the repo). Each
# lookup tries these in order and raises naming the keys actually present, so a
# wrong guess fails on the first record instead of mis-parsing silently.
IMAGE_KEYS: tuple[str, ...] = (
    "image", "image_id", "image_name", "image_path", "img", "file_name", "filename",
)
QUESTION_KEYS: tuple[str, ...] = ("question", "text", "query", "prompt")
ANSWER_KEYS: tuple[str, ...] = ("answer", "gt", "ground_truth", "label", "response")

_JUNK_NAMES = {"thumbs.db", "desktop.ini", ".ds_store"}
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

_ARTICLES = ("the ", "a ", "an ")
_GENERIC_NOUNS = {
    "side", "sides", "part", "parts", "direction", "directions", "area", "region",
}

CSV_COLUMNS = ("image_id", "component_type", "question", "answer", "target")


@dataclass(frozen=True)
class Candidate:
    image_id: str
    component_type: str
    question: str
    answer: str


# ---------------------------------------------------------------- raw reading


def _pick(record: dict, keys: tuple[str, ...], what: str, source: str, lineno: int):
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    raise ValueError(
        f"{source}:{lineno}: no {what} field. Tried {list(keys)}; this record "
        f"has {sorted(record)}. Fix the key list at the top of "
        f"scripts/ingest_odibench.py."
    )


def iter_raw_components(raw_dir: Path):
    """Yield ``(component_type, image_id, question, answer)`` from QAs/*.jsonl."""
    qa_dir = Path(raw_dir) / "QAs"
    if not qa_dir.is_dir():
        raise FileNotFoundError(f"{qa_dir} not found.")
    for file_name, component_type in QA_FILES.items():
        path = qa_dir / file_name
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{lineno}: malformed JSON: {exc}") from exc
                image_ref = _pick(record, IMAGE_KEYS, "image", str(path), lineno)
                question = _pick(record, QUESTION_KEYS, "question", str(path), lineno)
                answer = _pick(record, ANSWER_KEYS, "answer", str(path), lineno)
                yield (
                    component_type,
                    Path(str(image_ref)).stem,
                    str(question).strip(),
                    str(answer),
                )


# ---------------------------------------------------------------- filters


def normalize_existence(answer) -> str | None:
    text = str(answer).strip().lower().rstrip(".").strip()
    return text if text in {"yes", "no"} else None


def normalize_count(answer) -> int | None:
    text = str(answer).strip().rstrip(".").strip()
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


def normalize_direction(answer, directions) -> str | None:
    """Return the canonical direction, or None when it is not a single axis.

    Drops composites (``front-left``), whole-sentence answers, and the values
    that are not directions at all -- the annotation carries all three.
    """
    text = str(answer).strip().lower().rstrip(".").strip()
    for article in _ARTICLES:
        if text.startswith(article):
            text = text[len(article):].strip()
            break
    words = text.split()
    while len(words) > 1 and words[-1] in _GENERIC_NOUNS:
        words = words[:-1]
    text = " ".join(words)
    return directions.synonym_to_category.get(text)


def normalize_answer(component_type: str, answer, directions):
    if component_type == "existence":
        return normalize_existence(answer)
    if component_type == "count":
        return normalize_count(answer)
    if component_type == "direction":
        return normalize_direction(answer, directions)
    raise ValueError(f"unknown component type {component_type!r}")


def select_candidates(raw_components, directions) -> tuple[list[Candidate], dict]:
    """Apply every selection filter. Returns the survivors and a tally."""
    tally = {
        "seen": 0, "wrong_prefix": 0, "bad_answer": 0,
        "kept_by_type": defaultdict(int),
    }
    kept: list[Candidate] = []
    for component_type, image_id, question, answer in raw_components:
        tally["seen"] += 1
        if not image_id.startswith(IMAGE_PREFIX):
            tally["wrong_prefix"] += 1
            continue
        normalized = normalize_answer(component_type, answer, directions)
        if normalized is None:
            tally["bad_answer"] += 1
            continue
        kept.append(Candidate(image_id, component_type, question, str(normalized)))
        tally["kept_by_type"][component_type] += 1
    return kept, tally


def group_by_image(candidates: list[Candidate]) -> dict[str, list[Candidate]]:
    grouped: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.image_id].append(candidate)
    return grouped


def qualifies(components: list[Candidate]) -> bool:
    return len({c.component_type for c in components}) >= 2


def sample_images(image_ids: list[str], *, limit: int | None, seed: int) -> list[str]:
    """Deterministic subset. Sampled, not sorted-and-cut: the latter biases the
    selection towards whatever the low indices happen to be."""
    population = sorted(image_ids)
    if limit is None or limit >= len(population):
        return population
    return sorted(random.Random(seed).sample(population, limit))


# ---------------------------------------------------------------- images


def index_images(raw_dir: Path) -> tuple[dict[str, Path], int]:
    """``{image_id: path}`` over images_final, plus how many files were skipped.

    A base name shared by two different extensions is fatal: the pipeline keys
    everything on the stem, so picking one silently would drop the other's
    annotations without a word.
    """
    images_dir = Path(raw_dir) / "images_final"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"{images_dir} not found.")

    index: dict[str, Path] = {}
    skipped = 0
    for path in sorted(images_dir.iterdir()):
        if not path.is_file():
            continue
        if path.name.startswith(".") or path.name.lower() in _JUNK_NAMES:
            continue
        if path.suffix.lower() not in _IMAGE_SUFFIXES:
            skipped += 1
            continue
        if path.stem in index:
            raise ValueError(
                f"two files share the base name {path.stem!r}, which the "
                f"pipeline uses as the image_id:\n"
                f"  {index[path.stem]}\n  {path}\n"
                f"Rename one in the raw data; this script will not pick for you."
            )
        index[path.stem] = path
    return index, skipped


# ---------------------------------------------------------------- composition


def order_components(components: list[Candidate]) -> list[Candidate]:
    return sorted(components, key=lambda c: TYPE_ORDER.index(c.component_type))


def compose_question(components: list[Candidate]) -> str:
    """Concatenate the original statements. They were written by humans;
    rewriting them would put a variable of ours inside their ground truth."""
    return " ".join(c.question for c in components)


def apply_negative_existence_rule(
    ordered: list[tuple[Candidate, str]]
) -> tuple[list[tuple[Candidate, str]], int]:
    """Drop a component that asks about a target a negative existence just denied.

    Counting or locating something the answer already said is absent is a
    degenerate sub-question. Needs the curated targets, so it only runs in build
    mode. Returns the kept pairs and how many were dropped.
    """
    denied = {
        target for candidate, target in ordered
        if candidate.component_type == "existence" and candidate.answer == "no"
    }
    if not denied:
        return ordered, 0
    kept = [
        (candidate, target) for candidate, target in ordered
        if candidate.component_type == "existence" or target not in denied
    ]
    return kept, len(ordered) - len(kept)


# ---------------------------------------------------------------- emit mode


def write_targets_csv(candidates: list[Candidate], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for candidate in candidates:
            writer.writerow({
                "image_id": candidate.image_id,
                "component_type": candidate.component_type,
                "question": candidate.question,
                "answer": candidate.answer,
                "target": "",
            })
    return len(candidates)


def run_emit(args) -> int:
    directions = load_vocabulary(args.direction_terms)
    candidates, tally = select_candidates(iter_raw_components(args.raw), directions)

    grouped = group_by_image(candidates)
    qualified = [image_id for image_id, comps in grouped.items() if qualifies(comps)]
    chosen = sample_images(qualified, limit=args.limit, seed=args.seed)
    chosen_set = set(chosen)

    selected = [c for c in candidates if c.image_id in chosen_set]
    selected.sort(key=lambda c: (c.image_id, TYPE_ORDER.index(c.component_type)))
    n = write_targets_csv(selected, args.out)

    meta = {
        "dataset": DATASET_NAME,
        "raw": str(args.raw),
        "direction_terms": str(args.direction_terms),
        "seed": args.seed,
        "limit": args.limit,
        "components_seen": tally["seen"],
        "dropped_wrong_prefix": tally["wrong_prefix"],
        "dropped_bad_answer": tally["bad_answer"],
        "kept_by_type": dict(tally["kept_by_type"]),
        "images_with_any_component": len(grouped),
        "images_qualified": len(qualified),
        "images_selected": len(chosen),
        "rows_written": n,
    }
    meta_path = args.out.with_suffix(args.out.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print("=" * 78)
    print("ODI-BENCH INGESTION -- emit")
    print("=" * 78)
    print(f"  components seen        : {tally['seen']}")
    print(f"  dropped, wrong prefix  : {tally['wrong_prefix']}")
    print(f"  dropped, bad answer    : {tally['bad_answer']}")
    for component_type in TYPE_ORDER:
        print(f"  kept, {component_type:<10}       : {tally['kept_by_type'][component_type]}")
    print(f"  images with a component: {len(grouped)}")
    print(f"  images qualified (>=2 distinct types): {len(qualified)}")
    print(f"  images selected (limit={args.limit}, seed={args.seed}): {len(chosen)}")
    print(f"  rows written           : {n} -> {args.out}")
    print(f"  provenance             : {meta_path}")
    print()
    print("  Fill the 'target' column by hand, then re-run with --mode build.")
    print("=" * 78)
    return 0


# ---------------------------------------------------------------- build mode


def read_targets_csv(path: Path) -> tuple[list[tuple[Candidate, str]], int]:
    rows: list[tuple[Candidate, str]] = []
    untargeted = 0
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in CSV_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing column(s) {missing}.")
        for row in reader:
            target = (row.get("target") or "").strip()
            if not target:
                untargeted += 1
                continue
            rows.append((
                Candidate(
                    image_id=(row["image_id"] or "").strip(),
                    component_type=(row["component_type"] or "").strip(),
                    question=(row["question"] or "").strip(),
                    answer=(row["answer"] or "").strip(),
                ),
                target,
            ))
    return rows, untargeted


def build_components(candidate: Candidate, target: str) -> QuestionComponent:
    answer: str | int = candidate.answer
    if candidate.component_type == "count":
        answer = int(candidate.answer)
    return QuestionComponent(
        component_type=candidate.component_type, target=target, answer=answer,
    )


def run_build(args) -> int:
    rows, untargeted = read_targets_csv(args.targets)
    if not rows:
        print(
            f"ERROR: {args.targets} has no row with a filled 'target'.",
            file=sys.stderr,
        )
        return 1

    index, skipped_files = index_images(args.raw)

    grouped: dict[str, list[tuple[Candidate, str]]] = defaultdict(list)
    for candidate, target in rows:
        grouped[candidate.image_id].append((candidate, target))

    out_dir = Path(args.out)
    images_out = out_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    manifest: list[ImageRecord] = []
    questions: list[QuestionAnnotation] = []
    n_dropped_by_negative = 0
    n_single_component = 0
    missing_images: list[str] = []

    for image_id in sorted(grouped):
        if image_id not in index:
            missing_images.append(image_id)
            continue
        ordered = sorted(
            grouped[image_id], key=lambda pair: TYPE_ORDER.index(pair[0].component_type)
        )
        ordered, dropped = apply_negative_existence_rule(ordered)
        n_dropped_by_negative += dropped
        if not ordered:
            continue
        if len(ordered) == 1:
            n_single_component += 1

        source = index[image_id]
        destination = images_out / source.name
        # Copied, never moved: the raw data stays exactly as delivered.
        shutil.copyfile(source, destination)
        # Real pixel dimensions, and no resizing anywhere: handing the image
        # over raw is the condition the experiment measures.
        with Image.open(destination) as image:
            width, height = image.size

        manifest.append(ImageRecord(
            image_id=image_id,
            file_name=destination.name,
            width=int(width),
            height=int(height),
            source=DATASET_NAME,
            dataset=DATASET_NAME,
        ))
        questions.append(QuestionAnnotation(
            image_id=image_id,
            question_id=f"{image_id}_q1",
            question_text=compose_question([c for c, _ in ordered]),
            components=tuple(build_components(c, t) for c, t in ordered),
        ))

    if missing_images:
        print(
            f"ERROR: {len(missing_images)} image_id(s) in {args.targets} have no "
            f"file under {args.raw}/images_final: {missing_images[:5]}",
            file=sys.stderr,
        )
        return 1

    manifest_path = out_dir / "manifest.jsonl"
    questions_path = out_dir / "questions.jsonl"
    write_manifest(manifest, manifest_path)
    write_question_annotations(questions, questions_path)

    print("=" * 78)
    print("ODI-BENCH INGESTION -- build")
    print("=" * 78)
    print(f"  rows with a target     : {len(rows)}")
    print(f"  rows without a target  : {untargeted} (ignored)")
    print(f"  files skipped in images_final (not an image): {skipped_files}")
    print(f"  components dropped by the negative-existence rule: {n_dropped_by_negative}")
    print(f"  items left with a single component: {n_single_component}")
    print(f"  manifest               : {len(manifest)} item(s) -> {manifest_path}")
    print(f"  questions              : {len(questions)} item(s) -> {questions_path}")
    print(f"  images copied to       : {images_out}")
    print("=" * 78)
    return 0


# ---------------------------------------------------------------- cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("emit", "build"), default="build",
        help="'emit' writes the candidate CSV for manual targeting; 'build' "
             "(the default) turns the filled CSV into manifest + questions.")
    parser.add_argument("--raw", type=Path, default=Path("data/raw/odibench"),
        help="Raw ODI-Bench root, holding QAs/ and images_final/.")
    parser.add_argument("--direction-terms", type=Path,
        default=Path("configs/direction_terms.json"),
        help="Direction table; a direction answer is kept only if it resolves "
             "here, which is what rejects composites and noise. Emit mode only.")
    parser.add_argument("--out", type=Path, default=None,
        help="emit: the CSV to write. build: the output dataset directory.")
    parser.add_argument("--targets", type=Path, default=None,
        help="build: the CSV with the 'target' column filled in.")
    parser.add_argument("--limit", type=int, default=None,
        help="emit: keep this many qualified images, sampled with --seed.")
    parser.add_argument("--seed", type=int, default=0,
        help="emit: sampling seed, recorded alongside the CSV.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.mode == "emit":
        if args.out is None:
            print("ERROR: --out is required in emit mode.", file=sys.stderr)
            return 1
        return run_emit(args)
    if args.targets is None or args.out is None:
        print("ERROR: --targets and --out are required in build mode.", file=sys.stderr)
        return 1
    return run_build(args)


if __name__ == "__main__":
    raise SystemExit(main())
