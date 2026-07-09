#!/usr/bin/env python
"""Build the POPE yes/no question set over the manifest images from
instances_val2017.json, with the three official negative-sampling strategies.

Run: python scripts/build_pope.py --auto-download
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger
from pyprojroot import here

try:
    from vr_modality_bias.data.coco_annotations import (
        DEFAULT_TARGET_DIR as DEFAULT_ANNOTATIONS_DIR,
        ensure_coco_annotations,
    )
    from vr_modality_bias.data.manifests import read_manifest
    from vr_modality_bias.data.prompts import get_prompt
    from vr_modality_bias.metrics.chair import (
        COCO_CATEGORIES,
        load_ground_truth_objects,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(here()))

    from src.vr_modality_bias.data.coco_annotations import (
        DEFAULT_TARGET_DIR as DEFAULT_ANNOTATIONS_DIR,
        ensure_coco_annotations,
    )
    from src.vr_modality_bias.data.manifests import read_manifest
    from src.vr_modality_bias.data.prompts import get_prompt
    from src.vr_modality_bias.metrics.chair import (
        COCO_CATEGORIES,
        load_ground_truth_objects,
    )


STRATEGIES = ("random", "popular", "adversarial")

DEFAULT_MANIFEST = Path("data/processed/mscoco_baseline/manifest.jsonl")
DEFAULT_OUTPUT = Path("data/processed/mscoco_baseline/pope_questions.jsonl")


def _iso_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def render_question(object_name: str) -> str:
    """Render the official POPE template for one object."""
    return get_prompt("vqa_pope").format(object=object_name)


def object_frequency(image_to_objects: dict[str, set[str]]) -> Counter:
    """How many images contain each category. Drives the ``popular`` strategy."""
    freq: Counter = Counter()
    for objects in image_to_objects.values():
        freq.update(objects)
    return freq


def cooccurrence(image_to_objects: dict[str, set[str]]) -> dict[str, Counter]:
    """``co[a][b]`` = number of images containing both ``a`` and ``b`` (a != b).

    Computed over the WHOLE annotation file, not just the images we sample, so
    the adversarial ranking is a property of COCO and not of our 100-image
    subset. That matches the POPE protocol.
    """
    co: dict[str, Counter] = defaultdict(Counter)
    for objects in image_to_objects.values():
        ordered = sorted(objects)
        for a in ordered:
            for b in ordered:
                if a != b:
                    co[a][b] += 1
    return co


def negative_objects(
    present: set[str],
    strategy: str,
    *,
    k: int,
    frequency: Counter,
    co: dict[str, Counter],
    rng: random.Random,
    categories: tuple[str, ...] = COCO_CATEGORIES,
) -> list[str]:
    """Pick ``k`` categories absent from the image, per the POPE strategy.

    * ``random``      -- uniform over the absent categories.
    * ``popular``     -- the most frequent categories in the dataset.
    * ``adversarial`` -- the categories that co-occur most with what IS in the
      image; the hardest negatives, because context predicts them.

    Ties are broken deterministically (frequency, then name) so the same seed
    always yields the same question set.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}. Known: {STRATEGIES}.")

    absent = sorted(c for c in categories if c not in present)
    k = min(k, len(absent))
    if k == 0:
        return []

    if strategy == "random":
        return sorted(rng.sample(absent, k))
    if strategy == "popular":
        ranked = sorted(absent, key=lambda c: (-frequency[c], c))
        return ranked[:k]
    scores = {c: sum(co[p][c] for p in present) for c in absent}
    ranked = sorted(absent, key=lambda c: (-scores[c], -frequency[c], c))
    return ranked[:k]


def questions_for_image(
    image_id: str,
    present: set[str],
    *,
    k: int,
    frequency: Counter,
    co: dict[str, Counter],
    rng: random.Random,
    categories: tuple[str, ...] = COCO_CATEGORIES,
) -> list[dict]:
    """Rows for one image: ``k`` yes + ``k`` no, per strategy.

    The positive questions are drawn once and reused across the three
    strategies, so each strategy is a self-contained 50/50 balanced set. That
    is what the official release does, and it is what makes the per-strategy
    accuracy comparable.
    """
    candidates = sorted(present & set(categories))
    if not candidates:
        return []
    n = min(k, len(candidates))
    positives = sorted(rng.sample(candidates, n))

    rows: list[dict] = []
    for strategy in STRATEGIES:
        negatives = negative_objects(
            present, strategy, k=n, frequency=frequency, co=co,
            rng=rng, categories=categories,
        )
        for obj in positives:
            rows.append({
                "image_id": image_id,
                "question": render_question(obj),
                "expected": "yes",
                "strategy": strategy,
                "object": obj,
            })
        for obj in negatives:
            if obj in present:
                raise AssertionError(
                    f"negative object {obj!r} is annotated in image {image_id}; "
                    f"strategy={strategy}. The question set would be unanswerable."
                )
            rows.append({
                "image_id": image_id,
                "question": render_question(obj),
                "expected": "no",
                "strategy": strategy,
                "object": obj,
            })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
        help=f"Image manifest (default: {DEFAULT_MANIFEST}).")
    parser.add_argument("--annotations-dir", type=Path,
        default=DEFAULT_ANNOTATIONS_DIR,
        help="Where instances_val2017.json lives.")
    parser.add_argument("--auto-download", action="store_true",
        help="If instances_val2017.json is missing, download it.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"Where to write pope_questions.jsonl (default: {DEFAULT_OUTPUT}).")
    parser.add_argument("--questions-per-image", type=int, default=3,
        help="Number of yes questions per image; the same number of no "
             "questions is drawn per strategy (default 3).")
    parser.add_argument("--limit", type=int, default=0,
        help="Cap the number of manifest images (0 = all).")
    parser.add_argument("--seed", type=int, default=42,
        help="Seed for the random-strategy sampling and the positive draw.")
    parser.add_argument("--overwrite", action="store_true",
        help="Overwrite an existing output file.")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        logger.error(f"{args.output} exists. Pass --overwrite to rebuild.")
        return 1

    instances_path = args.annotations_dir / "instances_val2017.json"
    if not instances_path.exists():
        if args.auto_download:
            logger.info(f"{instances_path} not found; downloading...")
            ensure_coco_annotations(args.annotations_dir)
        else:
            logger.error(f"{instances_path} not found. Re-run with --auto-download.")
            return 1

    records = read_manifest(args.manifest)
    if args.limit:
        records = records[: args.limit]
    logger.info(f"manifest images  : {len(records)}")

    image_to_objects = load_ground_truth_objects(instances_path)
    logger.info(f"annotated images : {len(image_to_objects)}")

    frequency = object_frequency(image_to_objects)
    co = cooccurrence(image_to_objects)
    logger.info(f"categories seen  : {len(frequency)} / {len(COCO_CATEGORIES)}")

    rng = random.Random(args.seed)
    rows: list[dict] = []
    n_skipped = 0
    for record in records:
        present = image_to_objects.get(record.image_id, set())
        image_rows = questions_for_image(
            record.image_id, present,
            k=args.questions_per_image, frequency=frequency, co=co, rng=rng,
        )
        if not image_rows:
            n_skipped += 1
            logger.warning(f"{record.image_id}: no annotated COCO-80 object; skipped.")
            continue
        rows.extend(image_rows)

    n_yes = sum(1 for r in rows if r["expected"] == "yes")
    n_no = len(rows) - n_yes
    if n_yes != n_no:
        logger.error(f"question set is not balanced: {n_yes} yes vs {n_no} no.")
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info("=" * 70)
    logger.info(f"POPE questions written: {args.output}")
    logger.info(f"  images used   : {len(records) - n_skipped} (skipped {n_skipped})")
    logger.info(f"  questions     : {len(rows)}  ({n_yes} yes / {n_no} no)")
    for strategy in STRATEGIES:
        n = sum(1 for r in rows if r["strategy"] == strategy)
        logger.info(f"  {strategy:<12}: {n}")
    logger.info(f"  seed          : {args.seed}")
    logger.info(f"  built         : {_iso_now()}")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
