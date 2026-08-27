#!/usr/bin/env python
# Run: python scripts/ingest_omnicot.py --raw data/raw/omnicot --out data/processed/omnicot
#      python scripts/ingest_omnicot.py --raw data/raw/omnicot --inspect 3

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
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
except ModuleNotFoundError:
    sys.path.insert(0, str(here()))

    from src.vr_modality_bias.data.annotations import (
        QuestionAnnotation,
        QuestionComponent,
        write_question_annotations,
    )
    from src.vr_modality_bias.data.manifests import ImageRecord, write_manifest


DATASET_NAME = "omnicot"
METADATA_RELPATH = Path("real") / "metadata.jsonl"
IMAGES_RELPATH = Path("real") / "images"

# The six task types of the OmniCoT paper (Table 12), keyed by the labels the
# metadata carries in `type`/`subtype`. Verified against the real file on
# 27/08 (--inspect): the six labels below cover all 1073 records, and the
# per-label counts match the paper exactly (198/170/173/158/197/177). An
# unmapped label is a hard error that prints every distinct label seen — never
# a silent drop of a task type.
TYPE_MAP: dict[str, str] = {
    "viewpoint_transform_identify": "mot",
    "viewpoint_transform_angle": "rac",
    "multi_hop_object": "moi",
    "multi_hop_direction": "mdi",
    "move_translation": "ptm",
    "move_turn_combined": "rtm",
}

TYPE_ORDER: tuple[str, ...] = ("mot", "rac", "moi", "mdi", "ptm", "rtm")

# Part of the method, versioned here on purpose: the official protocol feeds
# the anchor objects (positions and orientations) together with the question,
# so the cardinal directions are grounded. Without this preamble the results
# do not compare with the paper's.
ANCHOR_HEADER: str = (
    "The scene contains the following anchor objects with known positions. "
    "Coordinates are (x, y) with the positive x axis pointing east and the "
    "positive y axis pointing north; the orientation is the direction the "
    "object faces."
)


# Measured on the real file (27/08): the image lives in `file_name`
# ("images/balcony_0002.jpg"), not in `image`. Both are accepted so the
# synthetic fixtures and any future schema drift keep working.
IMAGE_KEYS: tuple[str, ...] = ("file_name", "image")


def image_id_of(record: dict, *, source: str, lineno: int) -> str:
    for key in IMAGE_KEYS:
        value = record.get(key)
        if value:
            return Path(str(value)).stem
    raise ValueError(
        f"{source}:{lineno}: no image field. Tried {list(IMAGE_KEYS)}; this "
        f"record has {sorted(record)}."
    )


def map_component_type(record: dict, *, source: str, lineno: int) -> str:
    for key in ("type", "subtype"):
        label = str(record.get(key, "")).strip()
        if label in TYPE_MAP:
            return TYPE_MAP[label]
    raise KeyError(
        f"{source}:{lineno}: neither type={record.get('type')!r} nor "
        f"subtype={record.get('subtype')!r} is in TYPE_MAP. Add the mapping at "
        f"the top of scripts/ingest_omnicot.py after inspecting the data "
        f"(--inspect)."
    )


def _fmt_number(value) -> str:
    number = float(value)
    if number == int(number):
        return str(int(number))
    return f"{number:.2f}"


def _anchor_line(anchor: dict, *, source: str, lineno: int) -> str:
    name = anchor.get("name") or anchor.get("object") or anchor.get("label")
    if not name:
        raise ValueError(
            f"{source}:{lineno}: anchor without a name field: {anchor!r}. "
            f"Adjust _anchor_line after inspecting the data (--inspect)."
        )
    position = (
        anchor.get("position") or anchor.get("coords")
        or anchor.get("coordinates") or anchor.get("pos")
    )
    if isinstance(position, dict):
        coords = [position.get("x"), position.get("y")]
        if position.get("z") is not None:
            coords.append(position.get("z"))
    elif isinstance(position, (list, tuple)):
        coords = list(position)
    else:
        coords = None
    orientation = (
        anchor.get("orientation") or anchor.get("facing")
        or anchor.get("rotation") or anchor.get("direction")
    )
    parts = [str(name)]
    if coords is not None:
        parts.append(
            "position (" + ", ".join(_fmt_number(c) for c in coords if c is not None) + ")"
        )
    if orientation is not None:
        if isinstance(orientation, (int, float)):
            parts.append(f"facing {_fmt_number(orientation)} degrees")
        else:
            parts.append(f"facing {orientation}")
    if len(parts) == 1:
        raise ValueError(
            f"{source}:{lineno}: anchor {name!r} carries neither position nor "
            f"orientation in a recognized key: {anchor!r}. Adjust _anchor_line "
            f"after inspecting the data (--inspect)."
        )
    return "- " + ": ".join([parts[0], ", ".join(parts[1:])])


def format_anchors(random_objects, *, source: str, lineno: int) -> str:
    """Render the anchor preamble, or an empty string when there are none.

    The structure of `random_objects` was not verifiable before the download;
    every unrecognized shape raises with the record attached, so the template
    is locked against the real data before any generation runs.
    """
    if random_objects is None:
        return ""
    if isinstance(random_objects, str):
        text = random_objects.strip()
        if not text:
            return ""
        try:
            random_objects = json.loads(text)
        except json.JSONDecodeError:
            return ANCHOR_HEADER + "\n" + text
    if isinstance(random_objects, dict):
        random_objects = [random_objects]
    if not isinstance(random_objects, (list, tuple)):
        raise ValueError(
            f"{source}:{lineno}: random_objects has unexpected shape "
            f"{type(random_objects).__name__}: {random_objects!r}."
        )
    if not random_objects:
        return ""
    lines = [
        _anchor_line(anchor, source=source, lineno=lineno)
        if isinstance(anchor, dict)
        else "- " + str(anchor)
        for anchor in random_objects
    ]
    return ANCHOR_HEADER + "\n" + "\n".join(lines)


def compose_question_text(record: dict, *, source: str, lineno: int) -> str:
    question = str(record["question"]).strip()
    preamble = format_anchors(
        record.get("random_objects"), source=source, lineno=lineno
    )
    if not preamble:
        return question
    return preamble + "\n\n" + question


def iter_metadata(raw_dir: Path):
    path = Path(raw_dir) / METADATA_RELPATH
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found.")
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: malformed JSON: {exc}") from exc
            yield lineno, record


def select_image_ids(
    records: list[tuple[int, dict]], *, limit: int | None, seed: int, source: str
) -> set[str]:
    image_ids = sorted({
        image_id_of(r, source=source, lineno=lineno) for lineno, r in records
    })
    if limit is None or limit >= len(image_ids):
        return set(image_ids)
    random.Random(seed).shuffle(image_ids)
    return set(image_ids[:limit])


def build_items(
    records: list[tuple[int, dict]], *, source: str, selected: set[str]
) -> tuple[list[QuestionAnnotation], dict]:
    tally = {
        "seen": 0, "kept": 0, "dropped_empty_answer": 0,
        "kept_by_type": defaultdict(int), "with_anchors": 0,
    }
    seen_ids: set[str] = set()
    items: list[QuestionAnnotation] = []
    for lineno, record in records:
        tally["seen"] += 1
        image_id = image_id_of(record, source=source, lineno=lineno)
        if image_id not in selected:
            continue
        answer = str(record.get("answer", "")).strip()
        if not answer:
            tally["dropped_empty_answer"] += 1
            continue
        component_type = map_component_type(record, source=source, lineno=lineno)
        question_id = str(record.get("qa_id", "")).strip() or f"{image_id}_l{lineno}"
        if question_id in seen_ids:
            question_id = f"{image_id}_{question_id}_l{lineno}"
        seen_ids.add(question_id)
        question_text = compose_question_text(record, source=source, lineno=lineno)
        if question_text != str(record["question"]).strip():
            tally["with_anchors"] += 1
        items.append(QuestionAnnotation(
            image_id=image_id,
            question_id=question_id,
            question_text=question_text,
            components=(QuestionComponent(
                component_type=component_type,
                question=str(record["question"]).strip(),
                answer=answer,
            ),),
        ))
        tally["kept"] += 1
        tally["kept_by_type"][component_type] += 1
    return items, tally


def index_images(raw_dir: Path) -> dict[str, Path]:
    images_dir = Path(raw_dir) / IMAGES_RELPATH
    if not images_dir.is_dir():
        raise FileNotFoundError(f"{images_dir} not found.")
    index: dict[str, Path] = {}
    for path in sorted(images_dir.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.stem in index:
            raise ValueError(
                f"two files share the base name {path.stem!r}:\n"
                f"  {index[path.stem]}\n  {path}"
            )
        index[path.stem] = path
    return index


def inspect(raw_dir: Path, n: int) -> int:
    source = str(Path(raw_dir) / METADATA_RELPATH)
    labels: dict[tuple[str, str], int] = defaultdict(int)
    anchor_shapes: dict[str, int] = defaultdict(int)
    n_empty_anchors = 0
    printed = 0
    for lineno, record in iter_metadata(raw_dir):
        labels[(str(record.get("type")), str(record.get("subtype")))] += 1
        anchors = record.get("random_objects")
        anchor_shapes[type(anchors).__name__] += 1
        if not anchors:
            n_empty_anchors += 1
        if printed < n:
            printed += 1
            print(f"── record {lineno} " + "─" * 50)
            print(json.dumps(record, indent=2, ensure_ascii=False))
            try:
                print("rendered question_text:")
                print(compose_question_text(record, source=source, lineno=lineno))
            except (ValueError, KeyError) as exc:
                print(f"!! anchor rendering failed: {exc}")
            print()
    print("distinct (type, subtype) labels:")
    for (type_label, subtype_label), count in sorted(labels.items()):
        mapped = TYPE_MAP.get(type_label) or TYPE_MAP.get(subtype_label)
        status = mapped or "UNMAPPED — add to TYPE_MAP"
        print(f"  type={type_label!r} subtype={subtype_label!r}: {count}  -> {status}")
    print(f"random_objects container shapes: {dict(anchor_shapes)}")
    print(f"random_objects empty in {n_empty_anchors} record(s); "
          f"non-empty in {sum(labels.values()) - n_empty_anchors}")
    return 0


def run(args) -> int:
    if args.inspect is not None:
        return inspect(args.raw, args.inspect)

    source = str(Path(args.raw) / METADATA_RELPATH)
    records = list(iter_metadata(args.raw))
    selected = select_image_ids(
        records, limit=args.limit, seed=args.seed, source=source
    )
    items, tally = build_items(records, source=source, selected=selected)
    index = index_images(args.raw)

    missing = sorted({i.image_id for i in items} - set(index))
    if missing:
        print(
            f"ERROR: {len(missing)} image id(s) in the metadata have no file "
            f"under {Path(args.raw) / IMAGES_RELPATH}: {missing[:5]}",
            file=sys.stderr,
        )
        return 1

    meta = {
        "dataset": DATASET_NAME,
        "raw": str(args.raw),
        "seed": args.seed,
        "limit": args.limit,
        "records_seen": tally["seen"],
        "dropped_empty_answer": tally["dropped_empty_answer"],
        "items_written": tally["kept"],
        "items_with_anchor_preamble": tally["with_anchors"],
        "kept_by_type": dict(tally["kept_by_type"]),
        "images_selected": len(selected),
        "anchor_header": ANCHOR_HEADER,
    }

    print("=" * 78)
    print(f"OMNICOT-REAL INGESTION{'  -- dry run, nothing written' if args.dry_run else ''}")
    print("=" * 78)
    print(f"  records seen           : {meta['records_seen']}")
    print(f"  dropped, empty answer  : {meta['dropped_empty_answer']}")
    for component_type in TYPE_ORDER:
        print(f"  kept, {component_type:<6}           : "
              f"{meta['kept_by_type'].get(component_type, 0)}")
    print(f"  items with anchors     : {meta['items_with_anchor_preamble']}")
    print(f"  images selected        : {meta['images_selected']}")
    print(f"  items written          : {meta['items_written']}")
    print("=" * 78)

    if args.dry_run:
        return 0

    out_dir = Path(args.out)
    images_out = out_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    manifest: list[ImageRecord] = []
    for image_id in sorted({i.image_id for i in items}):
        source_path = index[image_id]
        destination = images_out / source_path.name
        shutil.copyfile(source_path, destination)
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

    write_manifest(manifest, out_dir / "manifest.jsonl")
    write_question_annotations(items, out_dir / "questions.jsonl")
    (out_dir / "meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(f"  manifest               : {out_dir / 'manifest.jsonl'}")
    print(f"  questions              : {out_dir / 'questions.jsonl'}")
    print(f"  images copied to       : {images_out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Turn raw OmniCoT-Real into manifest.jsonl + questions.jsonl "
                    "(original format: one item per QA, anchors in the prompt)."
    )
    parser.add_argument("--raw", type=Path, default=Path("data/raw/omnicot"),
        help="Raw OmniCoT root, holding real/metadata.jsonl and real/images/.")
    parser.add_argument("--out", type=Path, default=Path("data/processed/omnicot"),
        help="Output dataset directory.")
    parser.add_argument("--limit", type=int, default=None,
        help="Keep the QAs of this many images, sampled with --seed. "
             "Default: all 200.")
    parser.add_argument("--seed", type=int, default=0,
        help="Sampling seed, recorded in meta.json.")
    parser.add_argument("--inspect", type=int, default=None, metavar="N",
        help="Print N raw records, the rendered anchor preamble, and every "
             "distinct type/subtype label; write nothing.")
    parser.add_argument("--dry-run", action="store_true",
        help="Print the tally and write nothing.")
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
