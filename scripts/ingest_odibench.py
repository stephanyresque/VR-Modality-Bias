#!/usr/bin/env python
# Run: python scripts/ingest_odibench.py --raw data/raw/odibench --out data/processed/odibench

from __future__ import annotations

import argparse
import json
import random
import re
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

# All ten QA files of the benchmark. The first ingestion (exp 1) read only
# existence/counting and merged view_orientation+relative into "direction";
# the full-evaluation round keeps one component type per source file so the
# report can break accuracy down by the benchmark's own taxonomy.
QA_FILES: dict[str, str] = {
    "existence.jsonl": "existence",
    "counting.jsonl": "count",
    "ocr.jsonl": "ocr",
    "object_attribute.jsonl": "object_attribute",
    "human_attribute.jsonl": "human_attribute",
    "view_orientation.jsonl": "direction_ego",
    "allocentric.jsonl": "direction_allo",
    "relative.jsonl": "direction_rel",
    "scene_simulation.jsonl": "scene_simulation",
    "ODI_reasoning.jsonl": "odi_reasoning",
}

# Object/human attributes came from an automatic pipeline with human
# refinement; the other eight types are fully human-annotated. The report and
# meta.json carry this flag so the paper can signal it.
AUTO_ANNOTATED_TYPES: frozenset[str] = frozenset(
    {"object_attribute", "human_attribute"}
)

DIRECTION_TYPES: frozenset[str] = frozenset(
    {"direction_ego", "direction_allo", "direction_rel"}
)

# Composition order: perception first, then spatial, then reasoning.
TYPE_ORDER: tuple[str, ...] = (
    "existence",
    "count",
    "ocr",
    "object_attribute",
    "human_attribute",
    "direction_ego",
    "direction_allo",
    "direction_rel",
    "scene_simulation",
    "odi_reasoning",
)

IMAGE_KEYS: tuple[str, ...] = (
    "image", "imagename", "image_id", "image_name", "image_path", "img",
    "file_name", "filename",
)
QUESTION_KEYS: tuple[str, ...] = ("question", "text", "query", "prompt")
ANSWER_KEYS: tuple[str, ...] = ("answer", "gt", "ground_truth", "label", "response")

_JUNK_NAMES = {"thumbs.db", "desktop.ini", ".ds_store"}
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

_ARTICLES = ("the ", "a ", "an ")
_GENERIC_NOUNS = {
    "side", "sides", "part", "parts", "direction", "directions", "area", "region",
}

_QUESTION_STEMS = (
    "can you see", "how many", "are there", "is there", "where are", "where is",
    "what are", "what is", "do you see", "are the", "is the",
)
_DETERMINERS = ("the ", "a ", "an ", "any ", "some ", "this ", "these ", "those ")
_HEAD_STOP = {
    "in", "on", "at", "of", "to", "from", "near", "beside", "under", "over",
    "above", "below", "behind", "between", "with", "and", "or", "that", "which",
    "is", "are", "there", "do", "you", "see", "the", "a", "an", "this", "these",
    "it", "its",
}
_PUNCTUATION = re.compile(r"[^\w\s-]")


@dataclass(frozen=True)
class Candidate:
    image_id: str
    component_type: str
    question: str
    answer: str


# ---------------------------------------------------------------- raw reading


def _pick(record: dict, keys: tuple[str, ...], what: str, source: str, lineno: int):
    """Return the first present key's value; raise only if none of them exist.

    Presence, not truthiness: an empty answer is annotation noise and belongs in
    the answer filter alongside the rest of it, not as a fatal error that stops
    the whole ingestion.
    """
    for key in keys:
        if key in record:
            return record[key]
    raise ValueError(
        f"{source}:{lineno}: no {what} field. Tried {list(keys)}; this record "
        f"has {sorted(record)}. Fix the key list at the top of "
        f"scripts/ingest_odibench.py."
    )


def resolve_option(record: dict):
    """Return the option text the ``correct`` field points at, or None.

    Most ODI files are multiple-choice: ``options`` plus a ``correct`` that is
    either a letter (A-D), an index, or the option text itself. The evaluation
    is open-format, so the reference the judge receives is the winning option's
    TEXT, never the letter.
    """
    options = record.get("options")
    correct = record.get("correct")
    if not isinstance(options, (list, tuple)) or not options or correct is None:
        return None
    texts = [str(o).strip() for o in options]
    label = str(correct).strip()
    if len(label) == 1 and label.isalpha():
        index = ord(label.upper()) - ord("A")
        if 0 <= index < len(texts):
            return texts[index]
        return None
    if label.isdigit():
        index = int(label)
        if 0 <= index < len(texts):
            return texts[index]
        return None
    for text in texts:
        if text.lower() == label.lower():
            return text
    return None


def _strip_option_prefix(text: str) -> str:
    return re.sub(r"^[A-Da-d][\.\):]\s+", "", text.strip())


def iter_raw_components(raw_dir: Path):
    """Yield ``(component_type, image_id, question, answer)`` from QAs/*.jsonl.

    Also returns, via the second element of the outer tuple, which QA files
    were actually found: silently narrowing the benchmark because one file is
    missing is exactly the failure the tally exists to expose.
    """
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
                try:
                    answer = _pick(record, ANSWER_KEYS, "answer", str(path), lineno)
                except ValueError:
                    # No answer key at all: a schema error, unless the record
                    # is multiple-choice and the option resolves.
                    answer = resolve_option(record)
                    if answer is None and "options" not in record:
                        raise
                if answer is None or not str(answer).strip():
                    answer = resolve_option(record)
                yield (
                    component_type,
                    Path(str(image_ref)).stem,
                    str(question).strip(),
                    "" if answer is None else str(answer),
                )


def list_missing_qa_files(raw_dir: Path) -> list[str]:
    qa_dir = Path(raw_dir) / "QAs"
    return sorted(
        file_name for file_name in QA_FILES if not (qa_dir / file_name).is_file()
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
    that are not directions at all -- the annotation carries all three. The
    judge does not replace this filter: it decides whether an answer is right,
    not whether the reference was a direction in the first place.
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


def normalize_free_text(answer) -> str | None:
    text = _strip_option_prefix(str(answer))
    return text if text else None


def normalize_answer(component_type: str, answer, directions):
    """Per-type reference normalization.

    existence and count stay strict (same rule as exp 1). The three direction
    types canonicalize through the vocabulary when the answer resolves there
    and otherwise keep the raw text: allocentric references are relational
    sentences by nature, and the judge grades text, so dropping them would
    throw away the type. The remaining types keep the raw (or option-resolved)
    text. Empty always drops.
    """
    if component_type == "existence":
        return normalize_existence(answer)
    if component_type == "count":
        return normalize_count(answer)
    if component_type in DIRECTION_TYPES:
        canonical = normalize_direction(answer, directions)
        if canonical is not None:
            return canonical
        return normalize_free_text(answer)
    if component_type in QA_FILES.values():
        return normalize_free_text(answer)
    raise ValueError(f"unknown component type {component_type!r}")


def select_candidates(
    raw_components, directions, *, image_prefix: str | None = None
) -> tuple[list[Candidate], dict]:
    """Apply every selection filter. Returns the survivors and a tally."""
    tally = {
        "seen": 0, "wrong_prefix": 0, "bad_answer": 0,
        "kept_by_type": defaultdict(int),
        "kept_by_image_prefix": defaultdict(int),
        "direction_canonical": 0, "direction_raw_text": 0,
    }
    kept: list[Candidate] = []
    for component_type, image_id, question, answer in raw_components:
        tally["seen"] += 1
        if image_prefix and not image_id.startswith(image_prefix):
            tally["wrong_prefix"] += 1
            continue
        normalized = normalize_answer(component_type, answer, directions)
        if normalized is None:
            tally["bad_answer"] += 1
            continue
        if component_type in DIRECTION_TYPES:
            if normalize_direction(answer, directions) is not None:
                tally["direction_canonical"] += 1
            else:
                tally["direction_raw_text"] += 1
        kept.append(Candidate(image_id, component_type, question, str(normalized)))
        tally["kept_by_type"][component_type] += 1
        tally["kept_by_image_prefix"][image_id.split("_")[0]] += 1
    return kept, tally


def group_by_image(candidates: list[Candidate]) -> dict[str, list[Candidate]]:
    grouped: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.image_id].append(candidate)
    return grouped


def qualifies(components: list[Candidate]) -> bool:
    """Two verifiable components, of any type.

    Requiring two DISTINCT types rejected 174 of the 259 qualifying images,
    because view_orientation.jsonl and relative.jsonl both feed `direction` and
    two direction questions on one image is the commonest shape in the set.
    """
    return len(components) >= 2


def select_images(
    grouped: dict[str, list[Candidate]], *, limit: int | None, seed: int
) -> list[str]:
    """Pick the images to keep: round-robin over the component types.

    Richest-first alone spends the quota on the majority types; with ten types
    the scarce ones (ocr, odi_reasoning) would land too thin for the per-type
    breakdown the evaluation exists to produce. The round-robin walks the
    types in a fixed order and, at each turn, takes the not-yet-selected image
    that carries that type and is richest (most distinct types, then most
    components; shuffled by seed as the last tiebreak). Every type present in
    the data gets images until it runs out or the quota fills.
    """
    population = sorted(grouped)
    random.Random(seed).shuffle(population)
    population.sort(key=lambda image_id: (
        -len({c.component_type for c in grouped[image_id]}),
        -len(grouped[image_id]),
    ))
    if limit is None or limit >= len(population):
        return sorted(population)

    # Queues come from the types PRESENT in the data (TYPE_ORDER first, then
    # anything else, e.g. the legacy "direction"), so an unexpected type still
    # gets its share instead of being silently unreachable.
    present = sorted(
        {c.component_type for components in grouped.values() for c in components},
        key=_type_rank,
    )
    queues: dict[str, list[str]] = {
        component_type: [
            image_id for image_id in population
            if any(c.component_type == component_type for c in grouped[image_id])
        ]
        for component_type in present
    }
    selected: list[str] = []
    chosen: set[str] = set()
    while len(selected) < limit:
        progressed = False
        for component_type in present:
            if len(selected) >= limit:
                break
            queue = queues[component_type]
            while queue and queue[0] in chosen:
                queue.pop(0)
            if not queue:
                continue
            image_id = queue.pop(0)
            chosen.add(image_id)
            selected.append(image_id)
            progressed = True
        if not progressed:
            break
    return sorted(selected)


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


def _type_rank(component_type: str) -> int:
    # Unknown types (the legacy merged "direction" of old files) go last
    # instead of crashing the ordering.
    try:
        return TYPE_ORDER.index(component_type)
    except ValueError:
        return len(TYPE_ORDER)


def order_components(components: list[Candidate]) -> list[Candidate]:
    return sorted(components, key=lambda c: _type_rank(c.component_type))


def compose_question(components: list[Candidate]) -> str:
    """Concatenate the original statements. They were written by humans;
    rewriting them would put a variable of ours inside their ground truth."""
    return " ".join(c.question for c in components)


def _singular(word: str) -> str:
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("ses", "xes", "zes", "ches", "shes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def heuristic_target(question: str) -> str:
    """The noun a question is about, guessed from its wording.

    Only ``apply_negative_existence_rule`` uses this, and it is never written to
    disk. It replaces the hand-curated target column, so it is allowed to be
    wrong: a wrong target used to mean a wrong verdict, and now the worst case
    is one slightly awkward composed question surviving the filter.
    """
    text = _PUNCTUATION.sub(" ", question.lower())
    text = re.sub(r"\s+", " ", text).strip()

    for stem in _QUESTION_STEMS:
        if text.startswith(stem + " "):
            text = text[len(stem) + 1:]
            break
    while True:
        for determiner in _DETERMINERS:
            if text.startswith(determiner):
                text = text[len(determiner):].strip()
                break
        else:
            break

    words: list[str] = []
    for word in text.split():
        if word in _HEAD_STOP:
            break
        words.append(word)
    while len(words) > 1 and words[-1] in _GENERIC_NOUNS:
        words = words[:-1]
    if not words:
        return ""
    return _singular(words[-1])


def apply_negative_existence_rule(
    ordered: list[Candidate],
) -> tuple[list[Candidate], int]:
    """Drop a component that asks about a target a negative existence just denied.

    Counting or locating something the answer already said is absent is a
    degenerate sub-question, and no judge repairs a degenerate question.
    Returns the kept candidates and how many were dropped.
    """
    denied = {
        heuristic_target(candidate.question)
        for candidate in ordered
        if candidate.component_type == "existence" and candidate.answer == "no"
    }
    denied.discard("")
    if not denied:
        return ordered, 0
    kept = [
        candidate for candidate in ordered
        if candidate.component_type == "existence"
        or heuristic_target(candidate.question) not in denied
    ]
    return kept, len(ordered) - len(kept)


def build_component(candidate: Candidate) -> QuestionComponent:
    answer: str | int = candidate.answer
    if candidate.component_type == "count":
        answer = int(candidate.answer)
    return QuestionComponent(
        component_type=candidate.component_type,
        question=candidate.question,
        answer=answer,
    )


# ---------------------------------------------------------------- run


def plan(args, directions):
    candidates, tally = select_candidates(
        iter_raw_components(args.raw), directions, image_prefix=args.image_prefix
    )
    grouped = group_by_image(candidates)
    qualified = {
        image_id: comps for image_id, comps in grouped.items() if qualifies(comps)
    }
    index, skipped_files = index_images(args.raw)

    # The QA files reference images that images_final does not carry (the HF
    # release ships fewer files than the paper's 2000). Those images are
    # excluded BEFORE the selection, so the quota is spent on items that can
    # actually run, and the exclusion is counted instead of silent. Zero
    # overlap stays fatal below: that is the id-mismatch signature, not an
    # availability gap.
    absent = sorted(image_id for image_id in qualified if image_id not in index)
    qualified = {
        image_id: comps for image_id, comps in qualified.items()
        if image_id in index
    }
    if absent and not qualified:
        raise ValueError(
            f"every qualified image is missing from {args.raw}/images_final "
            f"(e.g. {absent[:5]}). The QA ids and the file names do not match."
        )

    chosen = select_images(qualified, limit=args.limit, seed=args.seed)

    kept: list[tuple[str, list[Candidate]]] = []
    missing_images: list[str] = []
    n_dropped_by_negative = 0
    n_discarded_single_component = 0

    for image_id in chosen:
        if image_id not in index:
            missing_images.append(image_id)
            continue
        ordered, dropped = apply_negative_existence_rule(
            order_components(qualified[image_id])
        )
        n_dropped_by_negative += dropped
        # A one-component item is an ordinary short question, which is the
        # opposite of what this track measures: the answer has to be long
        # because the question asks for several things.
        if len(ordered) < 2:
            n_discarded_single_component += 1
            continue
        kept.append((image_id, ordered))

    components_per_item = defaultdict(int)
    written_by_type = defaultdict(int)
    for _, ordered in kept:
        components_per_item[len(ordered)] += 1
        for candidate in ordered:
            written_by_type[candidate.component_type] += 1

    meta = {
        "dataset": DATASET_NAME,
        "raw": str(args.raw),
        "direction_terms": str(args.direction_terms),
        "seed": args.seed,
        "limit": args.limit,
        "image_prefix": args.image_prefix,
        "qa_files_missing": list_missing_qa_files(args.raw),
        "auto_annotated_types": sorted(AUTO_ANNOTATED_TYPES),
        "components_seen": tally["seen"],
        "dropped_wrong_prefix": tally["wrong_prefix"],
        "dropped_bad_answer": tally["bad_answer"],
        "kept_by_type": dict(tally["kept_by_type"]),
        "kept_by_image_prefix": dict(tally["kept_by_image_prefix"]),
        "direction_canonical": tally["direction_canonical"],
        "direction_raw_text": tally["direction_raw_text"],
        "images_with_any_component": len(grouped),
        "images_qualified": len(qualified),
        "images_qualified_without_file": len(absent),
        "images_qualified_without_file_sample": absent[:10],
        "images_selected": len(chosen),
        "files_skipped_in_images_final": skipped_files,
        "components_dropped_by_negative_existence": n_dropped_by_negative,
        "items_discarded_single_component": n_discarded_single_component,
        "items_written": len(kept),
        "components_per_item": {str(k): v for k, v in sorted(components_per_item.items())},
        "components_written_by_type": dict(written_by_type),
    }
    return kept, meta, missing_images, index


def _report(meta: dict, *, out_dir: Path, dry_run: bool) -> None:
    print("=" * 78)
    print(f"ODI-BENCH INGESTION{'  -- dry run, nothing written' if dry_run else ''}")
    print("=" * 78)
    if meta["qa_files_missing"]:
        print(f"  WARN: QA file(s) absent : {meta['qa_files_missing']}")
        print("        the benchmark is being ingested NARROWER than planned.")
    print(f"  components seen        : {meta['components_seen']}")
    print(f"  image prefix filter    : {meta['image_prefix'] or '(none, all images)'}")
    print(f"  dropped, wrong prefix  : {meta['dropped_wrong_prefix']}")
    print(f"  dropped, bad answer    : {meta['dropped_bad_answer']}")
    for component_type in TYPE_ORDER:
        kept = meta["kept_by_type"].get(component_type, 0)
        auto = "  [auto-annotated]" if component_type in AUTO_ANNOTATED_TYPES else ""
        print(f"  kept, {component_type:<16} : {kept}{auto}")
    print(f"  kept by image prefix   : {meta['kept_by_image_prefix']}")
    print(f"  direction refs         : {meta['direction_canonical']} canonical, "
          f"{meta['direction_raw_text']} raw text")
    print(f"  images with a component: {meta['images_with_any_component']}")
    print(f"  images qualified (>=2) : {meta['images_qualified']}")
    if meta["images_qualified_without_file"]:
        print(f"  qualified but no file  : {meta['images_qualified_without_file']} "
              f"(e.g. {meta['images_qualified_without_file_sample'][:3]})")
    print(f"  images selected (limit={meta['limit']}, seed={meta['seed']}): "
          f"{meta['images_selected']}")
    print(f"  files skipped in images_final (not an image): "
          f"{meta['files_skipped_in_images_final']}")
    print(f"  components dropped by the negative-existence rule: "
          f"{meta['components_dropped_by_negative_existence']}")
    print(f"  items discarded, fewer than 2 components left: "
          f"{meta['items_discarded_single_component']}")
    print(f"  items written          : {meta['items_written']}")
    if not dry_run:
        print(f"  manifest               : {out_dir / 'manifest.jsonl'}")
        print(f"  questions              : {out_dir / 'questions.jsonl'}")
        print(f"  images copied to       : {out_dir / 'images'}")
        print(f"  provenance             : {out_dir / 'meta.json'}")
    print("=" * 78)


def run(args) -> int:
    directions = load_vocabulary(args.direction_terms)
    kept, meta, missing_images, index = plan(args, directions)

    if missing_images:
        print(
            f"ERROR: {len(missing_images)} selected image_id(s) have no file "
            f"under {args.raw}/images_final: {missing_images[:5]}",
            file=sys.stderr,
        )
        return 1

    out_dir = Path(args.out)
    if args.dry_run:
        _report(meta, out_dir=out_dir, dry_run=True)
        return 0

    images_out = out_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    manifest: list[ImageRecord] = []
    questions: list[QuestionAnnotation] = []
    for image_id, ordered in kept:
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
            question_text=compose_question(ordered),
            components=tuple(build_component(c) for c in ordered),
        ))

    write_manifest(manifest, out_dir / "manifest.jsonl")
    write_question_annotations(questions, out_dir / "questions.jsonl")
    (out_dir / "meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )

    _report(meta, out_dir=out_dir, dry_run=False)
    return 0


# ---------------------------------------------------------------- cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Turn raw ODI-Bench into manifest.jsonl + questions.jsonl."
    )
    parser.add_argument("--raw", type=Path, default=Path("data/raw/odibench"),
        help="Raw ODI-Bench root, holding QAs/ and images_final/.")
    parser.add_argument("--out", type=Path, default=Path("data/processed/odibench"),
        help="Output dataset directory.")
    parser.add_argument("--direction-terms", type=Path,
        default=Path("configs/direction_terms.json"),
        help="Direction table. A direction answer is kept only if it resolves "
             "here, which is what rejects composites and noise.")
    parser.add_argument("--limit", type=int, default=None,
        help="Keep this many qualified images, selected by the per-type "
             "round-robin with --seed as the tiebreak.")
    parser.add_argument("--seed", type=int, default=0,
        help="Sampling seed, recorded in meta.json.")
    parser.add_argument("--image-prefix", type=str, default=None,
        help="Keep only images whose id starts with this prefix (e.g. "
             "'indoor', the exp 1 restriction). Default: all images.")
    parser.add_argument("--dry-run", action="store_true",
        help="Print the tally and write nothing.")
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
