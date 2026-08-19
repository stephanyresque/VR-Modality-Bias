"""Tests for scripts/ingest_odibench.py, on synthetic raw data.

The filters here decide what the whole composed track measures, so each one is
pinned separately. The synthetic records use the field names the script tries
first; the real raw file lives on the DGX and its keys were not verified.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

from vr_modality_bias.data.annotations import read_question_annotations
from vr_modality_bias.data.manifests import read_manifest
from vr_modality_bias.data.vocabulary import load_vocabulary

_SCRIPTS = Path(__file__).parent.parent / "scripts"
_DIRECTIONS = load_vocabulary(
    Path(__file__).parent.parent / "configs" / "direction_terms.json"
)


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"_script_{name}", _SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ingest():
    return _load_script("ingest_odibench")


def _write_raw(tmp_path: Path, qa: dict[str, list[dict]], images: list[str]) -> Path:
    raw = tmp_path / "raw"
    (raw / "QAs").mkdir(parents=True, exist_ok=True)
    (raw / "images_final").mkdir(parents=True, exist_ok=True)
    for file_name, records in qa.items():
        with (raw / "QAs" / file_name).open("w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record) + "\n")
    for name in images:
        Image.new("RGB", (64, 32)).save(raw / "images_final" / name)
    return raw


def _qa(image: str, question: str, answer: str) -> dict:
    return {"image": image, "question": question, "answer": answer}


# ---------------------------------------------------------------- existence


@pytest.mark.parametrize("answer,expected", [
    ("yes", "yes"), ("no", "no"), ("Yes.", "yes"), ("NO.", "no"), (" yes ", "yes"),
])
def test_existence_accepts_yes_and_no_with_a_trailing_period(ingest, answer, expected):
    assert ingest.normalize_existence(answer) == expected


@pytest.mark.parametrize("answer", ["maybe", "there is a chair", "", "2"])
def test_existence_rejects_anything_else(ingest, answer):
    assert ingest.normalize_existence(answer) is None


# ---------------------------------------------------------------- count


@pytest.mark.parametrize("answer,expected", [("3", 3), ("3.", 3), (" 12 ", 12), ("0", 0)])
def test_count_accepts_an_integer_with_a_trailing_period(ingest, answer, expected):
    assert ingest.normalize_count(answer) == expected


@pytest.mark.parametrize("answer", ["three", "a few", "2-3", "", "3 chairs"])
def test_count_rejects_anything_that_is_not_an_integer(ingest, answer):
    assert ingest.normalize_count(answer) is None


# ---------------------------------------------------------------- direction


@pytest.mark.parametrize("answer,expected", [
    ("left", "left"),
    ("Left.", "left"),
    ("the left", "left"),
    ("the left side", "left"),
    ("a right side", "right"),
    ("back", "behind"),
    ("top", "above"),
    ("bottom", "below"),
    ("front", "front"),
])
def test_direction_normalises_to_a_canonical_axis(ingest, answer, expected):
    assert ingest.normalize_direction(answer, _DIRECTIONS) == expected


@pytest.mark.parametrize("answer", [
    "front-left",
    "front left",
    "top-right",
    "The chair is to the left of the table.",
    "unknown",
    "",
])
def test_direction_rejects_composites_sentences_and_noise(ingest, answer):
    assert ingest.normalize_direction(answer, _DIRECTIONS) is None, answer


def test_every_retained_direction_resolves_in_the_table(ingest):
    """The filter is defined by the table, so this cannot drift apart from it."""
    for answer in ("left", "right", "front", "back", "top", "bottom"):
        canonical = ingest.normalize_direction(answer, _DIRECTIONS)
        assert canonical is not None
        assert canonical in _DIRECTIONS.categories


# ---------------------------------------------------------------- selection


def test_only_indoor_images_are_kept(ingest, tmp_path: Path):
    raw = _write_raw(tmp_path, {
        "existence.jsonl": [
            _qa("indoor_1.png", "Is there a chair?", "yes"),
            _qa("outdoor_1.png", "Is there a tree?", "yes"),
        ],
    }, [])

    candidates, tally = ingest.select_candidates(
        ingest.iter_raw_components(raw), _DIRECTIONS
    )

    assert [c.image_id for c in candidates] == ["indoor_1"]
    assert tally["wrong_prefix"] == 1


def test_the_tally_separates_prefix_and_answer_rejections(ingest, tmp_path: Path):
    raw = _write_raw(tmp_path, {
        "counting.jsonl": [
            _qa("indoor_1.png", "How many?", "3"),
            _qa("indoor_1.png", "How many?", "a few"),
            _qa("outdoor_1.png", "How many?", "2"),
        ],
    }, [])

    _, tally = ingest.select_candidates(ingest.iter_raw_components(raw), _DIRECTIONS)

    assert tally["seen"] == 3
    assert tally["wrong_prefix"] == 1
    assert tally["bad_answer"] == 1
    assert tally["kept_by_type"]["count"] == 1


def test_both_direction_files_feed_the_same_type(ingest, tmp_path: Path):
    raw = _write_raw(tmp_path, {
        "view_orientation.jsonl": [_qa("indoor_1.png", "Facing where?", "left")],
        "relative.jsonl": [_qa("indoor_1.png", "Where is it?", "right")],
    }, [])

    candidates, _ = ingest.select_candidates(
        ingest.iter_raw_components(raw), _DIRECTIONS
    )

    assert {c.component_type for c in candidates} == {"direction"}
    assert len(candidates) == 2


# ---------------------------------------------------------------- qualifying


def test_an_image_needs_two_distinct_types(ingest):
    Candidate = ingest.Candidate
    two_of_one_type = [
        Candidate("i", "direction", "q1", "left"),
        Candidate("i", "direction", "q2", "right"),
    ]
    two_types = [
        Candidate("i", "existence", "q1", "yes"),
        Candidate("i", "count", "q2", "3"),
    ]

    assert not ingest.qualifies(two_of_one_type), (
        "two components of the same type are not a composed question"
    )
    assert ingest.qualifies(two_types)


def test_a_single_component_does_not_qualify(ingest):
    assert not ingest.qualifies([ingest.Candidate("i", "existence", "q", "yes")])


# ---------------------------------------------------------------- composition


def test_the_statements_are_concatenated_in_type_order(ingest):
    Candidate = ingest.Candidate
    scrambled = [
        Candidate("i", "direction", "Where is it?", "left"),
        Candidate("i", "existence", "Is there a chair?", "yes"),
        Candidate("i", "count", "How many?", "3"),
    ]

    ordered = ingest.order_components(scrambled)
    text = ingest.compose_question(ordered)

    assert text == "Is there a chair? How many? Where is it?"


def test_the_original_wording_is_never_rewritten(ingest):
    Candidate = ingest.Candidate
    text = ingest.compose_question([
        Candidate("i", "existence", "Is there anyone in the room?", "yes"),
    ])

    assert text == "Is there anyone in the room?"


def test_the_composed_text_carries_no_verbosity_instruction(ingest):
    Candidate = ingest.Candidate
    text = ingest.compose_question([
        Candidate("i", "existence", "Is there a chair?", "yes"),
        Candidate("i", "count", "How many?", "3"),
    ]).lower()

    for banned in ("describe", "detail", "long", "verbose", "thorough"):
        assert banned not in text, banned


# ---------------------------------------------------------------- negative rule


def test_a_negative_existence_drops_a_later_component_on_the_same_target(ingest):
    Candidate = ingest.Candidate
    ordered = [
        (Candidate("i", "existence", "Is there a chair?", "no"), "chair"),
        (Candidate("i", "count", "How many chairs?", "0"), "chair"),
        (Candidate("i", "direction", "Where is the table?", "left"), "table"),
    ]

    kept, dropped = ingest.apply_negative_existence_rule(ordered)

    assert dropped == 1
    assert [c.component_type for c, _ in kept] == ["existence", "direction"]


def test_a_positive_existence_drops_nothing(ingest):
    Candidate = ingest.Candidate
    ordered = [
        (Candidate("i", "existence", "Is there a chair?", "yes"), "chair"),
        (Candidate("i", "count", "How many chairs?", "3"), "chair"),
    ]

    kept, dropped = ingest.apply_negative_existence_rule(ordered)

    assert dropped == 0
    assert len(kept) == 2


def test_a_negative_existence_keeps_components_about_other_targets(ingest):
    Candidate = ingest.Candidate
    ordered = [
        (Candidate("i", "existence", "Is there a sofa?", "no"), "sofa"),
        (Candidate("i", "count", "How many chairs?", "3"), "chair"),
    ]

    kept, dropped = ingest.apply_negative_existence_rule(ordered)

    assert dropped == 0
    assert len(kept) == 2


# ---------------------------------------------------------------- images


def test_two_files_sharing_a_base_name_is_a_hard_error(ingest, tmp_path: Path):
    raw = _write_raw(tmp_path, {}, ["indoor_1.png", "indoor_1.jpg"])

    with pytest.raises(ValueError) as excinfo:
        ingest.index_images(raw)

    message = str(excinfo.value)
    assert "indoor_1" in message
    assert "indoor_1.png" in message and "indoor_1.jpg" in message, (
        "both paths must be named so the collision can be fixed at the source"
    )


def test_os_junk_files_are_ignored(ingest, tmp_path: Path):
    raw = _write_raw(tmp_path, {}, ["indoor_1.png"])
    (raw / "images_final" / ".DS_Store").write_bytes(b"junk")
    (raw / "images_final" / "Thumbs.db").write_bytes(b"junk")
    (raw / "images_final" / "._indoor_1.png").write_bytes(b"junk")

    index, skipped = ingest.index_images(raw)

    assert set(index) == {"indoor_1"}
    assert skipped == 0


def test_a_non_image_file_is_skipped_and_counted(ingest, tmp_path: Path):
    raw = _write_raw(tmp_path, {}, ["indoor_1.png"])
    (raw / "images_final" / "README.txt").write_text("hi", encoding="utf-8")

    index, skipped = ingest.index_images(raw)

    assert set(index) == {"indoor_1"}
    assert skipped == 1


# ---------------------------------------------------------------- sampling


def test_sampling_is_deterministic_for_a_seed(ingest):
    population = [f"indoor_{i:03d}" for i in range(100)]

    first = ingest.sample_images(population, limit=10, seed=7)
    second = ingest.sample_images(population, limit=10, seed=7)

    assert first == second
    assert len(first) == 10


def test_a_different_seed_gives_a_different_subset(ingest):
    population = [f"indoor_{i:03d}" for i in range(100)]

    assert ingest.sample_images(population, limit=10, seed=1) != \
        ingest.sample_images(population, limit=10, seed=2)


def test_sampling_is_not_just_the_lowest_indices(ingest):
    population = [f"indoor_{i:03d}" for i in range(100)]

    chosen = ingest.sample_images(population, limit=10, seed=7)

    assert chosen != population[:10], (
        "sorting and cutting biases the selection towards the low indices"
    )


def test_no_limit_keeps_everything(ingest):
    population = ["b", "a", "c"]

    assert ingest.sample_images(population, limit=None, seed=0) == ["a", "b", "c"]


# ---------------------------------------------------------------- csv


def test_a_row_without_a_target_is_ignored_and_counted(ingest, tmp_path: Path):
    path = tmp_path / "targets.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(ingest.CSV_COLUMNS))
        writer.writeheader()
        writer.writerow({"image_id": "indoor_1", "component_type": "existence",
                         "question": "Is there a chair?", "answer": "yes",
                         "target": "chair"})
        writer.writerow({"image_id": "indoor_1", "component_type": "count",
                         "question": "How many?", "answer": "3", "target": ""})
        writer.writerow({"image_id": "indoor_1", "component_type": "direction",
                         "question": "Where?", "answer": "left", "target": "   "})

    rows, untargeted = ingest.read_targets_csv(path)

    assert len(rows) == 1
    assert untargeted == 2


def test_a_csv_missing_a_column_is_rejected(ingest, tmp_path: Path):
    path = tmp_path / "targets.csv"
    path.write_text("image_id,question\nindoor_1,q\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing column"):
        ingest.read_targets_csv(path)


# ---------------------------------------------------------------- end to end


def test_emit_then_build_produces_a_usable_dataset(ingest, tmp_path: Path):
    raw = _write_raw(tmp_path, {
        "existence.jsonl": [
            _qa("indoor_1.png", "Is there a chair?", "yes"),
            _qa("indoor_2.png", "Is there a sofa?", "no"),
            _qa("outdoor_9.png", "Is there a tree?", "yes"),
        ],
        "counting.jsonl": [
            _qa("indoor_1.png", "How many chairs are there?", "3"),
            _qa("indoor_2.png", "How many sofas are there?", "0"),
        ],
        "relative.jsonl": [
            _qa("indoor_1.png", "Where is the chair?", "the left side"),
            _qa("indoor_2.png", "Where is the lamp?", "front-left"),
        ],
    }, ["indoor_1.png", "indoor_2.png", "outdoor_9.png"])

    csv_path = tmp_path / "targets.csv"
    args = ingest.build_parser().parse_args([
        "--mode", "emit", "--raw", str(raw), "--out", str(csv_path),
        "--direction-terms",
        str(Path(__file__).parent.parent / "configs" / "direction_terms.json"),
    ])
    assert ingest.run_emit(args) == 0

    with csv_path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    # front-left was dropped by the direction filter; outdoor was dropped by the
    # prefix filter. indoor_1 keeps 3 components, indoor_2 keeps 2.
    assert {r["image_id"] for r in rows} == {"indoor_1", "indoor_2"}
    assert len(rows) == 5

    targets = {"indoor_1": "chair", "indoor_2": "sofa"}
    for row in rows:
        row["target"] = targets[row["image_id"]]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(ingest.CSV_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)

    out_dir = tmp_path / "processed"
    args = ingest.build_parser().parse_args([
        "--raw", str(raw), "--targets", str(csv_path), "--out", str(out_dir),
    ])
    assert ingest.run_build(args) == 0

    manifest = read_manifest(out_dir / "manifest.jsonl")
    questions = read_question_annotations(out_dir / "questions.jsonl")

    assert [r.image_id for r in manifest] == ["indoor_1", "indoor_2"]
    assert all(r.width == 64 and r.height == 32 for r in manifest), (
        "the manifest records the real pixel size, and nothing is resized"
    )
    assert all(r.dataset == "odibench" for r in manifest)
    assert (out_dir / "images" / "indoor_1.png").is_file()
    assert (raw / "images_final" / "indoor_1.png").is_file(), "raw stays untouched"

    by_id = {q.image_id: q for q in questions}
    assert by_id["indoor_1"].question_text == (
        "Is there a chair? How many chairs are there? Where is the chair?"
    )
    assert [c.component_type for c in by_id["indoor_1"].components] == [
        "existence", "count", "direction",
    ]
    assert by_id["indoor_1"].components[1].answer == 3

    # indoor_2 said there is no sofa, so counting the sofas goes away.
    assert [c.component_type for c in by_id["indoor_2"].components] == ["existence"]


def test_build_fails_when_a_targeted_image_has_no_file(ingest, tmp_path: Path):
    raw = _write_raw(tmp_path, {}, ["indoor_1.png"])
    csv_path = tmp_path / "targets.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(ingest.CSV_COLUMNS))
        writer.writeheader()
        writer.writerow({"image_id": "indoor_missing", "component_type": "existence",
                         "question": "Is there a chair?", "answer": "yes",
                         "target": "chair"})

    args = ingest.build_parser().parse_args([
        "--raw", str(raw), "--targets", str(csv_path),
        "--out", str(tmp_path / "processed"),
    ])

    assert ingest.run_build(args) == 1


def test_the_emit_run_records_its_seed_and_counts(ingest, tmp_path: Path):
    raw = _write_raw(tmp_path, {
        "existence.jsonl": [_qa("indoor_1.png", "Is there a chair?", "yes")],
        "counting.jsonl": [_qa("indoor_1.png", "How many?", "3")],
    }, [])
    csv_path = tmp_path / "targets.csv"

    args = ingest.build_parser().parse_args([
        "--mode", "emit", "--raw", str(raw), "--out", str(csv_path),
        "--seed", "42", "--limit", "1",
        "--direction-terms",
        str(Path(__file__).parent.parent / "configs" / "direction_terms.json"),
    ])
    ingest.run_emit(args)

    meta = json.loads(
        (tmp_path / "targets.csv.meta.json").read_text(encoding="utf-8")
    )
    assert meta["seed"] == 42
    assert meta["limit"] == 1
    assert meta["images_qualified"] == 1
    assert meta["kept_by_type"] == {"existence": 1, "count": 1}


# ---------------------------------------------------------------- raw keys


def test_an_unrecognised_record_shape_fails_naming_the_keys(ingest, tmp_path: Path):
    raw = _write_raw(tmp_path, {
        "existence.jsonl": [{"img_file": "indoor_1.png", "q": "?", "a": "yes"}],
    }, [])

    with pytest.raises(ValueError) as excinfo:
        list(ingest.iter_raw_components(raw))

    message = str(excinfo.value)
    assert "img_file" in message, "the message must show the keys actually present"
    assert "existence.jsonl:1" in message
