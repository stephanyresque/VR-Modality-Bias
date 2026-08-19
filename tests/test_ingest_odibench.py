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


def test_two_components_of_the_same_type_qualify(ingest):
    """Requiring two DISTINCT types rejected 174 of 259 qualifying images:
    view_orientation and relative both feed `direction`, and two direction
    questions on one image is the commonest shape in the set."""
    Candidate = ingest.Candidate
    two_directions = [
        Candidate("i", "direction", "Where is the chair?", "left"),
        Candidate("i", "direction", "Where is the lamp?", "right"),
    ]

    assert ingest.qualifies(two_directions)


def test_two_components_of_different_types_qualify(ingest):
    Candidate = ingest.Candidate

    assert ingest.qualifies([
        Candidate("i", "existence", "q1", "yes"),
        Candidate("i", "count", "q2", "3"),
    ])


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


def _grouped(ingest, spec: dict[str, list[str]]) -> dict[str, list]:
    """``{image_id: [component_type, ...]}`` -> the grouped Candidate dict."""
    Candidate = ingest.Candidate
    return {
        image_id: [
            Candidate(image_id, component_type, f"q{i}", "left")
            for i, component_type in enumerate(types)
        ]
        for image_id, types in spec.items()
    }


def test_more_distinct_types_wins_the_quota_first(ingest):
    """A uniform draw would spend the quota on the majority type: the set has
    643 direction components against 90 existence ones."""
    grouped = _grouped(ingest, {
        "one_type": ["direction", "direction", "direction", "direction"],
        "three_types": ["existence", "count", "direction"],
        "two_types": ["count", "direction"],
    })

    assert ingest.select_images(grouped, limit=1, seed=0) == ["three_types"]
    assert sorted(ingest.select_images(grouped, limit=2, seed=0)) == [
        "three_types", "two_types",
    ]


def test_component_count_breaks_a_tie_between_equal_type_counts(ingest):
    grouped = _grouped(ingest, {
        "two_types_two_components": ["count", "direction"],
        "two_types_four_components": ["count", "direction", "direction", "direction"],
    })

    assert ingest.select_images(grouped, limit=1, seed=0) == [
        "two_types_four_components"
    ]


def test_the_priority_is_not_overridden_by_the_seed(ingest):
    grouped = _grouped(ingest, {
        "rich": ["existence", "count", "direction"],
        **{f"poor_{i}": ["direction", "direction"] for i in range(20)},
    })

    for seed in range(5):
        assert ingest.select_images(grouped, limit=1, seed=seed) == ["rich"], seed


def test_selection_is_deterministic_for_a_seed(ingest):
    grouped = _grouped(ingest, {
        f"indoor_{i:03d}": ["direction", "direction"] for i in range(100)
    })

    first = ingest.select_images(grouped, limit=10, seed=7)
    second = ingest.select_images(grouped, limit=10, seed=7)

    assert first == second
    assert len(first) == 10


def test_a_different_seed_breaks_the_remaining_ties_differently(ingest):
    grouped = _grouped(ingest, {
        f"indoor_{i:03d}": ["direction", "direction"] for i in range(100)
    })

    assert ingest.select_images(grouped, limit=10, seed=1) != \
        ingest.select_images(grouped, limit=10, seed=2)


def test_the_tie_break_is_not_just_the_lowest_indices(ingest):
    grouped = _grouped(ingest, {
        f"indoor_{i:03d}": ["direction", "direction"] for i in range(100)
    })

    chosen = ingest.select_images(grouped, limit=10, seed=7)

    assert chosen != sorted(grouped)[:10], (
        "sorting and cutting biases the selection towards the low indices"
    )


def test_no_limit_keeps_everything(ingest):
    grouped = _grouped(ingest, {
        "b": ["direction", "direction"],
        "a": ["existence", "count"],
        "c": ["count", "direction"],
    })

    assert ingest.select_images(grouped, limit=None, seed=0) == ["a", "b", "c"]


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

    assert [r.image_id for r in manifest] == ["indoor_1"]
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

    # indoor_2 said there is no sofa, so counting the sofas went away, leaving
    # one component -- an ordinary short question, which this track does not
    # measure. It is discarded rather than shipped.
    assert "indoor_2" not in by_id
    assert (out_dir / "images" / "indoor_2.png").exists() is False


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


# ---------------------------------------------------------------- raw keys, measured
#
# These two pin the corrections the audit forced: the real field name, and the
# fact that an empty value is annotation noise rather than a fatal error.


def test_the_real_odibench_image_key_is_recognised(ingest, tmp_path: Path):
    """The dataset writes `imagename`, with no underscore."""
    raw = tmp_path / "raw"
    (raw / "QAs").mkdir(parents=True)
    (raw / "QAs" / "existence.jsonl").write_text(
        json.dumps({
            "imagename": "indoor_1.png",
            "question": "Is there a chair?",
            "answer": "yes",
        }) + "\n",
        encoding="utf-8",
    )

    components = list(ingest.iter_raw_components(raw))

    assert components == [("existence", "indoor_1", "Is there a chair?", "yes")]


def test_an_empty_answer_reaches_the_filter_instead_of_raising(ingest, tmp_path: Path):
    """relative.jsonl carries one. A single such record used to kill the run."""
    raw = _write_raw(tmp_path, {
        "relative.jsonl": [
            _qa("indoor_1.png", "Where is the chair?", "left"),
            _qa("indoor_2.png", "Where is the lamp?", ""),
        ],
    }, [])

    candidates, tally = ingest.select_candidates(
        ingest.iter_raw_components(raw), _DIRECTIONS
    )

    assert [c.image_id for c in candidates] == ["indoor_1"]
    assert tally["bad_answer"] == 1, (
        "an empty answer is noise for the answer filter to count, not a crash"
    )


def test_an_empty_value_does_not_produce_a_self_contradicting_message(ingest):
    """The old message read: no answer field; this record has ['answer']."""
    value = ingest._pick({"answer": ""}, ingest.ANSWER_KEYS, "answer", "f", 1)

    assert value == ""


def test_a_record_missing_every_candidate_key_still_raises(ingest, tmp_path: Path):
    raw = tmp_path / "raw"
    (raw / "QAs").mkdir(parents=True)
    (raw / "QAs" / "existence.jsonl").write_text(
        json.dumps({"imagename": "indoor_1.png", "question": "?"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as excinfo:
        list(ingest.iter_raw_components(raw))

    message = str(excinfo.value)
    assert "no answer field" in message
    assert "imagename" in message, "the message must list the keys actually present"


# ---------------------------------------------------------------- short items


def test_an_item_left_with_one_component_is_discarded(ingest, tmp_path: Path):
    raw = _write_raw(tmp_path, {}, ["indoor_1.png", "indoor_2.png"])
    csv_path = tmp_path / "targets.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(ingest.CSV_COLUMNS))
        writer.writeheader()
        # indoor_1 survives with two components.
        writer.writerow({"image_id": "indoor_1", "component_type": "existence",
                         "question": "Is there a chair?", "answer": "yes",
                         "target": "chair"})
        writer.writerow({"image_id": "indoor_1", "component_type": "count",
                         "question": "How many chairs?", "answer": "3",
                         "target": "chair"})
        # indoor_2 loses its count to the negative-existence rule.
        writer.writerow({"image_id": "indoor_2", "component_type": "existence",
                         "question": "Is there a sofa?", "answer": "no",
                         "target": "sofa"})
        writer.writerow({"image_id": "indoor_2", "component_type": "count",
                         "question": "How many sofas?", "answer": "0",
                         "target": "sofa"})

    out_dir = tmp_path / "processed"
    args = ingest.build_parser().parse_args([
        "--raw", str(raw), "--targets", str(csv_path), "--out", str(out_dir),
    ])
    assert ingest.run_build(args) == 0

    questions = read_question_annotations(out_dir / "questions.jsonl")
    manifest = read_manifest(out_dir / "manifest.jsonl")

    assert [q.image_id for q in questions] == ["indoor_1"], (
        "a one-component item is an ordinary short question, the opposite of "
        "what this track measures"
    )
    assert [r.image_id for r in manifest] == ["indoor_1"], (
        "the manifest must not list an item with no question"
    )


def test_a_csv_row_that_never_had_a_partner_is_discarded_too(ingest, tmp_path: Path):
    raw = _write_raw(tmp_path, {}, ["indoor_1.png"])
    csv_path = tmp_path / "targets.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(ingest.CSV_COLUMNS))
        writer.writeheader()
        writer.writerow({"image_id": "indoor_1", "component_type": "existence",
                         "question": "Is there a chair?", "answer": "yes",
                         "target": "chair"})

    out_dir = tmp_path / "processed"
    args = ingest.build_parser().parse_args([
        "--raw", str(raw), "--targets", str(csv_path), "--out", str(out_dir),
    ])
    assert ingest.run_build(args) == 0

    assert read_question_annotations(out_dir / "questions.jsonl") == []
