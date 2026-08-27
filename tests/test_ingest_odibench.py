"""Tests for scripts/ingest_odibench.py, on synthetic raw data.

The filters here decide what the whole composed track measures, so each one is
pinned separately. The synthetic records use the field names the script tries
first; the real raw file lives on the DGX and its keys were not verified.
"""

from __future__ import annotations

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


def test_all_image_prefixes_are_kept_by_default(ingest, tmp_path: Path):
    """The exp 1 ingestion was indoor-only; the full evaluation keeps both
    halves of the benchmark, and the filter became opt-in."""
    raw = _write_raw(tmp_path, {
        "existence.jsonl": [
            _qa("indoor_1.png", "Is there a chair?", "yes"),
            _qa("outdoor_1.png", "Is there a tree?", "yes"),
        ],
    }, [])

    candidates, tally = ingest.select_candidates(
        ingest.iter_raw_components(raw), _DIRECTIONS
    )

    assert sorted(c.image_id for c in candidates) == ["indoor_1", "outdoor_1"]
    assert tally["wrong_prefix"] == 0
    assert tally["kept_by_image_prefix"] == {"indoor": 1, "outdoor": 1}


def test_an_explicit_prefix_restores_the_exp1_restriction(ingest, tmp_path: Path):
    raw = _write_raw(tmp_path, {
        "existence.jsonl": [
            _qa("indoor_1.png", "Is there a chair?", "yes"),
            _qa("outdoor_1.png", "Is there a tree?", "yes"),
        ],
    }, [])

    candidates, tally = ingest.select_candidates(
        ingest.iter_raw_components(raw), _DIRECTIONS, image_prefix="indoor"
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

    _, tally = ingest.select_candidates(
        ingest.iter_raw_components(raw), _DIRECTIONS, image_prefix="indoor"
    )

    assert tally["seen"] == 3
    assert tally["wrong_prefix"] == 1
    assert tally["bad_answer"] == 1
    assert tally["kept_by_type"]["count"] == 1


def test_the_direction_files_split_by_reference_frame(ingest, tmp_path: Path):
    """exp 1 merged view_orientation and relative into one `direction`; the
    full evaluation keeps one type per source file so the report separates the
    reference frames the ego/allocentric audit worried about."""
    raw = _write_raw(tmp_path, {
        "view_orientation.jsonl": [_qa("indoor_1.png", "Facing where?", "left")],
        "allocentric.jsonl": [_qa("indoor_1.png", "Left of the sofa?", "left")],
        "relative.jsonl": [_qa("indoor_1.png", "Where is it?", "right")],
    }, [])

    candidates, _ = ingest.select_candidates(
        ingest.iter_raw_components(raw), _DIRECTIONS
    )

    assert {c.component_type for c in candidates} == {
        "direction_ego", "direction_allo", "direction_rel",
    }


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


# ---------------------------------------------------------------- heuristic target
#
# The curated target column is gone, so the negative-existence rule guesses the
# noun from the wording. It is allowed to be wrong: a wrong target used to mean
# a wrong verdict, and now the worst case is one awkward composed question
# surviving the filter.


@pytest.mark.parametrize("question,expected", [
    ("Is there a chair?", "chair"),
    ("Is there a chair in the room?", "chair"),
    ("Are there any chairs?", "chair"),
    ("How many chairs?", "chair"),
    ("How many chairs are there?", "chair"),
    ("How many boxes are there?", "box"),
    ("Where is the table?", "table"),
    ("Where are the lamps?", "lamp"),
    ("Is the window to the left of the door?", "window"),
    ("Is there anyone in the room?", "anyone"),
])
def test_the_target_is_guessed_from_the_wording(ingest, question, expected):
    assert ingest.heuristic_target(question) == expected


def test_an_unparseable_question_yields_no_target(ingest):
    assert ingest.heuristic_target("?") == ""


def test_a_question_leading_with_a_modifier_is_a_known_miss(ingest):
    """Pinned as a limitation, not as correct behaviour.

    The guess takes the head of the first noun phrase, so a question that opens
    on a modifier picks the modifier. It is left wrong on purpose: the fix is
    syntactic parsing, and the cost of the error is one composed question that
    reads a little awkwardly, not a wrong verdict.
    """
    assert ingest.heuristic_target("Where is the left side of the sofa?") == "left"


def test_the_singular_and_plural_wordings_agree(ingest):
    assert (
        ingest.heuristic_target("Is there a sofa?")
        == ingest.heuristic_target("How many sofas are there?")
    ), "the rule only fires when the two wordings resolve to the same noun"


# ---------------------------------------------------------------- negative rule


def test_a_negative_existence_drops_a_later_component_on_the_same_target(ingest):
    Candidate = ingest.Candidate
    ordered = [
        Candidate("i", "existence", "Is there a chair?", "no"),
        Candidate("i", "count", "How many chairs are there?", "0"),
        Candidate("i", "direction", "Where is the table?", "left"),
    ]

    kept, dropped = ingest.apply_negative_existence_rule(ordered)

    assert dropped == 1
    assert [c.component_type for c in kept] == ["existence", "direction"]


def test_a_positive_existence_drops_nothing(ingest):
    Candidate = ingest.Candidate
    ordered = [
        Candidate("i", "existence", "Is there a chair?", "yes"),
        Candidate("i", "count", "How many chairs are there?", "3"),
    ]

    kept, dropped = ingest.apply_negative_existence_rule(ordered)

    assert dropped == 0
    assert len(kept) == 2


def test_a_negative_existence_keeps_components_about_other_targets(ingest):
    Candidate = ingest.Candidate
    ordered = [
        Candidate("i", "existence", "Is there a sofa?", "no"),
        Candidate("i", "count", "How many chairs are there?", "3"),
    ]

    kept, dropped = ingest.apply_negative_existence_rule(ordered)

    assert dropped == 0
    assert len(kept) == 2


def test_a_question_with_no_guessable_target_is_never_denied(ingest):
    Candidate = ingest.Candidate
    ordered = [
        Candidate("i", "existence", "?", "no"),
        Candidate("i", "count", "?", "3"),
    ]

    kept, dropped = ingest.apply_negative_existence_rule(ordered)

    assert dropped == 0, (
        "an empty guess must not match every other empty guess and wipe the item"
    )
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


# ---------------------------------------------------------------- end to end


def _full_raw(tmp_path: Path) -> Path:
    return _write_raw(tmp_path, {
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


def _run_ingest(ingest, raw: Path, out_dir: Path, *extra) -> int:
    args = ingest.build_parser().parse_args([
        "--raw", str(raw), "--out", str(out_dir),
        "--direction-terms",
        str(Path(__file__).parent.parent / "configs" / "direction_terms.json"),
        *extra,
    ])
    return ingest.run(args)


def test_one_pass_produces_a_usable_dataset(ingest, tmp_path: Path):
    raw = _full_raw(tmp_path)
    out_dir = tmp_path / "processed"

    assert _run_ingest(ingest, raw, out_dir) == 0

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
        "existence", "count", "direction_rel",
    ]
    assert by_id["indoor_1"].components[1].answer == 3

    # indoor_2 said there is no sofa, so counting the sofas went away. The
    # composite direction answer ("front-left") is kept as raw text now that
    # the judge grades free text, so the item survives with two components.
    assert [c.component_type for c in by_id["indoor_2"].components] == [
        "existence", "direction_rel",
    ]
    assert by_id["indoor_2"].components[1].answer == "front-left"

    # outdoor_9 has a single component: an ordinary short question, which this
    # track does not measure. It is discarded rather than shipped.
    assert "outdoor_9" not in by_id
    assert (out_dir / "images" / "outdoor_9.png").exists() is False


def test_no_curated_csv_is_read_or_written(ingest, tmp_path: Path):
    raw = _full_raw(tmp_path)
    out_dir = tmp_path / "processed"

    _run_ingest(ingest, raw, out_dir)

    assert list(out_dir.glob("*.csv")) == [], (
        "the target CSV existed only for manual curation, which is gone"
    )


def test_every_image_missing_is_fatal_as_an_id_mismatch(ingest, tmp_path: Path):
    """A missing file here or there is an availability gap (excluded and
    counted); EVERY qualified image missing means the ids do not match the
    file names, and that still stops the run."""
    raw = _write_raw(tmp_path, {
        "existence.jsonl": [_qa("indoor_missing.png", "Is there a chair?", "yes")],
        "counting.jsonl": [_qa("indoor_missing.png", "How many lamps are there?", "3")],
    }, ["indoor_1.png"])

    with pytest.raises(ValueError, match="do not match"):
        _run_ingest(ingest, raw, tmp_path / "processed")


def test_the_run_records_its_seed_and_counts(ingest, tmp_path: Path):
    raw = _write_raw(tmp_path, {
        "existence.jsonl": [_qa("indoor_1.png", "Is there a chair?", "yes")],
        "counting.jsonl": [_qa("indoor_1.png", "How many lamps are there?", "3")],
    }, ["indoor_1.png"])
    out_dir = tmp_path / "processed"

    assert _run_ingest(ingest, raw, out_dir, "--seed", "42", "--limit", "1") == 0

    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))

    assert meta["seed"] == 42
    assert meta["limit"] == 1
    assert meta["images_qualified"] == 1
    assert meta["kept_by_type"] == {"existence": 1, "count": 1}
    assert meta["items_written"] == 1


def test_the_meta_counts_what_the_negative_rule_dropped(ingest, tmp_path: Path):
    raw = _full_raw(tmp_path)
    out_dir = tmp_path / "processed"

    _run_ingest(ingest, raw, out_dir)
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))

    assert meta["components_dropped_by_negative_existence"] == 1, (
        "indoor_2 denies the sofa, so counting the sofas goes"
    )
    # indoor_2 keeps two components (the composite direction answer survives
    # as raw text), and outdoor_9 never qualifies, so nothing is discarded
    # AFTER the negative rule.
    assert meta["items_discarded_single_component"] == 0


# ---------------------------------------------------------------- dry run


def test_a_dry_run_writes_nothing(ingest, tmp_path: Path):
    raw = _full_raw(tmp_path)
    out_dir = tmp_path / "processed"

    assert _run_ingest(ingest, raw, out_dir, "--dry-run") == 0

    assert not out_dir.exists(), "a dry run must not even create the directory"


def test_a_dry_run_reports_the_same_tally(ingest, tmp_path: Path, capsys):
    raw = _full_raw(tmp_path)

    _run_ingest(ingest, raw, tmp_path / "processed", "--dry-run")
    dry = capsys.readouterr().out

    _run_ingest(ingest, raw, tmp_path / "processed")
    wet = capsys.readouterr().out

    for line in ("components seen", "items written", "negative-existence"):
        assert line in dry, line
        assert line in wet, line
    assert "dry run, nothing written" in dry
    assert "dry run" not in wet


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
    raw = _write_raw(tmp_path, {
        # indoor_1 survives with two components.
        "existence.jsonl": [
            _qa("indoor_1.png", "Is there a chair?", "yes"),
            _qa("indoor_2.png", "Is there a sofa?", "no"),
        ],
        # indoor_2 loses its count to the negative-existence rule.
        "counting.jsonl": [
            _qa("indoor_1.png", "How many chairs are there?", "3"),
            _qa("indoor_2.png", "How many sofas are there?", "0"),
        ],
    }, ["indoor_1.png", "indoor_2.png"])
    out_dir = tmp_path / "processed"

    assert _run_ingest(ingest, raw, out_dir) == 0

    questions = read_question_annotations(out_dir / "questions.jsonl")
    manifest = read_manifest(out_dir / "manifest.jsonl")

    assert [q.image_id for q in questions] == ["indoor_1"], (
        "a one-component item is an ordinary short question, the opposite of "
        "what this track measures"
    )
    assert [r.image_id for r in manifest] == ["indoor_1"], (
        "the manifest must not list an item with no question"
    )


def test_an_image_with_a_single_component_never_reaches_the_output(
    ingest, tmp_path: Path
):
    raw = _write_raw(tmp_path, {
        "existence.jsonl": [_qa("indoor_1.png", "Is there a chair?", "yes")],
    }, ["indoor_1.png"])
    out_dir = tmp_path / "processed"

    assert _run_ingest(ingest, raw, out_dir) == 0

    assert read_question_annotations(out_dir / "questions.jsonl") == []


# ---------------------------------------------------------------- full set (10 types)


def test_every_qa_file_maps_to_its_own_component_type(ingest):
    assert len(set(ingest.QA_FILES.values())) == len(ingest.QA_FILES) == 10


def test_a_multiple_choice_answer_resolves_from_the_correct_letter(ingest):
    record = {"options": ["red", "green", "blue"], "correct": "B"}
    assert ingest.resolve_option(record) == "green"


def test_a_multiple_choice_answer_resolves_from_an_index(ingest):
    record = {"options": ["red", "green"], "correct": "1"}
    assert ingest.resolve_option(record) == "green"


def test_a_multiple_choice_answer_resolves_from_the_option_text(ingest):
    record = {"options": ["Red", "Green"], "correct": "green"}
    assert ingest.resolve_option(record) == "Green"


@pytest.mark.parametrize("record", [
    {"options": ["red"], "correct": "E"},
    {"options": [], "correct": "A"},
    {"options": ["red"]},
    {"correct": "A"},
])
def test_an_unresolvable_option_returns_none(ingest, record):
    assert ingest.resolve_option(record) is None


def test_a_record_with_options_and_no_answer_key_uses_the_correct_option(
    ingest, tmp_path: Path
):
    raw = tmp_path / "raw"
    (raw / "QAs").mkdir(parents=True)
    (raw / "QAs" / "ocr.jsonl").write_text(
        json.dumps({
            "imagename": "indoor_1.png",
            "question": "What does the sign say?",
            "options": ["EXIT", "OPEN"],
            "correct": "A",
        }) + "\n",
        encoding="utf-8",
    )

    components = list(ingest.iter_raw_components(raw))

    assert components == [("ocr", "indoor_1", "What does the sign say?", "EXIT")]


def test_an_attribute_reference_keeps_the_raw_text(ingest):
    assert ingest.normalize_answer(
        "object_attribute", "B. dark brown leather", _DIRECTIONS
    ) == "dark brown leather"


def test_a_direction_that_resolves_is_canonicalised(ingest):
    assert ingest.normalize_answer("direction_ego", "the left side", _DIRECTIONS) == "left"


def test_a_direction_that_does_not_resolve_survives_as_raw_text(ingest):
    """The allocentric references are relational sentences by nature; dropping
    whatever the vocabulary cannot canonicalise would throw the type away."""
    kept = ingest.normalize_answer(
        "direction_allo", "to the left of the sofa", _DIRECTIONS
    )
    assert kept == "to the left of the sofa"


def test_missing_qa_files_are_listed_not_silently_skipped(ingest, tmp_path: Path):
    raw = _write_raw(tmp_path, {
        "existence.jsonl": [_qa("indoor_1.png", "Is there a chair?", "yes")],
    }, [])

    missing = ingest.list_missing_qa_files(raw)

    assert "ocr.jsonl" in missing
    assert "existence.jsonl" not in missing
    assert len(missing) == 9


def test_the_round_robin_covers_the_scarce_type(ingest):
    """80 images carry only the majority type and 2 carry the scarce one;
    richest-first alone would spend a limit of 10 before reaching them."""
    spec = {f"indoor_{i:03d}": ["direction_ego", "direction_ego"] for i in range(80)}
    spec["indoor_900"] = ["ocr", "ocr"]
    spec["indoor_901"] = ["ocr", "ocr"]
    grouped = _grouped(ingest, spec)

    chosen = ingest.select_images(grouped, limit=10, seed=0)

    assert "indoor_900" in chosen and "indoor_901" in chosen
    assert len(chosen) == 10


def test_the_round_robin_still_fills_the_quota_when_a_type_runs_out(ingest):
    spec = {f"indoor_{i:03d}": ["direction_ego"] * 2 for i in range(20)}
    spec["indoor_900"] = ["ocr", "ocr"]
    grouped = _grouped(ingest, spec)

    chosen = ingest.select_images(grouped, limit=10, seed=0)

    assert len(chosen) == 10


# ---------------------------------------------------------------- missing files


def test_a_qualified_image_without_a_file_is_excluded_before_selection(
    ingest, tmp_path: Path
):
    """The HF release carries fewer images than the QA files reference; the
    gap is an availability fact to count, never a fatal error, and the quota
    must be spent on images that can actually run."""
    raw = _write_raw(tmp_path, {
        "existence.jsonl": [
            _qa("indoor_1.png", "Is there a chair?", "yes"),
            _qa("outdoor_9.png", "Is there a tree?", "yes"),
        ],
        "counting.jsonl": [
            _qa("indoor_1.png", "How many chairs are there?", "3"),
            _qa("outdoor_9.png", "How many trees are there?", "2"),
        ],
    }, ["indoor_1.png"])
    out_dir = tmp_path / "processed"

    assert _run_ingest(ingest, raw, out_dir, "--limit", "2") == 0

    questions = read_question_annotations(out_dir / "questions.jsonl")
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))

    assert [q.image_id for q in questions] == ["indoor_1"]
    assert meta["images_qualified_without_file"] == 1
    assert meta["images_qualified_without_file_sample"] == ["outdoor_9"]


def test_no_qualified_image_having_a_file_is_the_id_mismatch_signature(
    ingest, tmp_path: Path
):
    raw = _write_raw(tmp_path, {
        "existence.jsonl": [_qa("indoor_1.png", "Is there a chair?", "yes")],
        "counting.jsonl": [_qa("indoor_1.png", "How many chairs?", "3")],
    }, ["completely_different_name.png"])

    with pytest.raises(ValueError) as excinfo:
        _run_ingest(ingest, raw, tmp_path / "processed")

    assert "do not match" in str(excinfo.value)
