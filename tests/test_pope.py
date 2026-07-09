"""Tests for the POPE pipeline: question construction, answer normalisation,
metrics, and the SPARC-condition wiring of the two CLIs.

The generation loop itself needs a GPU and a checkpoint, so the scripts keep
their pure logic in module-level functions and only those are exercised here.
"""

from __future__ import annotations

import importlib.util
import json
import math
import random
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from vr_modality_bias.data.prompts import get_prompt
from vr_modality_bias.metrics.pope import (
    ANSWER_INVALID,
    ANSWER_NO,
    ANSWER_YES,
    compute_pope_metrics,
    normalize_answer,
)

_SCRIPTS = Path(__file__).parent.parent / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"_script_{name}", _SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def build_pope():
    return _load_script("build_pope")


@pytest.fixture(scope="module")
def pope_generate():
    return _load_script("pope_generate")


@pytest.fixture(scope="module")
def pope_report():
    return _load_script("pope_report")


@pytest.fixture(scope="module")
def phase3():
    return _load_script("phase3_generate")


# ---------------------------------------------------------------- answer normalisation


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Yes", ANSWER_YES),
        ("yes.", ANSWER_YES),
        ("  YES  ", ANSWER_YES),
        ("yes, there is a dog", ANSWER_YES),
        ("No", ANSWER_NO),
        ("no.", ANSWER_NO),
        ("No, there is not", ANSWER_NO),
        ("Maybe", ANSWER_INVALID),
        ("", ANSWER_INVALID),
        ("   ", ANSWER_INVALID),
        ("123", ANSWER_INVALID),
        ("...", ANSWER_INVALID),
        (None, ANSWER_INVALID),
    ],
)
def test_normalize_answer(raw, expected):
    assert normalize_answer(raw) == expected


def test_normalize_answer_does_not_guess_from_a_buried_no():
    """The official POPE evaluator folds this into "no". We refuse to guess.

    "There is no dog" leads with "There", so it is invalid and gets reported
    apart instead of quietly becoming a correct negative prediction.
    """
    assert normalize_answer("There is no dog in the image") == ANSWER_INVALID


# ---------------------------------------------------------------- metrics


def _record(expected: str, answer: str) -> dict:
    return {"expected": expected, "answer": answer}


def test_compute_pope_metrics_hand_computed():
    records = [
        _record("yes", "yes"),  # TP
        _record("yes", "yes"),  # TP
        _record("yes", "no"),  # FN
        _record("no", "yes"),  # FP
        _record("no", "no"),  # TN
        _record("no", "no"),  # TN
        _record("no", "invalid"),  # excluded
    ]
    m = compute_pope_metrics(records)
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (2, 1, 2, 1)
    assert m["n_total"] == 7
    assert m["n_valid"] == 6
    assert m["n_invalid"] == 1
    assert m["pct_invalid"] == pytest.approx(100 * 1 / 7)
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["yes_ratio"] == pytest.approx(0.5)


def test_compute_pope_metrics_yes_ratio_detects_an_all_yes_answerer():
    records = [_record("yes", "yes"), _record("no", "yes")]
    m = compute_pope_metrics(records)
    assert m["yes_ratio"] == 1.0
    assert m["accuracy"] == 0.5
    assert m["recall"] == 1.0
    assert m["precision"] == 0.5


def test_compute_pope_metrics_is_nan_on_empty_input():
    m = compute_pope_metrics([])
    for key in ("accuracy", "precision", "recall", "f1", "yes_ratio"):
        assert math.isnan(m[key])
    assert m["n_total"] == 0


def test_compute_pope_metrics_is_nan_when_every_answer_is_invalid():
    m = compute_pope_metrics([_record("yes", "invalid"), _record("no", "invalid")])
    assert math.isnan(m["accuracy"])
    assert m["n_valid"] == 0
    assert m["n_invalid"] == 2
    assert m["pct_invalid"] == 100.0


def test_compute_pope_metrics_zero_denominator_is_zero_not_nan():
    """No 'yes' prediction at all: precision has a zero denominator."""
    m = compute_pope_metrics([_record("yes", "no"), _record("no", "no")])
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0
    assert m["yes_ratio"] == 0.0
    assert m["accuracy"] == 0.5


# ---------------------------------------------------------------- question construction

# Toy COCO: "boat" is the most frequent category overall but never co-occurs
# with "dog", while "car" does. That is what pulls `popular` and `adversarial`
# apart for an image whose only object is a dog.
CATS = ("bird", "boat", "car", "cat", "dog", "person")
IMAGES = {
    "img1": {"dog"},
    "img2": {"person", "car"},
    "img3": {"person", "dog", "car"},
    "img4": {"cat"},
    "img5": {"boat"},
    "img6": {"boat"},
    "img7": {"boat"},
    "img8": {"boat"},
}


@pytest.fixture()
def toy_stats(build_pope):
    frequency = build_pope.object_frequency(IMAGES)
    co = build_pope.cooccurrence(IMAGES)
    return frequency, co


def test_render_question_matches_the_official_template():
    assert get_prompt("vqa_pope") == "Is there a {object} in the image? Please answer yes or no."


def test_render_question_fills_the_placeholder(build_pope):
    expected = "Is there a dog in the image? Please answer yes or no."
    assert build_pope.render_question("dog") == expected


def test_object_frequency_counts_images_not_annotations(build_pope):
    frequency = build_pope.object_frequency(IMAGES)
    assert frequency == Counter({"boat": 4, "person": 2, "car": 2, "dog": 2, "cat": 1})
    assert frequency["bird"] == 0


def test_cooccurrence_is_symmetric_and_excludes_self(build_pope, toy_stats):
    _, co = toy_stats
    assert co["dog"]["car"] == 1  # img3 only
    assert co["car"]["dog"] == 1
    assert co["person"]["car"] == 2  # img2 and img3
    assert co["dog"]["dog"] == 0
    assert co["dog"]["boat"] == 0


def test_negative_strategies_disagree(build_pope, toy_stats):
    frequency, co = toy_stats
    rng = random.Random(0)
    present = {"dog"}

    popular = build_pope.negative_objects(
        present, "popular", k=1, frequency=frequency, co=co, rng=rng, categories=CATS
    )
    adversarial = build_pope.negative_objects(
        present, "adversarial", k=1, frequency=frequency, co=co, rng=rng, categories=CATS
    )
    # popular picks the globally most frequent absent object...
    assert popular == ["boat"]
    # ...adversarial picks the one that co-occurs with the dog.
    assert adversarial == ["car"]


def test_negative_objects_never_returns_a_present_object(build_pope, toy_stats):
    frequency, co = toy_stats
    rng = random.Random(1)
    for image_id, present in IMAGES.items():
        for strategy in build_pope.STRATEGIES:
            negatives = build_pope.negative_objects(
                present, strategy, k=3, frequency=frequency, co=co,
                rng=rng, categories=CATS,
            )
            assert not (set(negatives) & present), f"{image_id}/{strategy}"
            assert len(set(negatives)) == len(negatives)


def test_negative_objects_rejects_an_unknown_strategy(build_pope, toy_stats):
    frequency, co = toy_stats
    with pytest.raises(ValueError, match="strategy"):
        build_pope.negative_objects(
            {"dog"}, "nonsense", k=1, frequency=frequency, co=co,
            rng=random.Random(0), categories=CATS,
        )


def _rows_for(build_pope, image_id, present, frequency, co, seed=42, k=3):
    return build_pope.questions_for_image(
        image_id, present, k=k, frequency=frequency, co=co,
        rng=random.Random(seed), categories=CATS,
    )


def test_questions_are_balanced_fifty_fifty_per_strategy(build_pope, toy_stats):
    frequency, co = toy_stats
    rows = _rows_for(build_pope, "img3", IMAGES["img3"], frequency, co)
    assert len(rows) == 3 * (3 + 3)  # 3 strategies x (3 yes + 3 no)
    for strategy in build_pope.STRATEGIES:
        cell = [r for r in rows if r["strategy"] == strategy]
        n_yes = sum(1 for r in cell if r["expected"] == "yes")
        n_no = sum(1 for r in cell if r["expected"] == "no")
        assert n_yes == n_no == 3


def test_questions_balance_when_the_image_has_fewer_objects_than_k(build_pope, toy_stats):
    frequency, co = toy_stats
    rows = _rows_for(build_pope, "img1", IMAGES["img1"], frequency, co, k=3)
    # Only one annotated object, so one yes and one no per strategy.
    assert len(rows) == 3 * 2
    for strategy in build_pope.STRATEGIES:
        cell = [r for r in rows if r["strategy"] == strategy]
        assert sum(1 for r in cell if r["expected"] == "yes") == 1
        assert sum(1 for r in cell if r["expected"] == "no") == 1


def test_no_question_ever_targets_an_object_present_in_the_image(build_pope, toy_stats):
    frequency, co = toy_stats
    for image_id, present in IMAGES.items():
        rows = _rows_for(build_pope, image_id, present, frequency, co)
        for row in rows:
            if row["expected"] == "no":
                assert row["object"] not in present, f"{image_id}: {row}"
            else:
                assert row["object"] in present, f"{image_id}: {row}"


def test_questions_are_deterministic_under_a_fixed_seed(build_pope, toy_stats):
    frequency, co = toy_stats
    first = _rows_for(build_pope, "img3", IMAGES["img3"], frequency, co, seed=7)
    second = _rows_for(build_pope, "img3", IMAGES["img3"], frequency, co, seed=7)
    assert first == second


def test_positive_questions_are_shared_across_strategies(build_pope, toy_stats):
    frequency, co = toy_stats
    rows = _rows_for(build_pope, "img3", IMAGES["img3"], frequency, co)
    positives = {
        s: sorted(r["object"] for r in rows if r["strategy"] == s and r["expected"] == "yes")
        for s in build_pope.STRATEGIES
    }
    assert len(set(map(tuple, positives.values()))) == 1


def test_image_without_coco_objects_yields_no_questions(build_pope, toy_stats):
    frequency, co = toy_stats
    assert _rows_for(build_pope, "imgX", set(), frequency, co) == []


def test_question_rows_carry_the_rendered_prompt(build_pope, toy_stats):
    frequency, co = toy_stats
    rows = _rows_for(build_pope, "img1", IMAGES["img1"], frequency, co)
    for row in rows:
        assert row["question"] == build_pope.render_question(row["object"])
        assert set(row) == {"image_id", "question", "expected", "strategy", "object"}


# ---------------------------------------------------------------- pope_generate wiring


def _args(**overrides) -> SimpleNamespace:
    base = dict(
        alpha=1.05, beta=0.1, tau=3.0, selected_layer=18, se_layers=(0, 28),
        lam=0.5, ceiling=1.8,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_baseline_condition_has_no_sparc(pope_generate):
    assert pope_generate.hparams_for_condition("baseline", _args()) is None


def test_sparc_condition_uses_the_original_alpha_path(pope_generate):
    hp = pope_generate.hparams_for_condition("sparc", _args())
    assert hp.adaptive is False
    assert hp.alpha == 1.05
    assert hp.tau == 3.0
    assert hp.se_layers == (0, 28)


def test_adaptive_condition_turns_on_the_registry(pope_generate):
    hp = pope_generate.hparams_for_condition("adaptive", _args())
    assert hp.adaptive is True
    assert hp.lam == 0.5
    assert hp.ceiling == 1.8


def test_unknown_condition_is_rejected(pope_generate):
    with pytest.raises(ValueError, match="condition"):
        pope_generate.hparams_for_condition("nonsense", _args())


def test_adaptive_condition_rejects_a_negative_lam(pope_generate):
    with pytest.raises(ValueError, match="lam"):
        pope_generate.hparams_for_condition("adaptive", _args(lam=-1.0))


def test_prompt_for_returns_the_prerendered_question(pope_generate):
    assert pope_generate.prompt_for({"question": "Is there a dog in the image?"}) == (
        "Is there a dog in the image?"
    )


def test_prompt_for_rejects_a_row_without_a_question(pope_generate):
    with pytest.raises(ValueError, match="question"):
        pope_generate.prompt_for({"image_id": "x"})


def test_answer_key_separates_conditions_and_strategies(pope_generate):
    row = {"image_id": "img1", "strategy": "random", "object": "dog", "expected": "yes"}
    keys = {pope_generate.answer_key(row, c) for c in pope_generate.CONDITIONS}
    assert len(keys) == 3
    other = {**row, "strategy": "popular"}
    assert pope_generate.answer_key(row, "sparc") != pope_generate.answer_key(other, "sparc")


def test_read_done_round_trips_appended_answers(pope_generate, tmp_path):
    path = tmp_path / "pope_answers.jsonl"
    assert pope_generate.read_done(path) == set()

    entry = {
        "image_id": "img1", "strategy": "random", "object": "dog",
        "expected": "yes", "condition": "sparc", "answer_raw": "Yes", "answer": "yes",
    }
    pope_generate.append_answer(path, entry)
    done = pope_generate.read_done(path)
    assert pope_generate.answer_key(entry, "sparc") in done
    assert pope_generate.answer_key(entry, "baseline") not in done


def test_read_done_ignores_malformed_lines(pope_generate, tmp_path):
    path = tmp_path / "pope_answers.jsonl"
    path.write_text("not json\n\n" + json.dumps({
        "image_id": "i", "strategy": "random", "object": "dog",
        "expected": "yes", "condition": "baseline",
    }) + "\n", encoding="utf-8")
    assert len(pope_generate.read_done(path)) == 1


def test_group_by_image_preserves_file_order(pope_generate):
    rows = [
        {"image_id": "b"}, {"image_id": "a"}, {"image_id": "b"}, {"image_id": "c"},
    ]
    grouped = pope_generate.group_by_image(rows)
    assert list(grouped) == ["b", "a", "c"]
    assert len(grouped["b"]) == 2


# ---------------------------------------------------------------- pope_report


def _answer(condition, strategy, expected, raw):
    return {
        "condition": condition, "strategy": strategy, "expected": expected,
        "answer_raw": raw, "answer": normalize_answer(raw), "model_id": "mock/test",
    }


def test_collect_pope_rows_adds_a_pooled_all_row(pope_report):
    entries = [
        _answer("baseline", "random", "yes", "Yes"),
        _answer("baseline", "random", "no", "Yes"),
        _answer("baseline", "popular", "no", "No"),
    ]
    rows = pope_report.collect_pope_rows(entries, model_id="mock/test")
    by_strategy = {r["strategy"]: r for r in rows}
    assert set(by_strategy) == {"random", "popular", "all"}
    assert by_strategy["all"]["n_total"] == 3
    assert by_strategy["random"]["n_total"] == 2
    assert by_strategy["all"]["tp"] == 1
    assert by_strategy["all"]["fp"] == 1
    assert by_strategy["all"]["tn"] == 1


def test_collect_pope_rows_orders_conditions_canonically(pope_report):
    entries = [
        _answer("adaptive", "random", "yes", "Yes"),
        _answer("baseline", "random", "yes", "Yes"),
        _answer("sparc", "random", "yes", "Yes"),
    ]
    rows = pope_report.collect_pope_rows(entries, model_id="mock/test")
    seen = []
    for row in rows:
        if row["condition"] not in seen:
            seen.append(row["condition"])
    assert seen == ["baseline", "sparc", "adaptive"]


def test_renormalise_recovers_the_answer_from_the_raw_text(pope_report):
    entries = [{"answer_raw": "Yes, there is.", "answer": "no"}]
    assert pope_report._renormalise(entries)[0]["answer"] == "yes"


def test_write_pope_results_emits_json_and_csv(pope_report, tmp_path):
    entries = [_answer("baseline", "random", "yes", "Yes")]
    rows = pope_report.collect_pope_rows(entries, model_id="mock/test")
    json_path, csv_path = pope_report.write_pope_results(rows, tmp_path)
    assert json_path.exists() and csv_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["n_rows"] == len(rows)
    header = csv_path.read_text(encoding="utf-8").splitlines()[0]
    assert "yes_ratio" in header and "pct_invalid" in header


# ---------------------------------------------------------------- phase3 CLI wiring


def test_phase3_defaults_keep_the_original_alpha_path(phase3):
    hp = phase3.sparc_hparams_from_args(phase3.build_parser().parse_args([]))
    assert hp.adaptive is False
    assert hp.lam == 0.0
    assert hp.ceiling == 2.0


def test_phase3_adaptive_flag_switches_the_path(phase3):
    hp = phase3.sparc_hparams_from_args(phase3.build_parser().parse_args(["--adaptive"]))
    assert hp.adaptive is True
    assert hp.lam == 0.0


def test_phase3_adaptive_accepts_lam_and_ceiling(phase3):
    args = phase3.build_parser().parse_args(
        ["--adaptive", "--lam", "0.7", "--ceiling", "1.5"]
    )
    hp = phase3.sparc_hparams_from_args(args)
    assert (hp.adaptive, hp.lam, hp.ceiling) == (True, 0.7, 1.5)


def test_phase3_adaptive_does_not_require_alpha_above_one(phase3):
    args = phase3.build_parser().parse_args(["--adaptive", "--alpha", "1.0"])
    assert phase3.sparc_hparams_from_args(args).adaptive is True


def test_phase3_without_adaptive_still_rejects_alpha_at_one(phase3):
    args = phase3.build_parser().parse_args(["--alpha", "1.0"])
    with pytest.raises(ValueError, match="alpha"):
        phase3.sparc_hparams_from_args(args)


def test_phase3_sparc_hparams_land_in_the_run_params_snapshot(phase3):
    """run_params.json is built by splatting as_dict(), so new hyperparameters
    reach the artefact without touching the snapshot code."""
    hp = phase3.sparc_hparams_from_args(
        phase3.build_parser().parse_args(["--adaptive", "--lam", "0.3"])
    )
    snapshot = {"run_name": "x", **hp.as_dict()}
    assert snapshot["adaptive"] is True
    assert snapshot["lam"] == 0.3
    assert snapshot["ceiling"] == 2.0
    json.dumps(snapshot)  # must stay serialisable
