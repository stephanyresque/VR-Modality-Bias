"""Tests for scripts/composed_report.py.

Covers the join between answers.jsonl and the annotated questions, the
degeneration override, and the fact that the arm labelling comes from
chair_report.py rather than a second implementation.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from vr_modality_bias.data.annotations import (
    QuestionAnnotation,
    QuestionComponent,
    write_question_annotations,
)
from vr_modality_bias.data.vocabulary import load_vocabulary
from vr_modality_bias.metrics.composed import (
    VERDICT_CORRECT,
    VERDICT_INCORRECT,
    VERDICT_INDETERMINATE,
)

_SCRIPTS = Path(__file__).parent.parent / "scripts"
_DIRECTIONS = load_vocabulary(
    Path(__file__).parent.parent / "configs" / "direction_terms.json"
)


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"_script_{name}", _SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def report():
    return _load_script("composed_report")


def _question(
    image_id="img_a", question_id="q1", *, components=None
) -> QuestionAnnotation:
    return QuestionAnnotation(
        image_id=image_id,
        question_id=question_id,
        question_text="Are there chairs? How many, and where?",
        components=tuple(components or [
            QuestionComponent("existence", "chair", "yes"),
            QuestionComponent("count", "chair", 3),
            QuestionComponent("direction", "chair", "left"),
        ]),
    )


def _answer(text, *, image_id="img_a", question_id="q1", condition="off", sparc=None):
    return {
        "image_id": image_id,
        "question_id": question_id,
        "condition": condition,
        "alpha": None if condition == "off" else 1.1,
        "sparc": sparc,
        "answer": text,
        "model_id": "mock/test",
    }


def _index(questions):
    return {(q.image_id, q.question_id): q for q in questions}


# ---------------------------------------------------------------- grounding


def test_an_answer_without_an_annotated_question_is_fatal(report):
    questions = _index([_question()])
    entries = [_answer("x"), _answer("y", question_id="q_missing")]

    with pytest.raises(ValueError):
        report.assert_answers_are_grounded(entries, questions)


def test_the_failure_shows_identifiers_from_both_sides(report):
    questions = _index([_question(image_id="000000000139", question_id="q1")])
    entries = [_answer("x", image_id="adt_seq07", question_id="adt_q1")]

    with pytest.raises(ValueError) as excinfo:
        report.assert_answers_are_grounded(entries, questions)

    message = str(excinfo.value)
    assert "adt_seq07" in message, "must show the answer-side id"
    assert "000000000139" in message, "must show the question-side id"
    assert "1 answer(s)" in message


def test_fully_grounded_answers_pass(report):
    questions = _index([_question(), _question(question_id="q2")])

    report.assert_answers_are_grounded(
        [_answer("x"), _answer("y", question_id="q2")], questions
    )


def test_a_question_with_no_answer_is_not_an_error(report):
    questions = _index([_question(), _question(question_id="q2")])

    report.assert_answers_are_grounded([_answer("x")], questions)

    unanswered = set(questions) - {("img_a", "q1")}
    assert unanswered == {("img_a", "q2")}


# ---------------------------------------------------------------- wiring


def test_each_component_is_verified_against_the_answer(report):
    questions = _index([_question()])
    entries = [_answer(
        "Yes, there are three chairs, to the left of the table."
    )]

    verified = report.verify_entries(entries, questions, _DIRECTIONS)

    verdicts = {c["component_type"]: c["verdict"] for c in verified[0]["components"]}
    assert verdicts == {
        "existence": VERDICT_CORRECT,
        "count": VERDICT_CORRECT,
        "direction": VERDICT_CORRECT,
    }


def test_every_component_carries_its_evidence(report):
    questions = _index([_question()])
    entries = [_answer("There are three chairs to the left.")]

    components = report.verify_entries(entries, questions, _DIRECTIONS)[0]["components"]

    for component in components:
        assert component["evidence"], (
            "the audit sample is useless without the span that justified the "
            "verdict"
        )


def test_a_silent_answer_yields_indeterminate_not_incorrect(report):
    questions = _index([_question()])
    entries = [_answer("The room is bright and well furnished.")]

    components = report.verify_entries(entries, questions, _DIRECTIONS)[0]["components"]

    assert {c["verdict"] for c in components} == {VERDICT_INDETERMINATE}


def test_the_arm_label_comes_from_chair_report(report):
    from vr_modality_bias.experiment.sparc import SparcHyperparams

    sparc = SparcHyperparams(
        alpha=1.0, adaptive=True, qcond=True, selected_layer=20, se_layers=(0, 31)
    ).as_dict()
    questions = _index([_question()])
    entries = [_answer("x", condition="on", sparc=sparc)]

    verified = report.verify_entries(entries, questions, _DIRECTIONS)

    assert verified[0]["condition_label"] == "on adaptive+qcond q=0.05 L20"


# ---------------------------------------------------------------- degeneration


def test_a_degenerate_answer_may_not_score_a_correct_component(report):
    """A repetition loop can hit the right token by accident."""
    questions = _index([_question(components=[
        QuestionComponent("existence", "chair", "yes"),
    ])])
    entries = [_answer("chair chair chair chair chair chair")]

    verified = report.verify_entries(entries, questions, _DIRECTIONS)
    component = verified[0]["components"][0]

    assert verified[0]["is_degenerate"] is True
    assert component["verdict"] == VERDICT_INCORRECT, (
        "degeneration is a model failure, so it is incorrect rather than "
        "indeterminate: indeterminate is about OUR inability to decide"
    )
    assert "degenerate" in component["evidence"]


def test_the_same_answer_without_degeneration_scores_correct(report):
    questions = _index([_question(components=[
        QuestionComponent("existence", "chair", "yes"),
    ])])
    entries = [_answer("There is a chair next to the desk.")]

    verified = report.verify_entries(entries, questions, _DIRECTIONS)

    assert verified[0]["is_degenerate"] is False
    assert verified[0]["components"][0]["verdict"] == VERDICT_CORRECT


def test_degeneration_does_not_upgrade_an_indeterminate(report):
    questions = _index([_question(components=[
        QuestionComponent("existence", "sofa", "yes"),
    ])])
    entries = [_answer("chair chair chair chair chair chair")]

    component = report.verify_entries(entries, questions, _DIRECTIONS)[0]["components"][0]

    assert component["verdict"] == VERDICT_INDETERMINATE, (
        "the override only removes correctness; it must not invent a decision"
    )


# ---------------------------------------------------------------- rows


def test_rows_carry_the_three_rates_per_type(report):
    questions = _index([_question()])
    entries = [_answer("Yes, there are three chairs, to the left of the table.")]

    verified = report.verify_entries(entries, questions, _DIRECTIONS)
    rows = report.collect_rows(report.group_by_arm(verified), model_id="mock/test")

    assert len(rows) == 1
    row = rows[0]
    assert row["condition_label"] == "off"
    assert row["rate_correct"] == pytest.approx(1.0)
    assert row["rate_correct_existence"] == pytest.approx(1.0)
    assert row["rate_indeterminate_count"] == pytest.approx(0.0)
    assert row["n_components"] == 3


def test_arms_are_separate_rows(report):
    questions = _index([_question()])
    entries = [
        _answer("Yes, three chairs to the left.", condition="off"),
        _answer("Yes, three chairs to the left.", condition="on"),
    ]

    verified = report.verify_entries(entries, questions, _DIRECTIONS)
    rows = report.collect_rows(report.group_by_arm(verified), model_id="mock/test")

    assert {r["condition_label"] for r in rows} == {"off", "on α=1.1"}


def test_the_parser_requires_the_three_inputs(report):
    with pytest.raises(SystemExit):
        report.build_parser().parse_args(["--run-dir", "x"])
