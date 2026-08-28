"""Tests for :mod:`vr_modality_bias.metrics.judge`.

Prompt assembly, the strict parser, and the aggregation. No model is loaded
anywhere: the module takes text in and returns structure out, which is the whole
reason it is separate from the script that owns the GPU.
"""

from __future__ import annotations

import math

import pytest

from vr_modality_bias.metrics.judge import (
    ALL_LABELS,
    VERDICT_CORRECT,
    VERDICT_INCORRECT,
    VERDICT_INVALID,
    VERDICT_NOT_ADDRESSED,
    VERDICTS,
    JudgeVerdict,
    build_judge_prompt,
    compute_judge_aggregate,
    group_by_arm,
    parse_verdict,
)


def _prompt(**overrides) -> str:
    kwargs = {
        "composed_question": "Is there a sofa? How many lamps are there?",
        "sub_question": "How many lamps are there?",
        "reference_answer": 2,
        "generated_answer": "There is a sofa and two lamps above it.",
    }
    kwargs.update(overrides)
    return build_judge_prompt(**kwargs)


# ---------------------------------------------------------------- the prompt


def test_the_prompt_carries_all_four_pieces():
    prompt = _prompt()

    assert "Is there a sofa? How many lamps are there?" in prompt
    assert "How many lamps are there?" in prompt
    assert "2" in prompt
    assert "There is a sofa and two lamps above it." in prompt


def test_the_prompt_has_no_unrendered_placeholder():
    prompt = _prompt()

    for placeholder in ("{composed_question}", "{sub_question}",
                        "{reference_answer}", "{generated_answer}"):
        assert placeholder not in prompt, placeholder


def test_the_prompt_never_offers_the_image():
    """Text-only is the experimental design, not a saving.

    What is under test is whether the equirectangular projection breaks the
    model being evaluated. A judge that saw the same panorama would be exposed
    to the same defect and the measurement would be circular.
    """
    prompt = _prompt().lower()

    assert "cannot see the photograph" in prompt
    for banned in ("look at the image", "<image>", "shown below", "attached image"):
        assert banned not in prompt, banned


def test_the_prompt_forbids_grading_the_reference():
    prompt = _prompt().lower()

    assert "correct by definition" in prompt
    assert "never asked" in prompt


def test_the_prompt_names_exactly_the_three_verdicts():
    prompt = _prompt()

    for verdict in VERDICTS:
        assert f'"{verdict}"' in prompt, verdict
    assert '"invalid"' not in prompt, (
        "invalid is our label for a judge that failed, never one it may pick"
    )


def test_the_prompt_says_silence_is_not_addressed():
    prompt = _prompt().lower()

    assert 'silence is "not_addressed"' in prompt


def test_braces_in_the_generated_answer_survive():
    prompt = _prompt(generated_answer='The label reads {"a": 1} on the box.')

    assert '{"a": 1}' in prompt


# ---------------------------------------------------------------- the parser


@pytest.mark.parametrize("verdict", list(VERDICTS))
def test_each_valid_verdict_parses(verdict):
    out = parse_verdict('{"verdict": "%s"}' % verdict)

    assert out.verdict == verdict


def test_the_evidence_is_kept():
    out = parse_verdict('{"verdict": "correct", "evidence": "two lamps"}')

    assert out.verdict == VERDICT_CORRECT
    assert out.evidence == "two lamps"


def test_a_missing_evidence_is_empty_not_invalid():
    out = parse_verdict('{"verdict": "correct"}')

    assert out.verdict == VERDICT_CORRECT
    assert out.evidence == ""


def test_a_thinking_block_is_stripped():
    """Qwen3 emits <think> by default; a strict parser would reject every call."""
    out = parse_verdict(
        '<think>The answer says two lamps, the reference says 2, so they '
        'agree.</think>\n{"verdict": "correct", "evidence": "two lamps"}'
    )

    assert out.verdict == VERDICT_CORRECT
    assert out.evidence == "two lamps"


def test_an_unterminated_thinking_block_does_not_swallow_the_json():
    out = parse_verdict('{"verdict": "incorrect"}\n<think>wait, actually')

    assert out.verdict == VERDICT_INCORRECT


def test_a_code_fence_is_stripped():
    out = parse_verdict('```json\n{"verdict": "not_addressed"}\n```')

    assert out.verdict == VERDICT_NOT_ADDRESSED


def test_surrounding_prose_does_not_break_the_parse():
    out = parse_verdict('Here is my verdict:\n{"verdict": "correct"}\nHope that helps.')

    assert out.verdict == VERDICT_CORRECT


def test_the_label_is_case_insensitive_and_trimmed():
    out = parse_verdict('{"verdict": "  CORRECT "}')

    assert out.verdict == VERDICT_CORRECT


@pytest.mark.parametrize("text,reason", [
    ("", "empty response"),
    ("   ", "empty response"),
    (None, "empty response"),
    ("I think it is correct.", "no JSON object found"),
    ('{"answer": "correct"}', "no 'verdict' key"),
    ('{"verdict": "maybe"}', "verdict not one of"),
    ('{"verdict": "partially_correct"}', "verdict not one of"),
    ('{"verdict": 1}', "verdict not one of"),
    ('{"verdict": null}', "verdict not one of"),
    ('{"verdict": "correct"', "no JSON object found"),
])
def test_anything_the_parser_cannot_trust_is_invalid(text, reason):
    out = parse_verdict(text)

    assert out.verdict == VERDICT_INVALID
    assert reason in out.reason


def test_an_invalid_verdict_keeps_the_raw_text_for_auditing():
    out = parse_verdict("total nonsense")

    assert out.raw == "total nonsense"


def test_a_bare_word_is_not_accepted():
    """A judge answering 'correct' without JSON is a contract violation.

    Accepting it would silently widen the contract and hide a judge that
    stopped following the format.
    """
    assert parse_verdict("correct").verdict == VERDICT_INVALID


def test_invalid_is_never_one_of_the_three():
    assert VERDICT_INVALID not in VERDICTS
    assert set(ALL_LABELS) == set(VERDICTS) | {VERDICT_INVALID}


def test_the_dataclass_rejects_an_unknown_label():
    with pytest.raises(ValueError, match="probably"):
        JudgeVerdict("probably")


# ---------------------------------------------------------------- aggregation


def _item(*verdicts, answer="one two three four five", types=None, arm="off"):
    kinds = types or ["existence"] * len(verdicts)
    return {
        "condition_label": arm,
        "answer": answer,
        "verdicts": [
            {"component_type": kind, "verdict": verdict}
            for kind, verdict in zip(kinds, verdicts)
        ],
    }


def test_counts_and_rates_over_the_three_verdicts():
    agg = compute_judge_aggregate([
        _item(VERDICT_CORRECT, VERDICT_CORRECT, VERDICT_INCORRECT),
        _item(VERDICT_NOT_ADDRESSED),
    ])
    overall = agg["overall"]

    assert overall["n_subquestions"] == 4
    assert overall["n_correct"] == 2
    assert overall["n_incorrect"] == 1
    assert overall["n_not_addressed"] == 1
    assert overall["rate_correct"] == 0.5


def test_invalid_is_counted_and_stays_out_of_the_other_three():
    agg = compute_judge_aggregate([
        _item(VERDICT_CORRECT, VERDICT_INVALID),
    ])
    overall = agg["overall"]

    assert overall["n_invalid"] == 1
    assert overall["n_correct"] == 1
    assert overall["n_incorrect"] == 0
    assert overall["n_not_addressed"] == 0


def test_invalid_sits_inside_the_denominator():
    """Otherwise a judge that answers only when confident inflates %correct.

    Same pathology the three-verdict rule guards against, one level up.
    """
    agg = compute_judge_aggregate([_item(VERDICT_CORRECT, VERDICT_INVALID)])
    overall = agg["overall"]

    assert overall["rate_correct"] == 0.5, (
        "one correct out of two sub-questions, not one out of one"
    )
    assert math.isclose(
        sum(overall[f"rate_{label}"] for label in ALL_LABELS), 1.0
    )


def test_rates_are_broken_out_by_component_type():
    agg = compute_judge_aggregate([
        _item(VERDICT_CORRECT, VERDICT_INCORRECT,
              types=["existence", "direction"]),
    ])

    assert agg["by_type"]["existence"]["rate_correct"] == 1.0
    assert agg["by_type"]["direction"]["rate_incorrect"] == 1.0


def test_all_correct_counts_items_not_sub_questions():
    agg = compute_judge_aggregate([
        _item(VERDICT_CORRECT, VERDICT_CORRECT),
        _item(VERDICT_CORRECT, VERDICT_NOT_ADDRESSED),
    ])

    assert agg["n_all_correct"] == 1
    assert agg["rate_all_correct"] == 0.5


def test_an_item_with_one_invalid_is_not_all_correct():
    agg = compute_judge_aggregate([_item(VERDICT_CORRECT, VERDICT_INVALID)])

    assert agg["n_all_correct"] == 0, (
        "a judge that would not answer must not become a silent clean sweep"
    )


def test_mean_answer_words_is_reported():
    agg = compute_judge_aggregate([
        _item(VERDICT_CORRECT, answer="one two three"),
        _item(VERDICT_CORRECT, answer="one two three four five"),
    ])

    assert agg["mean_answer_words"] == 4.0


def test_degeneration_is_reported_alongside_the_verdicts():
    agg = compute_judge_aggregate([
        _item(VERDICT_CORRECT, answer="its its its surroundings"),
        _item(VERDICT_CORRECT, answer="a perfectly ordinary answer here"),
    ])

    assert agg["n_degenerate"] == 1
    assert agg["rate_degenerate"] == 0.5


def test_an_empty_input_does_not_divide_by_zero():
    agg = compute_judge_aggregate([])

    assert agg["n_items"] == 0
    assert math.isnan(agg["rate_all_correct"])
    assert math.isnan(agg["mean_answer_words"])
    assert math.isnan(agg["overall"]["rate_correct"])


def test_an_unknown_label_in_the_input_is_rejected():
    with pytest.raises(ValueError, match="skipped"):
        compute_judge_aggregate([_item("skipped")])


def test_group_by_arm_splits_the_items():
    groups = group_by_arm([
        _item(VERDICT_CORRECT, arm="off"),
        _item(VERDICT_CORRECT, arm="on sparc a=1.05 L15"),
        _item(VERDICT_INCORRECT, arm="off"),
    ])

    assert set(groups) == {"off", "on sparc a=1.05 L15"}
    assert len(groups["off"]) == 2


# ---------------------------------------------------------------- think


def test_think_statistics_are_absent_without_think_items():
    agg = compute_judge_aggregate([_item(VERDICT_CORRECT)])

    assert agg["n_think"] == 0
    assert math.isnan(agg["rate_think_well_formed"])
    assert math.isnan(agg["mean_think_words"])


def test_think_statistics_count_well_formed_blocks_and_their_length():
    agg = compute_judge_aggregate([
        {**_item(VERDICT_CORRECT), "think_well_formed": True, "think": "a b c d"},
        {**_item(VERDICT_CORRECT), "think_well_formed": False, "think": ""},
    ])

    assert agg["n_think"] == 2
    assert agg["rate_think_well_formed"] == 0.5
    assert agg["mean_think_words"] == 2.0
