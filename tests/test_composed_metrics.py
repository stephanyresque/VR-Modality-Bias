"""Tests for :mod:`vr_modality_bias.metrics.composed`.

The verifiers look for the ground truth INSIDE a long free-text answer; they
never test for equality, because a short annotation and a long answer never
coincide. The invariant that matters most is that an undecidable case comes
back ``indeterminate`` rather than guessing: a wrong guess in either direction
moves a headline number.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vr_modality_bias.data.vocabulary import load_vocabulary
from vr_modality_bias.metrics.composed import (
    VERDICT_CORRECT,
    VERDICT_INCORRECT,
    VERDICT_INDETERMINATE,
    ComponentVerdict,
    compute_composed_aggregate,
    verify_count,
    verify_direction,
    verify_existence,
)

_DIRECTIONS = load_vocabulary(
    Path(__file__).parent.parent / "configs" / "direction_terms.json"
)


# ---------------------------------------------------------------- existence


def test_an_affirmed_mention_matches_a_yes():
    out = verify_existence("There are three chairs by the table.", "chair", "yes")

    assert out.verdict == VERDICT_CORRECT
    assert "chairs" in out.evidence


def test_an_affirmed_mention_contradicts_a_no():
    out = verify_existence("There are three chairs by the table.", "chair", "no")

    assert out.verdict == VERDICT_INCORRECT


def test_a_negated_mention_matches_a_no():
    out = verify_existence("There are no chairs in this room.", "chair", "no")

    assert out.verdict == VERDICT_CORRECT


def test_a_negated_mention_contradicts_a_yes():
    """The negative half of the set is the half that matters: those questions
    exist to test recognition of absence."""
    out = verify_existence("There are no chairs in this room.", "chair", "yes")

    assert out.verdict == VERDICT_INCORRECT


@pytest.mark.parametrize("answer", [
    "I don't see any chairs.",
    "There aren't any chairs here.",
    "No chairs are visible.",
    "The room has no chairs.",
    "There is nothing that looks like a chair.",
    "A chair is absent from the scene.",
    "I cannot see a chair.",
])
def test_negation_forms_are_all_read_as_absence(answer):
    assert verify_existence(answer, "chair", "no").verdict == VERDICT_CORRECT


def test_a_target_never_mentioned_is_indeterminate_not_incorrect():
    out = verify_existence("There is a table and a lamp.", "chair", "yes")

    assert out.verdict == VERDICT_INDETERMINATE, (
        "silence is not a wrong answer; calling it incorrect would credit an "
        "arm for talking less"
    )
    assert out.evidence == ""


def test_a_target_never_mentioned_is_indeterminate_for_a_no_as_well():
    out = verify_existence("There is a table and a lamp.", "chair", "no")

    assert out.verdict == VERDICT_INDETERMINATE


def test_a_bare_no_answer_is_read_even_without_the_target():
    out = verify_existence("No, there isn't.", "chair", "no")

    assert out.verdict == VERDICT_CORRECT


def test_contradictory_clauses_are_indeterminate():
    out = verify_existence(
        "There are chairs here. Actually there are no chairs.", "chair", "yes"
    )

    assert out.verdict == VERDICT_INDETERMINATE


def test_a_negation_in_a_different_clause_does_not_leak():
    out = verify_existence("There are chairs, but no tables.", "chair", "yes")

    assert out.verdict == VERDICT_CORRECT, (
        "the negation belongs to the table clause, not the chair clause"
    )


def test_the_plural_of_the_target_is_matched():
    assert verify_existence("Two boxes sit there.", "box", "yes").verdict == VERDICT_CORRECT


def test_a_non_yes_no_annotation_is_rejected():
    with pytest.raises(ValueError, match="yes/no"):
        verify_existence("There is a chair.", "chair", "maybe")


# ---------------------------------------------------------------- count


def test_a_digit_next_to_the_target_matches():
    out = verify_count("There are 3 chairs around the table.", "chair", 3)

    assert out.verdict == VERDICT_CORRECT
    assert "3 chairs" in out.evidence


def test_a_number_word_next_to_the_target_matches():
    out = verify_count("There are three chairs around the table.", "chair", 3)

    assert out.verdict == VERDICT_CORRECT


@pytest.mark.parametrize("word,value", [
    ("zero", 0), ("one", 1), ("four", 4), ("seven", 7), ("twelve", 12),
    ("nineteen", 19), ("twenty", 20),
])
def test_number_words_are_read(word, value):
    assert verify_count(f"There are {word} chairs.", "chair", value).verdict == VERDICT_CORRECT


def test_a_wrong_number_is_incorrect():
    assert verify_count("There are 5 chairs.", "chair", 3).verdict == VERDICT_INCORRECT


def test_the_long_answer_only_needs_the_number_in_the_target_clause():
    """The composed answer names other quantities; only the chair clause counts."""
    out = verify_count(
        "There are three chairs, two beside the table and one near the window.",
        "chair", 3,
    )

    assert out.verdict == VERDICT_CORRECT


def test_no_number_at_all_is_indeterminate():
    out = verify_count("There are chairs around the table.", "chair", 3)

    assert out.verdict == VERDICT_INDETERMINATE


def test_the_target_missing_entirely_is_indeterminate():
    out = verify_count("There is a table.", "chair", 3)

    assert out.verdict == VERDICT_INDETERMINATE
    assert out.evidence == ""


def test_two_competing_numbers_in_one_clause_are_indeterminate():
    out = verify_count("I see 3 or 4 chairs there.", "chair", 3)

    assert out.verdict == VERDICT_INDETERMINATE, (
        "picking one of two candidates would be a guess"
    )


def test_conflicting_numbers_across_clauses_are_indeterminate():
    out = verify_count(
        "There are three chairs here. Two chairs are stacked.", "chair", 3
    )

    assert out.verdict == VERDICT_INDETERMINATE


def test_a_non_numeric_annotation_is_rejected():
    with pytest.raises(ValueError, match="numeric"):
        verify_count("There are 3 chairs.", "chair", "a few")


# ---------------------------------------------------------------- direction


def test_the_annotated_direction_found_near_the_target_matches():
    out = verify_direction(
        "The window is to the left of the door.", "window", "left", _DIRECTIONS
    )

    assert out.verdict == VERDICT_CORRECT


def test_an_equivalent_term_from_the_table_matches():
    out = verify_direction(
        "The lamp sits atop the cabinet.", "lamp", "above", _DIRECTIONS
    )

    assert out.verdict == VERDICT_CORRECT, "'atop' is a table synonym of 'above'"


def test_the_wrong_direction_is_incorrect():
    out = verify_direction(
        "The window is to the right of the door.", "window", "left", _DIRECTIONS
    )

    assert out.verdict == VERDICT_INCORRECT


def test_no_direction_term_near_the_target_is_indeterminate():
    out = verify_direction("There is a window.", "window", "left", _DIRECTIONS)

    assert out.verdict == VERDICT_INDETERMINATE


def test_the_target_missing_entirely_is_indeterminate_for_direction():
    out = verify_direction("There is a door.", "window", "left", _DIRECTIONS)

    assert out.verdict == VERDICT_INDETERMINATE
    assert out.evidence == ""


def test_a_direction_attached_across_a_comma_is_found():
    """Deliberate asymmetry with count: an orientation phrase hangs off its
    noun across punctuation, a quantity sits against it. Direction therefore
    searches the sentence, count searches the clause."""
    out = verify_direction(
        "There are three chairs, to the left of the table.",
        "chair", "left", _DIRECTIONS,
    )

    assert out.verdict == VERDICT_CORRECT


def test_count_stays_clause_scoped_so_a_neighbouring_quantity_does_not_confuse_it():
    out = verify_count(
        "There are three chairs, and two tables.", "chair", 3
    )

    assert out.verdict == VERDICT_CORRECT, (
        "widening count to the sentence would see 3 and 2 and give up"
    )


def test_two_directions_in_one_clause_are_indeterminate():
    out = verify_direction(
        "The window is to the left or to the right of the door.",
        "window", "left", _DIRECTIONS,
    )

    assert out.verdict == VERDICT_INDETERMINATE


def test_a_direction_outside_the_table_is_rejected():
    with pytest.raises(ValueError, match="northwest"):
        verify_direction("The window is there.", "window", "northwest", _DIRECTIONS)


def test_the_shipped_direction_table_loads():
    assert _DIRECTIONS.name == "directions_en"
    assert "left" in _DIRECTIONS.categories
    assert _DIRECTIONS.synonym_to_category["underneath"] == "below"


# ---------------------------------------------------------------- verdict type


def test_an_unknown_verdict_is_rejected():
    with pytest.raises(ValueError, match="Unknown verdict"):
        ComponentVerdict("probably", "")


# ---------------------------------------------------------------- aggregate


def _item(*verdicts: str, answer: str = "a b c d e", degenerate: bool = False) -> dict:
    return {
        "answer": answer,
        "is_degenerate": degenerate,
        "components": [
            {"component_type": t, "verdict": v}
            for t, v in zip(("existence", "count", "direction"), verdicts)
        ],
    }


def test_the_unit_of_aggregation_is_the_component():
    agg = compute_composed_aggregate([
        _item(VERDICT_CORRECT, VERDICT_CORRECT, VERDICT_INCORRECT),
        _item(VERDICT_CORRECT),
    ])

    assert agg["overall"]["n_components"] == 4, (
        "an item with three components must weigh three times an item with one"
    )
    assert agg["overall"]["n_correct"] == 3
    assert agg["overall"]["rate_correct"] == pytest.approx(3 / 4)


def test_rates_are_broken_out_by_component_type():
    agg = compute_composed_aggregate([
        _item(VERDICT_CORRECT, VERDICT_INDETERMINATE, VERDICT_INCORRECT),
        _item(VERDICT_INCORRECT, VERDICT_INDETERMINATE, VERDICT_CORRECT),
    ])

    assert agg["by_type"]["existence"]["rate_correct"] == pytest.approx(0.5)
    assert agg["by_type"]["count"]["rate_indeterminate"] == pytest.approx(1.0)
    assert agg["by_type"]["direction"]["rate_correct"] == pytest.approx(0.5)


def test_the_three_rates_are_reported_separately_and_sum_to_one():
    agg = compute_composed_aggregate([
        _item(VERDICT_CORRECT, VERDICT_INCORRECT, VERDICT_INDETERMINATE),
    ])
    overall = agg["overall"]

    assert overall["rate_correct"] == pytest.approx(1 / 3)
    assert overall["rate_incorrect"] == pytest.approx(1 / 3)
    assert overall["rate_indeterminate"] == pytest.approx(1 / 3)
    assert (
        overall["rate_correct"]
        + overall["rate_incorrect"]
        + overall["rate_indeterminate"]
    ) == pytest.approx(1.0)


def test_all_correct_counts_items_not_components():
    agg = compute_composed_aggregate([
        _item(VERDICT_CORRECT, VERDICT_CORRECT, VERDICT_CORRECT),
        _item(VERDICT_CORRECT, VERDICT_CORRECT, VERDICT_INDETERMINATE),
    ])

    assert agg["n_all_correct"] == 1
    assert agg["rate_all_correct"] == pytest.approx(0.5)


def test_an_indeterminate_component_blocks_all_correct():
    agg = compute_composed_aggregate([
        _item(VERDICT_CORRECT, VERDICT_INDETERMINATE),
    ])

    assert agg["n_all_correct"] == 0


def test_mean_answer_length_is_reported():
    agg = compute_composed_aggregate([
        _item(VERDICT_CORRECT, answer="one two three"),
        _item(VERDICT_CORRECT, answer="one two three four five"),
    ])

    assert agg["mean_answer_words"] == pytest.approx(4.0)


def test_the_degeneration_rate_is_reported():
    agg = compute_composed_aggregate([
        _item(VERDICT_INCORRECT, degenerate=True),
        _item(VERDICT_CORRECT),
    ])

    assert agg["n_degenerate"] == 1
    assert agg["rate_degenerate"] == pytest.approx(0.5)


def test_an_empty_group_yields_nan_rates_not_zeros():
    agg = compute_composed_aggregate([])

    assert agg["n_items"] == 0
    assert agg["overall"]["rate_correct"] != agg["overall"]["rate_correct"]
    assert agg["rate_all_correct"] != agg["rate_all_correct"]
