"""Tests for :mod:`vr_modality_bias.metrics.report`.

Arm labelling and ordering, the degeneration detector, and the id-format guard.
None of these need a judge, a vocabulary or a ground-truth file: they are the
deterministic part of reporting that survives the move to a judge model.
"""

from __future__ import annotations

import random

import pytest

from vr_modality_bias.experiment.sparc import SparcHyperparams
from vr_modality_bias.metrics.report import (
    CONSECUTIVE_REPEAT_TRIGGER,
    DEGENERATION_REASONS,
    alpha_from_entries,
    assert_entries_are_grounded,
    classify_degeneration,
    condition_label,
    condition_sort_key,
    derive_model_id,
    label_from_sparc,
)


def _sparc(**kwargs) -> dict:
    return SparcHyperparams(**kwargs).as_dict()


ARM_SPARC_DICT = _sparc(alpha=1.1, selected_layer=20)
ARM_ADAPTIVE_DICT = _sparc(alpha=1.0, adaptive=True, lam=0.5, ceiling=2.0, selected_layer=20)
ARM_QCOND_DICT = _sparc(
    alpha=1.0, adaptive=True, qcond=True, qtop_frac=0.05, selected_layer=20
)
ARM_CONSERVE_DICT = _sparc(
    alpha=1.0, adaptive=True, qcond=True, conserve=True, rho=0.5, sink_frac=0.05,
    selected_layer=20,
)
ARM_CONSERVE_L15_DICT = _sparc(
    alpha=1.0, adaptive=True, qcond=True, conserve=True, rho=0.5, sink_frac=0.05,
    selected_layer=15,
)


def _on(sparc: dict | None) -> dict:
    return {"condition": "on", "alpha": 1.1, "sparc": sparc}


# ---------------------------------------------------------------- labels


def test_off_label():
    assert condition_label({"condition": "off", "sparc": None}) == "off"


def test_legacy_on_without_sparc_keeps_the_alpha_label():
    assert condition_label({"condition": "on", "alpha": 1.05}) == "on α=1.05"


def test_legacy_on_with_sparc_none_keeps_the_alpha_label():
    assert condition_label({"condition": "on", "alpha": 1.1, "sparc": None}) == "on α=1.1"


def test_label_alpha_c_arm():
    assert condition_label(_on(ARM_SPARC_DICT)) == "on sparc a=1.1 L20"


def test_label_adaptive_arm():
    assert condition_label(_on(ARM_ADAPTIVE_DICT)) == "on adaptive lam=0.5 ceil=2 L20"


def test_label_adaptive_qcond_arm():
    assert condition_label(_on(ARM_QCOND_DICT)) == "on adaptive+qcond q=0.05 L20"


def test_label_adaptive_qcond_conserve_arm():
    assert (
        condition_label(_on(ARM_CONSERVE_DICT))
        == "on adaptive+qcond+conserve rho=0.5 s=0.05 L20"
    )


def test_label_conserve_derived_layer_differs_only_in_the_layer():
    assert (
        condition_label(_on(ARM_CONSERVE_L15_DICT))
        == "on adaptive+qcond+conserve rho=0.5 s=0.05 L15"
    )


def test_every_arm_label_includes_the_reference_layer():
    for sparc in (ARM_SPARC_DICT, ARM_ADAPTIVE_DICT, ARM_QCOND_DICT, ARM_CONSERVE_DICT):
        assert condition_label(_on(sparc)).endswith("L20")


def test_label_from_sparc_is_what_condition_label_delegates_to():
    assert condition_label(_on(ARM_QCOND_DICT)) == label_from_sparc(ARM_QCOND_DICT)


# ---------------------------------------------------------------- ordering


def test_condition_sort_key_orders_by_increasing_complexity():
    labels = [
        condition_label(_on(ARM_CONSERVE_DICT)),
        "on α=1.1",
        condition_label(_on(ARM_QCOND_DICT)),
        "off",
        condition_label(_on(ARM_CONSERVE_L15_DICT)),
        condition_label(_on(ARM_ADAPTIVE_DICT)),
        condition_label(_on(ARM_SPARC_DICT)),
    ]
    random.Random(0).shuffle(labels)

    assert sorted(labels, key=condition_sort_key) == [
        "off",
        "on α=1.1",
        "on sparc a=1.1 L20",
        "on adaptive lam=0.5 ceil=2 L20",
        "on adaptive+qcond q=0.05 L20",
        "on adaptive+qcond+conserve rho=0.5 s=0.05 L15",
        "on adaptive+qcond+conserve rho=0.5 s=0.05 L20",
    ]


def test_off_sorts_first():
    assert condition_sort_key("off") < condition_sort_key("on α=1.1")
    assert condition_sort_key("off") < condition_sort_key("on sparc a=1.1 L20")


def test_legacy_alpha_label_sorts_with_the_sparc_arm():
    assert condition_sort_key("on α=1.1")[0] == 1
    assert condition_sort_key("on sparc a=1.1 L20")[0] == 1
    assert condition_sort_key("on adaptive lam=0.5 ceil=2 L20")[0] == 2


# ---------------------------------------------------------------- provenance


def test_alpha_comes_from_the_sparc_record_not_the_flat_field():
    alpha = alpha_from_entries([{"condition": "on", "alpha": 1.1,
                                 "sparc": ARM_ADAPTIVE_DICT}])

    assert alpha == 1.0, (
        "the adaptive arms record alpha=1.0 while the entry's flat `alpha` "
        "field says 1.1; reading 1.1 back means the flat field was used"
    )


def test_alpha_falls_back_to_the_flat_field_for_a_legacy_entry():
    assert alpha_from_entries([{"condition": "on", "alpha": 1.05}]) == 1.05


def test_alpha_skips_the_off_entries():
    assert alpha_from_entries([{"condition": "off", "alpha": None}]) is None


def test_model_id_comes_from_the_first_entry_that_carries_one():
    assert derive_model_id([{}, {"model_id": "mock/test"}]) == "mock/test"


def test_model_id_without_any_entry_is_unknown():
    assert derive_model_id([]) == "unknown"


def test_several_model_ids_warn_and_pick_the_first(capsys):
    model_id = derive_model_id([{"model_id": "b/two"}, {"model_id": "a/one"}])

    assert model_id == "a/one"
    assert "multiple model_ids" in capsys.readouterr().err


# ---------------------------------------------------------------- degeneration


def test_every_reason_label_is_declared():
    assert DEGENERATION_REASONS == (
        "empty", "too_short", "word_repetition", "bigram_repetition",
    )


@pytest.mark.parametrize("text", ["", "   ", "\n\t "])
def test_an_empty_answer_is_degenerate(text):
    assert classify_degeneration(text) == (True, "empty")


@pytest.mark.parametrize("text", ["a", "a chair"])
def test_an_answer_under_three_words_is_too_short(text):
    assert classify_degeneration(text) == (True, "too_short")


def test_three_words_is_not_too_short():
    assert classify_degeneration("a b c") == (False, "")


def test_a_repeated_word_is_word_repetition():
    assert classify_degeneration("its its its surroundings") == (
        True, "word_repetition",
    )


def test_the_canonical_trailing_repeat_is_caught():
    """The 'allowed, allowed, allowed' tail is SPARC's characteristic failure.

    It only registers because the detector strips punctuation before comparing:
    on raw tokens 'allowed,' and 'allowed.' never match and the run never
    builds.
    """
    flag, reason = classify_degeneration(
        "The room has a wooden floor and a sofa. allowed, allowed, allowed."
    )

    assert (flag, reason) == (True, "word_repetition")


def test_a_repeat_one_below_the_trigger_is_not_degenerate():
    assert CONSECUTIVE_REPEAT_TRIGGER == 3
    text = (
        "A photo of its its surroundings. The lighting comes from a lamp "
        "above the desk while a fan spins overhead casting a soft shadow."
    )

    assert classify_degeneration(text) == (False, ""), (
        "two consecutive repeats is ordinary text; only three trips the trigger"
    )


def test_an_alternating_bigram_is_bigram_repetition():
    assert classify_degeneration("This The This The This The") == (
        True, "bigram_repetition",
    )


def test_a_punctuated_bigram_loop_is_caught():
    assert classify_degeneration("Description. the room. the room. the room. the room.") == (
        True, "bigram_repetition",
    )


def test_ordinary_prose_is_not_degenerate():
    text = (
        "A cozy living room with a yellow accent wall and a dining area in "
        "the background. A small dog rests on the rug while sunlight streams "
        "through tall windows."
    )

    assert classify_degeneration(text) == (False, "")


def test_heavy_stopword_use_is_not_degenerate():
    text = (
        "The cat sat on the mat near the chair beside the table under the "
        "window with the curtain in the room with the rug."
    )

    assert classify_degeneration(text) == (False, ""), (
        "repeating 'the' is normal English, not a decoding loop"
    )


def test_every_reason_returned_is_one_of_the_declared_labels():
    samples = [
        "", "a chair", "its its its surroundings", "This The This The This The",
        "A cozy living room with a small dog on the rug.",
    ]

    for text in samples:
        flag, reason = classify_degeneration(text)
        if flag:
            assert reason in DEGENERATION_REASONS, text
        else:
            assert reason == "", text


# ---------------------------------------------------------------- id guard


def _caption(image_id: str) -> dict:
    return {"image_id": image_id, "caption": "a mug on a desk"}


def _answer(image_id: str, question_id: str) -> dict:
    return {"image_id": image_id, "question_id": question_id, "answer": "yes"}


def test_an_entry_without_ground_truth_is_fatal():
    with pytest.raises(ValueError):
        assert_entries_are_grounded(
            [_caption("adt_seq07_000123"), _caption("000000000139")],
            {"adt_seq07_000123": {"mug"}},
            id_fields=("image_id",),
            entry_noun="caption",
        )


def test_the_failure_counts_the_entries_not_the_ids():
    with pytest.raises(ValueError) as excinfo:
        assert_entries_are_grounded(
            [_caption("missing_1"), _caption("missing_1"), _caption("missing_2")],
            {"a": {"mug"}},
            id_fields=("image_id",),
            entry_noun="caption",
        )

    message = str(excinfo.value)
    assert "3 caption(s)" in message
    assert "2 image_id(s)" in message


def test_the_failure_shows_identifiers_from_both_sides():
    with pytest.raises(ValueError) as excinfo:
        assert_entries_are_grounded(
            [_caption("adt_seq07_000123")],
            {"000000000139": {"mug"}, "000000000285": {"desk"}},
            id_fields=("image_id",),
            entry_noun="caption",
        )

    message = str(excinfo.value)
    assert "adt_seq07_000123" in message, "must show the generated-side id"
    assert "000000000139" in message, "must show the ground-truth-side id"


def test_a_sample_of_ids_is_shown_not_the_whole_set():
    with pytest.raises(ValueError) as excinfo:
        assert_entries_are_grounded(
            [_caption(f"cap_{i:04d}") for i in range(50)],
            {f"gt_{i:04d}": {"mug"} for i in range(50)},
            id_fields=("image_id",),
            entry_noun="caption",
        )

    message = str(excinfo.value)
    assert "cap_0004" in message
    assert "cap_0005" not in message, "the message must stay readable"


def test_fully_grounded_entries_pass():
    assert_entries_are_grounded(
        [_caption("a"), _caption("b")],
        {"a": {"mug"}, "b": {"desk"}},
        id_fields=("image_id",),
        entry_noun="caption",
    )


def test_ground_truth_without_entries_is_not_an_error():
    assert_entries_are_grounded(
        [_caption("a")],
        {"a": {"mug"}, "b": {"desk"}, "c": {"chair"}},
        id_fields=("image_id",),
        entry_noun="caption",
    )


def test_a_composite_key_is_supported():
    questions = {("img_a", "q1"): object()}

    assert_entries_are_grounded(
        [_answer("img_a", "q1")],
        questions,
        id_fields=("image_id", "question_id"),
        entry_noun="answer",
    )


def test_a_composite_key_failure_names_the_pair():
    with pytest.raises(ValueError) as excinfo:
        assert_entries_are_grounded(
            [_answer("img_a", "q9")],
            {("img_a", "q1"): object()},
            id_fields=("image_id", "question_id"),
            entry_noun="answer",
        )

    message = str(excinfo.value)
    assert "(image_id, question_id) pair(s)" in message
    assert "q9" in message and "q1" in message
