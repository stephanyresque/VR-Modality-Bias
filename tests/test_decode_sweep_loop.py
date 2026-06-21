from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def detect_loop():
    from vr_modality_bias.data.captions import detect_loop as fn
    return fn


# ---------------------------------------------------------------- tests
# Pad short captions to at least 6 words because detect_loop bails out on
# anything too short (no signal in a 3-word caption).


def test_normal_prose_no_loop(detect_loop):
    text = (
        "A cozy living room with a yellow accent wall and a dining area in "
        "the background. A small dog rests on the rug while sunlight streams "
        "through tall windows."
    )
    flag, why = detect_loop(text)
    assert flag is False, f"false positive on normal prose: {why!r}"


def test_canonical_trailing_repeat_is_caught(detect_loop):
    """The 'allowed, allowed, allowed' degeneration must be flagged."""
    text = (
        "The room has a wooden floor and a sofa. allowed, allowed, allowed, allowed."
    )
    flag, why = detect_loop(text)
    assert flag, "missed the canonical trailing-repeat degeneration"
    assert "allowed" in why.lower()


def test_caught_with_intervening_punctuation(detect_loop):
    """Punctuation between repeats shouldn't hide the loop."""
    text = "Description. the room. the room. the room. the room."
    flag, _ = detect_loop(text)
    assert flag


def test_tail_window_dominance_is_caught(detect_loop):
    """A non-stopword saturating the last 20 tokens flags loop."""
    text = (
        "The image shows a wooden table holding several plates and napkins "
        "and forks plates plates plates plates plates plates plates plates."
    )
    flag, why = detect_loop(text)
    assert flag, f"missed unigram-dominance loop: {why!r}"
    assert "plates" in why


def test_stopwords_dont_false_fire(detect_loop):
    """Heavy 'the'/'a' usage is normal English and must NOT flag."""
    text = (
        "The cat sat on the mat near the chair beside the table under the "
        "window with the curtain in the room with the rug."
    )
    flag, _ = detect_loop(text)
    assert flag is False


def test_very_short_caption_returns_false(detect_loop):
    flag, _ = detect_loop("a b c")
    assert flag is False


def test_empty_caption_returns_false(detect_loop):
    flag, _ = detect_loop("")
    assert flag is False


def test_double_word_repeat_below_threshold_no_flag(detect_loop):
    """A single doubled word ('its its') is fine — the threshold is ≥3."""
    text = (
        "A photo of its its surroundings. The lighting comes from a lamp "
        "above the desk while a fan spins overhead casting a soft shadow."
    )
    flag, _ = detect_loop(text)
    assert flag is False, "threshold of ≥3 consecutive is intentional — 2 is fine"
