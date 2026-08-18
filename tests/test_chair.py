"""Tests for :mod:`vr_modality_bias.metrics.chair`.

The score itself is trivial arithmetic; the hard part is the **noun
recogniser**. Specifically:

  * Whole-word matching: ``cat`` is in ``"a cat sleeps"`` but NOT in
    ``"a category of pets"``.
  * Multi-word synonyms: ``hot dog`` matches in ``"a hot dog on a plate"``.
  * Plurals: ``cats`` → ``cat``, ``children`` → ``person``.
  * Non-COCO words: ``unicorn`` is NOT counted at all (zero-influence).
  * Aggregation: an off-by-one in CHAIR_i / CHAIR_s ratio changes a
    headline number, so the basic arithmetic is asserted.
"""

from __future__ import annotations

import functools
import math
from pathlib import Path

import pytest

from vr_modality_bias.data.vocabulary import load_vocabulary
from vr_modality_bias.metrics.chair import (
    chair_per_caption as _chair_per_caption,
    compute_chair_aggregate,
    extract_mentioned_objects as _extract_mentioned_objects,
)

# The COCO-80 vocabulary used to live inside chair.py as two module constants.
# It is now a test fixture, bound here so the assertions below stay exactly as
# they were written against the hard-coded version — that is what makes them
# evidence that extracting the vocabulary did not change behaviour.
COCO80 = load_vocabulary(Path(__file__).parent / "fixtures" / "vocab_coco80.json")

extract_mentioned_objects = functools.partial(
    _extract_mentioned_objects, vocabulary=COCO80
)
chair_per_caption = functools.partial(_chair_per_caption, vocabulary=COCO80)


# ---------------------------------------------------------------- extraction


def test_extract_recognises_canonical_name():
    assert extract_mentioned_objects("a bicycle on the street") == {"bicycle"}


def test_extract_recognises_synonym():
    assert extract_mentioned_objects("a man riding a bike") == {"person", "bicycle"}


def test_extract_recognises_plural():
    out = extract_mentioned_objects("two cats and three dogs")
    assert out == {"cat", "dog"}


def test_extract_recognises_multi_word():
    out = extract_mentioned_objects("a hot dog on a plate")
    assert "hot dog" in out
    # Sanity: NOT a regular "dog" (the bigram match wins; the word "dog"
    # would also single-match, but both map to canonical categories).
    # Both "hot dog" and "dog" are in COCO-80 with distinct meanings; the
    # match registers each independently when the synonym hits.
    assert "dog" in out  # single word "dog" is also present


def test_extract_no_partial_word_match():
    """``cat`` should NOT match inside ``category``, ``catch``, etc."""
    out = extract_mentioned_objects("a category of catchy things")
    assert out == set()


def test_extract_ignores_non_coco_words():
    """Words with no COCO synonym contribute nothing to the mentioned set."""
    out = extract_mentioned_objects("a man riding a unicorn")
    # "man" → person, "unicorn" → nothing.
    assert out == {"person"}


def test_extract_handles_punctuation():
    out = extract_mentioned_objects("A man, a cat, and a dog.")
    assert out == {"person", "cat", "dog"}


def test_extract_case_insensitive():
    assert extract_mentioned_objects("A Person walks a Dog.") == {"person", "dog"}


def test_extract_empty_caption():
    assert extract_mentioned_objects("") == set()
    assert extract_mentioned_objects("   ") == set()


# ---------------------------------------------------------------- per-caption


def test_no_hallucination_when_all_in_ground_truth():
    out = chair_per_caption("a man riding a bicycle", {"person", "bicycle"})
    assert out["mentioned"] == {"person", "bicycle"}
    assert out["hallucinated"] == set()
    assert out["n_hallucinated"] == 0
    assert out["has_hallucination"] is False


def test_hallucination_when_mentioned_but_not_in_gt():
    out = chair_per_caption("a man with a car", {"person"})
    assert out["mentioned"] == {"person", "car"}
    assert out["hallucinated"] == {"car"}
    assert out["n_hallucinated"] == 1
    assert out["has_hallucination"] is True


def test_unknown_word_does_not_count_as_hallucination():
    """If the caption invents an object NOT in COCO-80, CHAIR can't see it
    — which is correct: CHAIR only scores hallucination of known categories."""
    out = chair_per_caption("a man with a unicorn", {"person"})
    assert "person" in out["mentioned"]
    assert out["hallucinated"] == set()
    assert out["has_hallucination"] is False


# ---------------------------------------------------------------- aggregate


def test_aggregate_basic_arithmetic():
    """3 captions: clean / 1-halluc / clean → CHAIR_i = 1/5, CHAIR_s = 1/3."""
    per_caption = [
        {"n_mentioned": 2, "n_hallucinated": 0, "has_hallucination": False},
        {"n_mentioned": 2, "n_hallucinated": 1, "has_hallucination": True},
        {"n_mentioned": 1, "n_hallucinated": 0, "has_hallucination": False},
    ]
    agg = compute_chair_aggregate(per_caption)
    assert agg["chair_i"] == pytest.approx(1 / 5)
    assert agg["chair_s"] == pytest.approx(1 / 3)
    assert agg["n_captions"] == 3
    assert agg["n_captions_with_hallucination"] == 1
    assert agg["total_mentioned"] == 5
    assert agg["total_hallucinated"] == 1


def test_aggregate_empty_input_returns_nan():
    agg = compute_chair_aggregate([])
    assert math.isnan(agg["chair_i"])
    assert math.isnan(agg["chair_s"])
    assert agg["n_captions"] == 0


def test_aggregate_zero_mentions_does_not_divide_by_zero():
    """If no caption mentions any object, CHAIR_i is 0 (not NaN, not error)."""
    per_caption = [
        {"n_mentioned": 0, "n_hallucinated": 0, "has_hallucination": False},
        {"n_mentioned": 0, "n_hallucinated": 0, "has_hallucination": False},
    ]
    agg = compute_chair_aggregate(per_caption)
    assert agg["chair_i"] == 0.0
    assert agg["chair_s"] == 0.0


def test_aggregate_all_hallucinated():
    """Edge: every caption is fully hallucinated."""
    per_caption = [
        {"n_mentioned": 1, "n_hallucinated": 1, "has_hallucination": True},
        {"n_mentioned": 2, "n_hallucinated": 2, "has_hallucination": True},
    ]
    agg = compute_chair_aggregate(per_caption)
    assert agg["chair_i"] == pytest.approx(1.0)
    assert agg["chair_s"] == pytest.approx(1.0)


# ================================================================
# Precision / Recall / F1 — added on top of CHAIR (recall-GT block).
#
# The new compute_chair_aggregate exposes three classifier-style
# scores derived from the same per-caption decomposition. Test them
# against a hand-worked example AND check the identity precision =
# 1 - chair_i (the orchestrator relies on it).
# ================================================================


def test_chair_per_caption_carries_correct_and_gt_sizes():
    """Per-caption result must expose ``correct`` and ``n_ground_truth``."""
    out = chair_per_caption(
        "a cat and a dog on the bed",
        ground_truth_objects={"cat", "bed", "person"},  # GT
    )
    assert out["mentioned"] == {"cat", "dog", "bed"}
    assert out["correct"] == {"cat", "bed"}
    assert out["hallucinated"] == {"dog"}
    assert out["n_correct"] == 2
    assert out["n_ground_truth"] == 3


def test_aggregate_precision_recall_f1_known_case():
    """mentioned={a,b,c}, GT={a,b,d} -> correct=2, precision=2/3, recall=2/3, f1=2/3.

    Built as a single per-caption result so the aggregate math is the same
    as the per-caption math.
    """
    per_caption = [{
        "mentioned": {"a", "b", "c"},
        "hallucinated": {"c"},
        "correct": {"a", "b"},
        "n_mentioned": 3,
        "n_hallucinated": 1,
        "n_correct": 2,
        "n_ground_truth": 3,
        "has_hallucination": True,
    }]
    agg = compute_chair_aggregate(per_caption)
    assert agg["total_mentioned"] == 3
    assert agg["total_hallucinated"] == 1
    assert agg["total_correct"] == 2
    assert agg["total_ground_truth"] == 3
    assert math.isclose(agg["precision"], 2 / 3, rel_tol=1e-9)
    assert math.isclose(agg["recall"], 2 / 3, rel_tol=1e-9)
    assert math.isclose(agg["f1"], 2 / 3, rel_tol=1e-9)


def test_precision_equals_one_minus_chair_i():
    """The orchestrator uses this identity. Check it on a real-shaped sample."""
    per_caption = [
        chair_per_caption("a cat and a dog", {"cat", "person"}),       # m=2 h=1 c=1
        chair_per_caption("two birds on a fence", {"bird"}),           # m=1 h=0 c=1
        chair_per_caption("a sandwich and a pizza", {"chair"}),        # m=2 h=2 c=0
    ]
    agg = compute_chair_aggregate(per_caption)
    # precision must match 1 - chair_i to floating-point precision
    assert math.isclose(agg["precision"], 1.0 - agg["chair_i"], rel_tol=1e-9)
    # spot-check the numbers: total_mentioned=5, total_correct=2, total_hallucinated=3
    assert agg["total_mentioned"] == 5
    assert agg["total_correct"] == 2
    assert agg["total_hallucinated"] == 3
    assert math.isclose(agg["precision"], 2 / 5, rel_tol=1e-9)
    assert math.isclose(agg["chair_i"], 3 / 5, rel_tol=1e-9)


def test_aggregate_zero_mentions_gives_zero_precision_zero_chair_i():
    per_caption = [{
        "mentioned": set(), "hallucinated": set(), "correct": set(),
        "n_mentioned": 0, "n_hallucinated": 0, "n_correct": 0,
        "n_ground_truth": 5, "has_hallucination": False,
    }]
    agg = compute_chair_aggregate(per_caption)
    assert agg["precision"] == 0.0
    assert agg["chair_i"] == 0.0
    # recall = 0/5 = 0; F1 of (0, 0) -> 0.0 by our guard
    assert agg["recall"] == 0.0
    assert agg["f1"] == 0.0


def test_aggregate_zero_ground_truth_gives_zero_recall():
    """No GT objects -> nothing to recall. recall = 0.0, F1 = 0.0."""
    per_caption = [{
        "mentioned": {"a", "b"}, "hallucinated": {"a", "b"}, "correct": set(),
        "n_mentioned": 2, "n_hallucinated": 2, "n_correct": 0,
        "n_ground_truth": 0, "has_hallucination": True,
    }]
    agg = compute_chair_aggregate(per_caption)
    assert agg["recall"] == 0.0
    assert agg["f1"] == 0.0
    assert agg["precision"] == 0.0  # 0 correct / 2 mentioned
    assert agg["chair_i"] == 1.0


def test_aggregate_empty_input_returns_nan_for_metrics():
    agg = compute_chair_aggregate([])
    for k in ("chair_i", "chair_s", "precision", "recall", "f1"):
        assert math.isnan(agg[k])


# ---------------------------------------------------------------- cover

# Cover is AMBER's macro recall: the mean of the per-caption ratios, as
# opposed to `recall`, which pools the totals and divides once. Every test
# here uses ground truths of DIFFERENT sizes, because with equal sizes the
# two collapse onto the same number and prove nothing.


def _result(*, n_correct: int, n_ground_truth: int, n_mentioned: int = 1) -> dict:
    return {
        "n_mentioned": n_mentioned,
        "n_hallucinated": 0,
        "has_hallucination": False,
        "n_correct": n_correct,
        "n_ground_truth": n_ground_truth,
    }


def test_cover_is_the_mean_of_the_per_caption_ratios():
    agg = compute_chair_aggregate([
        _result(n_correct=1, n_ground_truth=1),
        _result(n_correct=1, n_ground_truth=4),
    ])

    assert agg["cover"] == pytest.approx((1 / 1 + 1 / 4) / 2)


def test_cover_and_recall_disagree_when_ground_truths_differ_in_size():
    per_caption = [
        _result(n_correct=1, n_ground_truth=1),
        _result(n_correct=1, n_ground_truth=4),
    ]

    agg = compute_chair_aggregate(per_caption)

    assert agg["recall"] == pytest.approx(2 / 5), "recall must stay micro"
    assert agg["cover"] == pytest.approx(0.625), "cover must be macro"
    assert agg["cover"] != pytest.approx(agg["recall"]), (
        "if these two ever agree on unequal ground-truth sizes, one of them "
        "stopped being the average it is supposed to be."
    )


def test_cover_equals_recall_when_every_ground_truth_has_the_same_size():
    agg = compute_chair_aggregate([
        _result(n_correct=1, n_ground_truth=2),
        _result(n_correct=2, n_ground_truth=2),
    ])

    assert agg["cover"] == pytest.approx(agg["recall"])


def test_a_caption_with_an_empty_ground_truth_is_left_out_of_cover():
    agg = compute_chair_aggregate([
        _result(n_correct=1, n_ground_truth=4),
        _result(n_correct=0, n_ground_truth=0, n_mentioned=0),
    ])

    assert agg["cover"] == pytest.approx(0.25), (
        "covering nothing is undefined, not zero; averaging a 0.0 in would "
        "drag the metric down for items the dataset simply did not annotate."
    )


def test_cover_is_nan_when_no_caption_has_a_ground_truth():
    agg = compute_chair_aggregate([_result(n_correct=0, n_ground_truth=0, n_mentioned=0)])

    assert math.isnan(agg["cover"])


def test_cover_is_nan_on_empty_input():
    assert math.isnan(compute_chair_aggregate([])["cover"])
