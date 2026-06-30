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

import json
import math
from pathlib import Path

import pytest

from vr_modality_bias.metrics.chair import (
    chair_per_caption,
    compute_chair_aggregate,
    extract_mentioned_objects,
    load_ground_truth_objects,
)


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


# ---------------------------------------------------------------- gt loader


def test_load_ground_truth_objects_keys_are_zero_padded(tmp_path: Path):
    """image_id keys must be 12-digit zero-padded strings matching COCO file stems."""
    fake_instances = {
        "categories": [
            {"id": 1, "name": "person"},
            {"id": 18, "name": "dog"},
        ],
        "annotations": [
            {"image_id": 139, "category_id": 1},
            {"image_id": 139, "category_id": 18},
            {"image_id": 285, "category_id": 1},
        ],
        "images": [
            {"id": 139}, {"id": 285}, {"id": 9999},
        ],
    }
    p = tmp_path / "instances_val2017.json"
    p.write_text(json.dumps(fake_instances), encoding="utf-8")
    gt = load_ground_truth_objects(p)

    assert "000000000139" in gt
    assert gt["000000000139"] == {"person", "dog"}
    assert gt["000000000285"] == {"person"}
    # Image with no annotations still appears with an empty set.
    assert gt["000000009999"] == set()


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


# ================================================================
# GT-B loader (load_reference_caption_objects)
# Mini synthetic captions_val2017.json: 1 image with 3 ref captions,
# one image with 0 captions but listed in images.
# ================================================================


def test_load_reference_caption_objects_unions_per_image(tmp_path):
    from vr_modality_bias.metrics.chair import load_reference_caption_objects

    fake = {
        "images": [
            {"id": 139, "file_name": "139.jpg"},
            {"id": 285, "file_name": "285.jpg"},   # 0 captions -> empty set
            {"id": 632, "file_name": "632.jpg"},
        ],
        "annotations": [
            # image 139: 3 ref captions, mention {cat, bed, dog}
            {"image_id": 139, "id": 1, "caption": "a cat sleeps on the bed"},
            {"image_id": 139, "id": 2, "caption": "a dog and a cat together"},
            {"image_id": 139, "id": 3, "caption": "the cat is on the pillow"},
            # image 632: 2 ref captions, mention {bicycle, person}
            {"image_id": 632, "id": 4, "caption": "a man riding a bicycle"},
            {"image_id": 632, "id": 5, "caption": "two cyclists on the road"},  # bike + person via synonyms
        ],
    }
    p = tmp_path / "captions_val2017.json"
    p.write_text(json.dumps(fake), encoding="utf-8")

    out = load_reference_caption_objects(p)
    assert out["000000000139"] == {"cat", "bed", "dog"}
    assert out["000000000632"] == {"bicycle", "person"}
    # image 285 listed but no captions -> empty set, NOT KeyError on lookup
    assert out["000000000285"] == set()


def test_load_reference_caption_objects_uses_zero_padded_image_id(tmp_path):
    """ID keys must be 12-digit zero-padded strings, matching the file-stem convention."""
    from vr_modality_bias.metrics.chair import load_reference_caption_objects

    fake = {
        "images": [{"id": 7, "file_name": "7.jpg"}],
        "annotations": [{"image_id": 7, "id": 1, "caption": "a dog"}],
    }
    p = tmp_path / "captions_val2017.json"
    p.write_text(json.dumps(fake), encoding="utf-8")
    out = load_reference_caption_objects(p)
    # zero-padded to 12 digits
    assert "000000000007" in out
    assert "7" not in out
    assert out["000000000007"] == {"dog"}
