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
