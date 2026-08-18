"""CHAIR (Rohrbach et al. 2018) — caption-hallucination metrics against a
vocabulary of object categories: CHAIR_i (per-mention) and CHAIR_s
(per-caption), the precision/recall/F1 ingredients, and AMBER's Cover. The
categories and their synonyms are not defined here: the caller supplies them
as a :class:`~vr_modality_bias.data.vocabulary.Vocabulary`, so the metric is
not bound to any one dataset. Reading the ground truth is not this module's
job either — it scores whatever ``{image_id: set(object)}`` mapping it is
handed.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vr_modality_bias.data.vocabulary import Vocabulary

__all__ = [
    "extract_mentioned_objects",
    "chair_per_caption",
    "compute_chair_aggregate",
]


def _normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace, pad with spaces.

    The leading/trailing spaces let us do whole-word substring matches like
    ``" cat " in text`` without false hits on ``"cattle"``.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s'-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return f" {text} "


def extract_mentioned_objects(caption: str, vocabulary: Vocabulary) -> set[str]:
    """Return the ``vocabulary`` categories mentioned in ``caption``.

    Multi-word synonyms (``hot dog``, ``fire hydrant``) match correctly
    because we use space-padded substring search on the normalised text.
    Single-word synonyms also need surrounding spaces, so ``"cat"`` does
    NOT match inside ``"category"``.
    """
    syn_index = vocabulary.synonym_to_category

    text = _normalise(caption)
    mentioned: set[str] = set()
    for syn, cat in syn_index.items():
        if cat in mentioned:
            continue
        if f" {syn} " in text:
            mentioned.add(cat)
    return mentioned


def chair_per_caption(
    caption: str,
    ground_truth_objects: set[str],
    vocabulary: Vocabulary,
) -> dict:
    """Per-caption decomposition (CHAIR + precision/recall ingredients).

    Returns a dict with:
        mentioned        : set of ``vocabulary`` categories named in the caption
        hallucinated     : mentioned − ground_truth   (false positives)
        correct          : mentioned ∩ ground_truth   (true positives)
        n_mentioned      : len(mentioned)
        n_hallucinated   : len(hallucinated)
        n_correct        : len(correct)
        n_ground_truth   : len(ground_truth)          (denominator for recall)
        has_hallucination: bool, True iff any object is hallucinated

    The ``correct``/``n_correct``/``n_ground_truth`` fields are post-port
    additions used by :func:`compute_chair_aggregate` to derive precision
    (= 1 − CHAIR_i), recall, and F1.
    """
    gt_set = set(ground_truth_objects)
    mentioned = extract_mentioned_objects(caption, vocabulary)
    hallucinated = mentioned - gt_set
    correct = mentioned & gt_set
    return {
        "mentioned": mentioned,
        "hallucinated": hallucinated,
        "correct": correct,
        "n_mentioned": len(mentioned),
        "n_hallucinated": len(hallucinated),
        "n_correct": len(correct),
        "n_ground_truth": len(gt_set),
        "has_hallucination": len(hallucinated) > 0,
    }


def compute_chair_aggregate(per_caption_results: list[dict]) -> dict:
    """CHAIR_s/CHAIR_i + precision/recall/F1 over a set of per-caption results.

    Definitions, with totals taken over the full set of captions:

        chair_i   = total_hallucinated / total_mentioned      (lower is better)
        chair_s   = n_with_halluc / n_captions                (lower is better)
        precision = total_correct / total_mentioned           (= 1 - chair_i)
        recall    = total_correct / total_ground_truth        (higher is better)
        f1        = 2 * P * R / (P + R)                       (harmonic mean)

    Plus one metric that is NOT a ratio of totals:

        cover     = mean over captions of (n_correct / n_ground_truth)

    ``cover`` is AMBER's Cover. It differs from ``recall`` on purpose:
    ``recall`` pools every caption's counts and divides once (micro-average),
    so images with large ground truths dominate it; ``cover`` averages the
    per-caption ratios (macro-average), so every image weighs the same. The
    two agree only when all captions share one ground-truth size.

    ``chair_s`` is also AMBER's Hal — the fraction of responses with at least
    one hallucinated object. It is not recomputed under that name.

    Zero-handling:
        * Empty input -> chair_i/chair_s/precision/recall/f1/cover all NaN.
        * total_mentioned == 0 -> precision = 0.0; chair_i = 0.0
          (no mentions can't produce hallucinations *or* correct hits).
        * total_ground_truth == 0 -> recall = 0.0
          (nothing to recall; documented degenerate case).
        * precision + recall == 0 -> f1 = 0.0 (instead of 0/0 NaN, so
          downstream aggregations don't choke on a single edge case).
        * A caption whose ground truth is empty contributes nothing to
          ``cover`` — covering nothing is undefined, not zero. When every
          caption is in that state, ``cover`` is NaN.

    The per_caption_results MUST come from chair_per_caption against ONE
    fixed ground-truth source.
    """
    n = len(per_caption_results)
    if n == 0:
        return {
            "chair_i": float("nan"),
            "chair_s": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "cover": float("nan"),
            "n_captions": 0,
            "n_captions_with_hallucination": 0,
            "total_mentioned": 0,
            "total_hallucinated": 0,
            "total_correct": 0,
            "total_ground_truth": 0,
        }
    total_mentioned = sum(int(r["n_mentioned"]) for r in per_caption_results)
    total_hallucinated = sum(int(r["n_hallucinated"]) for r in per_caption_results)
    total_correct = sum(int(r.get("n_correct", 0)) for r in per_caption_results)
    total_ground_truth = sum(int(r.get("n_ground_truth", 0)) for r in per_caption_results)
    n_with_halluc = sum(1 for r in per_caption_results if r["has_hallucination"])

    chair_i = (total_hallucinated / total_mentioned) if total_mentioned > 0 else 0.0
    chair_s = n_with_halluc / n
    precision = (total_correct / total_mentioned) if total_mentioned > 0 else 0.0
    recall = (total_correct / total_ground_truth) if total_ground_truth > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    per_caption_cover = [
        int(r.get("n_correct", 0)) / int(r["n_ground_truth"])
        for r in per_caption_results
        if int(r.get("n_ground_truth", 0)) > 0
    ]
    cover = (
        sum(per_caption_cover) / len(per_caption_cover)
        if per_caption_cover
        else float("nan")
    )

    return {
        "chair_i": chair_i,
        "chair_s": chair_s,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cover": cover,
        "n_captions": n,
        "n_captions_with_hallucination": n_with_halluc,
        "total_mentioned": total_mentioned,
        "total_hallucinated": total_hallucinated,
        "total_correct": total_correct,
        "total_ground_truth": total_ground_truth,
    }
