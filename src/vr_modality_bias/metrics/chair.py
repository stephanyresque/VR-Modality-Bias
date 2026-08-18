"""CHAIR (Rohrbach et al. 2018) — caption-hallucination metrics against a
vocabulary of object categories: CHAIR_i (per-mention) and CHAIR_s
(per-caption), plus the precision/recall/F1 ingredients. The categories and
their synonyms are not defined here: the caller supplies them as a
:class:`~vr_modality_bias.data.vocabulary.Vocabulary`, so the metric is not
bound to any one dataset.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vr_modality_bias.data.vocabulary import Vocabulary

__all__ = [
    "extract_mentioned_objects",
    "chair_per_caption",
    "compute_chair_aggregate",
    "load_ground_truth_objects",
    "load_reference_caption_objects",
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

    Zero-handling:
        * Empty input -> chair_i/chair_s/precision/recall/f1 all NaN.
        * total_mentioned == 0 -> precision = 0.0; chair_i = 0.0
          (no mentions can't produce hallucinations *or* correct hits).
        * total_ground_truth == 0 -> recall = 0.0
          (nothing to recall; documented degenerate case).
        * precision + recall == 0 -> f1 = 0.0 (instead of 0/0 NaN, so
          downstream aggregations don't choke on a single edge case).

    The per_caption_results MUST come from chair_per_caption against ONE
    fixed ground-truth source. To compute against two GTs (instances vs
    captions), call this function twice with the two result lists.
    """
    n = len(per_caption_results)
    if n == 0:
        return {
            "chair_i": float("nan"),
            "chair_s": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
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

    return {
        "chair_i": chair_i,
        "chair_s": chair_s,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_captions": n,
        "n_captions_with_hallucination": n_with_halluc,
        "total_mentioned": total_mentioned,
        "total_hallucinated": total_hallucinated,
        "total_correct": total_correct,
        "total_ground_truth": total_ground_truth,
    }


def load_ground_truth_objects(instances_path: Path) -> dict[str, set[str]]:
    """Read ``instances_val2017.json`` -> ``{image_id_str: set(category_name)}``.

    image_id is returned as the **zero-padded 12-digit string** matching the
    MSCOCO file-naming convention (e.g. ``"000000000139"``) so it can be
    keyed directly off the file stem.

    This is the "GT-A" definition used historically by CHAIR: every COCO
    category whose instance is detection-annotated in the image. Strict —
    captures objects that are physically present but might not be the
    focus of any human description.
    """
    with Path(instances_path).open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    cat_id_to_name = {int(c["id"]): str(c["name"]) for c in data["categories"]}

    image_to_objects: dict[str, set[str]] = {}
    for ann in data["annotations"]:
        img_id_str = f"{int(ann['image_id']):012d}"
        cat_name = cat_id_to_name[int(ann["category_id"])]
        image_to_objects.setdefault(img_id_str, set()).add(cat_name)

    # Also make sure every image in the index has an entry (possibly empty)
    # so look-ups for images-without-annotations don't raise KeyError.
    for img in data.get("images", []):
        img_id_str = f"{int(img['id']):012d}"
        image_to_objects.setdefault(img_id_str, set())

    return image_to_objects


def load_reference_caption_objects(
    captions_path: Path,
    vocabulary: Vocabulary,
) -> dict[str, set[str]]:
    """Read ``captions_val2017.json`` -> ``{image_id_str: set(category_name)}``.

    This is the "GT-B" definition used by the SPARC paper: every
    ``vocabulary`` category mentioned in any of the ~5 reference captions
    humans wrote for the image (union across captions). Aligned with what
    humans chose to describe -- the recall denominator under GT-B is "did the
    model name the things humans named?", not "did it name every
    physically-present object?".

    Implementation: for each image, take the union of
    ``extract_mentioned_objects(caption, vocabulary)`` over all reference
    captions. Same vocabulary as the per-caption extraction (no
    special-casing).

    image_id format: zero-padded 12-digit string, same as
    :func:`load_ground_truth_objects`.

    Like :func:`load_ground_truth_objects`, images that appear in
    ``images`` but have no captions get an empty-set entry so look-ups
    never raise.
    """
    with Path(captions_path).open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    image_to_objects: dict[str, set[str]] = {}
    for ann in data["annotations"]:
        img_id_str = f"{int(ann['image_id']):012d}"
        caption = str(ann.get("caption", ""))
        if not caption.strip():
            continue
        mentioned = extract_mentioned_objects(caption, vocabulary)
        image_to_objects.setdefault(img_id_str, set()).update(mentioned)

    for img in data.get("images", []):
        img_id_str = f"{int(img['id']):012d}"
        image_to_objects.setdefault(img_id_str, set())

    return image_to_objects
