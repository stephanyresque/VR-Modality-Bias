#!/usr/bin/env python
"""Phase-1 gate: equivalence between single-pass TF and forced decoding (SPARC OFF).

EXPERIMENT.md §4.4 — before we trust *any* SPARC ON / OFF comparison, we
have to show that the new collection path
(:func:`vr_modality_bias.experiment.forced_decoding.collect_forced_decoding`)
reproduces the diagnostic when SPARC is **off**. If it doesn't, every later
result is unattributable.

What this script does
---------------------
For each of ``--limit`` images:
    1. Generate ``caption_ref`` once (free generation, seed deterministic).
    2. **Path A (legacy):** ``model.run_teacher_forcing(...)`` for the real
       image and the noise image → KL matrix → ``head_tail_ratio``.
    3. **Path B (new, SPARC OFF):** ``collect_forced_decoding(...)`` for
       both conditions → KL matrix → ``head_tail_ratio``.
    4. Compute per-image deltas: ``|Δhtr|`` and the median relative
       difference of the deep-block KL curve.

Acceptance (decided in §12.5)
-----------------------------
The check **passes** when either of:
    * ``|Δhtr| ≤ 0.02`` for at least 45 of 50 images, OR
    * Median relative difference of the deep KL curve ≤ 1% per image.

Treat these as a ceiling — the math is the same except for cache/position
handling, so the expectation is to come in much tighter than the ceiling.
If we exceed the ceiling, the first place to look is the predictive-state
index alignment in ``forced_decoding`` (the off-by-one the unit test was
built to catch).

CLI
---
    python scripts/13_equivalence_check.py --config configs/baseline.yaml --limit 5
    python scripts/13_equivalence_check.py --config configs/baseline.yaml --limit 50
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from loguru import logger
from PIL import Image
from pyprojroot import here

try:
    from vr_modality_bias.data.perturbations import noise_image_uniform
    from vr_modality_bias.data.prompts import get_prompt
    from vr_modality_bias.experiment.forced_decoding import collect_forced_decoding
    from vr_modality_bias.metrics.kl import compute_kl_matrix
    from vr_modality_bias.metrics.residual import deep_block, head_tail_ratio
    from vr_modality_bias.models.registry import build_model
    from vr_modality_bias.utils.config import load_config
    from vr_modality_bias.utils.device import resolve_dtype, select_device
    from vr_modality_bias.utils.seeds import derive_image_seed
except ModuleNotFoundError:
    sys.path.insert(0, str(here()))

    from src.vr_modality_bias.data.perturbations import noise_image_uniform
    from src.vr_modality_bias.data.prompts import get_prompt
    from src.vr_modality_bias.experiment.forced_decoding import collect_forced_decoding
    from src.vr_modality_bias.metrics.kl import compute_kl_matrix
    from src.vr_modality_bias.metrics.residual import deep_block, head_tail_ratio
    from src.vr_modality_bias.models.registry import build_model
    from src.vr_modality_bias.utils.config import load_config
    from src.vr_modality_bias.utils.device import resolve_dtype, select_device
    from src.vr_modality_bias.utils.seeds import derive_image_seed


# Acceptance thresholds (EXPERIMENT.md §12.5).
HTR_TOL = 0.02
HTR_MIN_FRACTION = 45 / 50  # ≥45 in 50 images
CURVE_REL_TOL = 0.01        # 1%


def _iso_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def _kl_for_pair(lm_head, result_A, result_B, *, top_k: int) -> np.ndarray:
    return compute_kl_matrix(
        lm_head,
        result_A.hidden_states,
        result_B.hidden_states,
        caption_start=int(result_A.caption_start),
        caption_len=int(result_A.caption_len),
        top_k=top_k,
    )


def _deep_curve(kl_matrix: np.ndarray) -> np.ndarray:
    n_layers = kl_matrix.shape[0]
    l0, l1 = deep_block(n_layers)
    return kl_matrix[l0:l1, :].astype(np.float64).mean(axis=0)


def _median_relative_curve_diff(curve_old: np.ndarray, curve_new: np.ndarray) -> float:
    """Median over positions of ``|new - old| / mean(|old|)`` — gives a single
    "fractional drift" number per image without being dominated by outliers."""
    denom = float(np.mean(np.abs(curve_old)))
    if denom <= 0.0 or not np.isfinite(denom):
        return float("nan")
    return float(np.median(np.abs(curve_new - curve_old)) / denom)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--limit", type=int, default=5,
        help="Number of images to compare. Use 50 for the official check.",
    )
    parser.add_argument(
        "--report-path", type=Path, default=None,
        help="Where to write equivalence_report.json (default: under config-named run dir).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # --- model load ---
    model_wrapper = build_model(cfg["model"]["key"])
    model_wrapper.model_id = str(cfg["model"]["model_id"])
    dtype = resolve_dtype(str(cfg["model"]["dtype"]))
    device = select_device("cuda")
    if hasattr(model_wrapper, "_dtype"):
        model_wrapper._dtype = dtype  # noqa: SLF001
    logger.info(f"Loading model {model_wrapper.model_id} on {device} (dtype={dtype})...")
    model_wrapper.load(device)
    lm_head = model_wrapper.get_lm_head()
    logger.info(f"Model loaded. n_layers={model_wrapper.n_layers}")

    # --- images ---
    images_dir = cfg["dataset"]["images_dir"]
    image_files = sorted(glob.glob(f"{images_dir}{os.sep}*.jpg"))[: args.limit]
    if not image_files:
        logger.error(f"No images under {images_dir}")
        return 1
    logger.info(f"Comparing on {len(image_files)} image(s).")

    # --- generation params ---
    prompt = get_prompt(str(cfg["task"]["prompt_key"]))
    seed_global = int(cfg["run"]["seed_global"])
    max_new_tokens = int(cfg["generation"]["max_new_tokens"])
    gen_kwargs = {
        "do_sample": bool(cfg["generation"]["do_sample"]),
        "temperature": float(cfg["generation"]["temperature"]),
        "top_p": float(cfg["generation"]["top_p"]),
        "repetition_penalty": float(cfg["generation"]["repetition_penalty"]),
    }
    top_k = int(cfg["metrics"]["logits_top_k"])

    # --- iterate ---
    rows: list[dict] = []
    for image_path in image_files:
        image_id = Path(image_path).stem
        with Image.open(image_path) as raw:
            image = raw.convert("RGB")
        noise_seed = derive_image_seed(seed_global, image_id)
        noise_img = noise_image_uniform(image, seed=int(noise_seed))

        # 1. caption_ref (free generation, deterministic per image_id)
        caption_ref = model_wrapper.generate_caption(
            image=image, prompt=prompt,
            max_new_tokens=max_new_tokens,
            seed=noise_seed,
            generation_kwargs=gen_kwargs,
        )
        if not caption_ref.strip():
            logger.warning(f"[{image_id}] empty caption — skipping.")
            continue

        # 2. path A (legacy single-pass)
        try:
            old_A = model_wrapper.run_teacher_forcing(image, prompt, caption_ref)
            old_B = model_wrapper.run_teacher_forcing(noise_img, prompt, caption_ref)
        except Exception as exc:
            logger.error(f"[{image_id}] run_teacher_forcing failed: {exc}")
            continue

        # 3. path B (forced decoding, SPARC OFF)
        try:
            new_A = collect_forced_decoding(model_wrapper, image, prompt, caption_ref)
            new_B = collect_forced_decoding(model_wrapper, noise_img, prompt, caption_ref)
        except Exception as exc:
            logger.error(f"[{image_id}] collect_forced_decoding failed: {exc}")
            continue

        # Sanity: shapes must match, otherwise the comparison is meaningless.
        if old_A.hidden_states.shape != new_A.hidden_states.shape:
            logger.error(
                f"[{image_id}] hidden_states shape mismatch: "
                f"old={tuple(old_A.hidden_states.shape)} new={tuple(new_A.hidden_states.shape)}"
            )
            continue
        if old_A.caption_start != new_A.caption_start or old_A.caption_len != new_A.caption_len:
            logger.error(
                f"[{image_id}] caption_start/len mismatch: "
                f"old=({old_A.caption_start},{old_A.caption_len}) "
                f"new=({new_A.caption_start},{new_A.caption_len})"
            )
            continue

        # 4. KL + head_tail_ratio for each path
        kl_old = _kl_for_pair(lm_head, old_A, old_B, top_k=top_k)
        kl_new = _kl_for_pair(lm_head, new_A, new_B, top_k=top_k)
        htr_old = head_tail_ratio(kl_old, t0=int(cfg["residual"]["t0"]))
        htr_new = head_tail_ratio(kl_new, t0=int(cfg["residual"]["t0"]))

        curve_old = _deep_curve(kl_old)
        curve_new = _deep_curve(kl_new)
        median_rel = _median_relative_curve_diff(curve_old, curve_new)

        delta_htr = (
            abs(htr_new - htr_old)
            if (htr_old == htr_old and htr_new == htr_new)  # both finite
            else float("nan")
        )
        rows.append({
            "image_id": image_id,
            "caption_len": int(old_A.caption_len),
            "htr_old": float(htr_old) if htr_old == htr_old else None,
            "htr_new": float(htr_new) if htr_new == htr_new else None,
            "abs_delta_htr": float(delta_htr) if delta_htr == delta_htr else None,
            "median_relative_curve_diff": float(median_rel) if median_rel == median_rel else None,
        })
        logger.info(
            f"[{image_id}] caption_len={int(old_A.caption_len)}  "
            f"htr_old={htr_old:.4f}  htr_new={htr_new:.4f}  "
            f"|Δhtr|={delta_htr:.4f}  med_rel_curve={median_rel:.4%}"
        )

    if not rows:
        logger.error("No usable rows — equivalence check inconclusive.")
        return 1

    # --- aggregate + verdict ---
    finite_deltas = [r["abs_delta_htr"] for r in rows if r["abs_delta_htr"] is not None]
    finite_curve_rel = [
        r["median_relative_curve_diff"] for r in rows
        if r["median_relative_curve_diff"] is not None
    ]

    n_within_htr_tol = sum(1 for d in finite_deltas if d <= HTR_TOL)
    n_within_curve_tol = sum(1 for d in finite_curve_rel if d <= CURVE_REL_TOL)
    htr_pass = (n_within_htr_tol / len(rows)) >= HTR_MIN_FRACTION
    curve_pass = (n_within_curve_tol / len(rows)) >= HTR_MIN_FRACTION  # same ratio

    verdict_pass = htr_pass or curve_pass

    summary = {
        "n_images": len(rows),
        "htr_tol": HTR_TOL,
        "n_within_htr_tol": n_within_htr_tol,
        "fraction_within_htr_tol": n_within_htr_tol / len(rows),
        "htr_pass": htr_pass,
        "median_abs_delta_htr": statistics.median(finite_deltas) if finite_deltas else None,
        "max_abs_delta_htr": max(finite_deltas) if finite_deltas else None,
        "curve_rel_tol": CURVE_REL_TOL,
        "n_within_curve_tol": n_within_curve_tol,
        "fraction_within_curve_tol": n_within_curve_tol / len(rows),
        "curve_pass": curve_pass,
        "median_curve_relative_diff": statistics.median(finite_curve_rel) if finite_curve_rel else None,
        "verdict_pass": verdict_pass,
        "timestamp_iso": _iso_now(),
        "model_id": model_wrapper.model_id,
        "config_path": str(args.config),
    }

    report = {"summary": summary, "rows": rows}
    report_path = args.report_path or Path(
        f"results/equivalence_{Path(args.config).stem}_{_iso_now().replace(':', '-')}.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    logger.info("=" * 70)
    logger.info("EQUIVALENCE REPORT")
    logger.info("=" * 70)
    logger.info(f"images               : {summary['n_images']}")
    logger.info(
        f"|Δhtr| within {HTR_TOL}    : {n_within_htr_tol}/{len(rows)} "
        f"({summary['fraction_within_htr_tol']*100:.1f}%)  -> {'PASS' if htr_pass else 'FAIL'}"
    )
    logger.info(
        f"median curve rel diff: {summary['median_curve_relative_diff']:.4%}; "
        f"≤{CURVE_REL_TOL*100:.1f}% in {n_within_curve_tol}/{len(rows)}  "
        f"-> {'PASS' if curve_pass else 'FAIL'}"
    )
    logger.info(f"VERDICT              : {'PASS' if verdict_pass else 'FAIL'}")
    logger.info(f"report saved to      : {report_path}")
    return 0 if verdict_pass else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover (operator-side script)
        logger.debug(f"exception: {exc}")
        logger.error(traceback.format_exc())
        raise SystemExit(1)
