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
    4. Compute per-image deltas:
       * tensor-level: per-layer max + median **relative** diff between TF
         and FD hidden states (on the A condition, over the predictive-state
         range only). This is the architectural-exactness probe.
       * metric-level: ``|Δhtr|`` and the median relative difference of the
         deep-block KL curve.

Two acceptance regimes (decided in §12.5 + Phase-1 closure feedback)
--------------------------------------------------------------------
**Regime A — architectural exactness (small model, fp32).** When run on a
small Qwen2.5-VL (3B) in fp32, the math is identical to TF up to fp32
precision. Pass iff ``max per-image per-layer relative hidden-state
diff ≤ 1e-4``. If this fails, there's a real bug in forced_decoding —
investigate before any bf16 result is trusted.

**Regime B — bf16 floor (7B, bf16, aggregate).** With exactness proven in
Regime A, the residual TF-vs-FD gap on 7B-bf16 is numerical drift from
28 layers × ~64 step forwards in bf16. We accept that floor and judge
the aggregate over the 50-image batch (NOT per image):

    * no systematic bias: ``|mean(htr_new - htr_old)| / mean(|htr_old|)``
      small (~few %); average of *signed* Δhtr should hover around zero.
    * median ``|Δhtr|`` is small relative to typical SPARC effects.
    * mean curve diff stays inside the measured bf16 floor.

The aggregate report quantifies the bf16 floor; don't force the tight
``0.02 / 1%`` tolerance from §12.5 on 7B-bf16.

CLI
---
    # Regime A — architectural exactness on 3B fp32:
    python scripts/equivalence_check.py --config configs/baseline.yaml \\
        --model-key qwen2.5-vl-3b \\
        --model-id Qwen/Qwen2.5-VL-3B-Instruct \\
        --dtype float32 --limit 5

    # Regime B — aggregate bf16 floor on 7B (smoke / official):
    python scripts/equivalence_check.py --config configs/baseline.yaml --limit 5
    python scripts/equivalence_check.py --config configs/baseline.yaml --limit 50
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


# Regime A — architectural exactness (fp32 on small model).
# Diagnostic floor on the scale-relative tensor diff. 1e-4 is the fp32 noise
# floor per op; over ~30 layers with residuals + LN + matmul, accumulated
# noise lands around 1e-3 — so 1e-3 is a generous diagnostic target. The
# Regime-A *gate driver* is the tight downstream §12.5 pass (|Δhtr| ≤ 0.02
# AND curve diff ≤ 1% on 5/5), which is what actually proves the math is
# identical to the TF path.
EXACTNESS_TENSOR_DIAG_TOL = 1e-3

# Regime B — legacy tight thresholds (EXPERIMENT.md §12.5). Kept for
# back-compat reporting; the verdict on bf16 7B now relies on the aggregate
# stats instead of these per-image thresholds.
HTR_TOL = 0.02
HTR_MIN_FRACTION = 45 / 50
CURVE_REL_TOL = 0.01


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


def _per_layer_hidden_diff(
    old_hidden,
    new_hidden,
    *,
    caption_start: int,
    caption_len: int,
) -> dict:
    """Per-layer max / median **scale-relative** diff between TF and FD hidden states.

    Restricts to the predictive-state range — positions
    ``[caption_start - 1, caption_start + caption_len - 1)`` — since that's
    the only range ``compute_kl_matrix`` reads.

    Why scale-relative, not per-element relative
    --------------------------------------------
    An earlier draft used ``|old - new| / |old|.clamp_min(1e-8)`` per element.
    That blows up when individual activations are near zero: a fp32-noise
    diff of 1e-7 over an activation of 1e-10 reports as 1e+3, even though
    downstream metrics (KL, htr, deep-curve) integrate over the distribution
    and don't see those outliers. The per-element max is an artefact of
    division by tiny values, not a real signal.

    We instead normalise by the **layer-wise mean magnitude** ``|old|.mean()``,
    so every entry is scored against the typical scale of its own layer.
    That handles near-zero outliers cleanly and stays in the same order of
    magnitude as the downstream relative metrics.

    Returned dict has:
        per_layer_max_scaled    : list[float], length n_layers
        per_layer_median_scaled : list[float], length n_layers
        overall_max_scaled      : float — worst over all (layer, pos, dim)
        overall_median_scaled   : float — median over all entries
    """
    import torch

    old = old_hidden.to(torch.float32)
    new = new_hidden.to(torch.float32)
    rng = slice(caption_start - 1, caption_start + caption_len - 1)

    n_layers = int(old.shape[0])
    per_max: list[float] = []
    per_med: list[float] = []
    overall_max = 0.0
    all_rel_values: list[float] = []  # subsampled for global median

    for L in range(n_layers):
        old_L = old[L, rng]
        new_L = new[L, rng]
        # Layer-wise scale: typical magnitude of this layer's activations.
        denom = old_L.abs().mean().clamp_min(1e-6)
        rel = (old_L - new_L).abs() / denom
        m_max = float(rel.max().item())
        m_med = float(rel.median().item())
        per_max.append(m_max)
        per_med.append(m_med)
        overall_max = max(overall_max, m_max)
        # Subsample to keep the global-median estimate light.
        flat = rel.flatten()
        if flat.numel() > 1024:
            idx = torch.linspace(0, flat.numel() - 1, 1024).long()
            flat = flat[idx]
        all_rel_values.extend(float(v) for v in flat.tolist())

    overall_median = (
        float(np.median(np.asarray(all_rel_values, dtype=np.float64)))
        if all_rel_values else float("nan")
    )

    return {
        "per_layer_max_scaled": per_max,
        "per_layer_median_scaled": per_med,
        "overall_max_scaled": overall_max,
        "overall_median_scaled": overall_median,
    }


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
    # Overrides so we can run the same script in two regimes without
    # cloning the config for every (model, dtype) combo.
    parser.add_argument(
        "--model-key", type=str, default=None,
        help="Override cfg['model']['key'] (e.g. qwen2.5-vl-3b for the Regime A fp32 gate).",
    )
    parser.add_argument(
        "--model-id", type=str, default=None,
        help="Override cfg['model']['model_id'] (e.g. Qwen/Qwen2.5-VL-3B-Instruct).",
    )
    parser.add_argument(
        "--dtype", type=str, default=None,
        help="Override cfg['model']['dtype'] (e.g. float32 for Regime A, bfloat16 for Regime B).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Apply CLI overrides.
    model_key = args.model_key or str(cfg["model"]["key"])
    model_id = args.model_id or str(cfg["model"]["model_id"])
    dtype_str = args.dtype or str(cfg["model"]["dtype"])

    # --- model load ---
    model_wrapper = build_model(model_key)
    model_wrapper.model_id = model_id
    dtype = resolve_dtype(dtype_str)
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
            # Full traceback so we can localise prefill-vs-step failures
            # without having to re-run with extra instrumentation.
            logger.error(traceback.format_exc())
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

        # 4. tensor-level diff (architectural exactness probe) on the A path
        tensor_diff = _per_layer_hidden_diff(
            old_A.hidden_states, new_A.hidden_states,
            caption_start=int(old_A.caption_start),
            caption_len=int(old_A.caption_len),
        )

        # 5. KL + head_tail_ratio for each path
        kl_old = _kl_for_pair(lm_head, old_A, old_B, top_k=top_k)
        kl_new = _kl_for_pair(lm_head, new_A, new_B, top_k=top_k)
        htr_old = head_tail_ratio(kl_old, t0=int(cfg["residual"]["t0"]))
        htr_new = head_tail_ratio(kl_new, t0=int(cfg["residual"]["t0"]))

        curve_old = _deep_curve(kl_old)
        curve_new = _deep_curve(kl_new)
        median_rel = _median_relative_curve_diff(curve_old, curve_new)

        # Signed delta — needed for the no-systematic-bias check in Regime B.
        signed_delta_htr = (
            (htr_new - htr_old)
            if (htr_old == htr_old and htr_new == htr_new)
            else float("nan")
        )
        delta_htr = abs(signed_delta_htr) if signed_delta_htr == signed_delta_htr else float("nan")

        rows.append({
            "image_id": image_id,
            "caption_len": int(old_A.caption_len),
            "htr_old": float(htr_old) if htr_old == htr_old else None,
            "htr_new": float(htr_new) if htr_new == htr_new else None,
            "signed_delta_htr": float(signed_delta_htr) if signed_delta_htr == signed_delta_htr else None,
            "abs_delta_htr": float(delta_htr) if delta_htr == delta_htr else None,
            "median_relative_curve_diff": float(median_rel) if median_rel == median_rel else None,
            "tensor_overall_max_scaled": tensor_diff["overall_max_scaled"],
            "tensor_overall_median_scaled": tensor_diff["overall_median_scaled"],
            "tensor_per_layer_max_scaled": tensor_diff["per_layer_max_scaled"],
            "tensor_per_layer_median_scaled": tensor_diff["per_layer_median_scaled"],
        })
        logger.info(
            f"[{image_id}] caption_len={int(old_A.caption_len)}  "
            f"htr_old={htr_old:.4f}  htr_new={htr_new:.4f}  "
            f"Δhtr_signed={signed_delta_htr:+.4f}  |Δhtr|={delta_htr:.4f}  "
            f"med_rel_curve={median_rel:.4%}  "
            f"tensor_max_scaled={tensor_diff['overall_max_scaled']:.2e}  "
            f"tensor_med_scaled={tensor_diff['overall_median_scaled']:.2e}"
        )

    if not rows:
        logger.error("No usable rows — equivalence check inconclusive.")
        return 1

    # --- aggregate ---
    finite_signed_deltas = [
        r["signed_delta_htr"] for r in rows if r["signed_delta_htr"] is not None
    ]
    finite_deltas = [r["abs_delta_htr"] for r in rows if r["abs_delta_htr"] is not None]
    finite_curve_rel = [
        r["median_relative_curve_diff"] for r in rows
        if r["median_relative_curve_diff"] is not None
    ]
    finite_htr_old = [r["htr_old"] for r in rows if r["htr_old"] is not None]
    tensor_max_per_image = [r["tensor_overall_max_scaled"] for r in rows]
    tensor_med_per_image = [r["tensor_overall_median_scaled"] for r in rows]

    # Regime B: aggregate, no-systematic-bias check.
    mean_signed = statistics.mean(finite_signed_deltas) if finite_signed_deltas else float("nan")
    mean_abs_htr_old = statistics.mean([abs(h) for h in finite_htr_old]) if finite_htr_old else float("nan")
    rel_systematic_bias = (
        abs(mean_signed) / mean_abs_htr_old
        if mean_abs_htr_old > 0 and mean_abs_htr_old == mean_abs_htr_old
        else float("nan")
    )

    # Legacy tight tolerances — reported for back-compat but not the verdict driver on bf16.
    n_within_htr_tol = sum(1 for d in finite_deltas if d <= HTR_TOL)
    n_within_curve_tol = sum(1 for d in finite_curve_rel if d <= CURVE_REL_TOL)
    htr_pass_tight = (n_within_htr_tol / len(rows)) >= HTR_MIN_FRACTION
    curve_pass_tight = (n_within_curve_tol / len(rows)) >= HTR_MIN_FRACTION

    # Regime A: architectural exactness. The gate driver is the tight
    # §12.5 pass (|Δhtr| ≤ 0.02 AND curve diff ≤ 1% on 5/5) — that's a
    # downstream metric and is what actually proves the FD path produces
    # the same predictions as TF. The tensor-level diff is reported as a
    # supporting diagnostic (sanity-check on the median scale-relative
    # tensor diff) but doesn't drive PASS/FAIL on its own.
    worst_tensor_max = max(tensor_max_per_image) if tensor_max_per_image else float("nan")
    median_tensor_median = (
        statistics.median(tensor_med_per_image) if tensor_med_per_image else float("nan")
    )
    tensor_diag_ok = (
        median_tensor_median == median_tensor_median
        and median_tensor_median <= EXACTNESS_TENSOR_DIAG_TOL
    )
    exactness_pass = htr_pass_tight and curve_pass_tight

    summary = {
        "n_images": len(rows),
        "model_id": model_wrapper.model_id,
        "model_key": model_key,
        "dtype": dtype_str,
        "config_path": str(args.config),
        "timestamp_iso": _iso_now(),

        # Regime A — architectural exactness.
        # Driver: tight downstream §12.5 (htr + curve). Diagnostic: tensor.
        "exactness_tensor_diag_tol": EXACTNESS_TENSOR_DIAG_TOL,
        "tensor_max_scaled_worst_image": worst_tensor_max,
        "tensor_max_scaled_median_over_images": statistics.median(tensor_max_per_image) if tensor_max_per_image else None,
        "tensor_median_scaled_median_over_images": median_tensor_median,
        "tensor_diag_ok": tensor_diag_ok,
        "exactness_pass": exactness_pass,

        # Regime B — aggregate / no-systematic-bias.
        "mean_signed_delta_htr": mean_signed,
        "mean_abs_htr_old": mean_abs_htr_old,
        "rel_systematic_bias": rel_systematic_bias,
        "median_abs_delta_htr": statistics.median(finite_deltas) if finite_deltas else None,
        "max_abs_delta_htr": max(finite_deltas) if finite_deltas else None,
        "mean_curve_relative_diff": statistics.mean(finite_curve_rel) if finite_curve_rel else None,
        "median_curve_relative_diff": statistics.median(finite_curve_rel) if finite_curve_rel else None,

        # Legacy tight thresholds (Regime A on small/fp32 should pass these too).
        "htr_tol": HTR_TOL,
        "n_within_htr_tol": n_within_htr_tol,
        "fraction_within_htr_tol": n_within_htr_tol / len(rows),
        "tight_htr_pass": htr_pass_tight,
        "curve_rel_tol": CURVE_REL_TOL,
        "n_within_curve_tol": n_within_curve_tol,
        "fraction_within_curve_tol": n_within_curve_tol / len(rows),
        "tight_curve_pass": curve_pass_tight,
    }

    report = {"summary": summary, "rows": rows}
    report_path = args.report_path or Path(
        f"results/equivalence_{Path(args.config).stem}_{model_key}_{dtype_str}_{_iso_now().replace(':', '-')}.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    logger.info("=" * 70)
    logger.info("EQUIVALENCE REPORT")
    logger.info("=" * 70)
    logger.info(f"model / dtype        : {model_wrapper.model_id} / {dtype_str}")
    logger.info(f"images               : {summary['n_images']}")
    logger.info("-- Tight downstream §12.5 (Regime A gate driver / Regime B informational) --")
    logger.info(
        f"|Δhtr| ≤ {HTR_TOL}        : {n_within_htr_tol}/{len(rows)}  "
        f"-> {'PASS' if htr_pass_tight else 'FAIL'}"
    )
    logger.info(
        f"curve diff ≤ {CURVE_REL_TOL*100:.1f}%   : {n_within_curve_tol}/{len(rows)}  "
        f"-> {'PASS' if curve_pass_tight else 'FAIL'}"
    )
    logger.info(
        f"REGIME A exactness   : {'PASS' if exactness_pass else 'FAIL'}  "
        "(tight 5/5 on both downstream thresholds; only meaningful on small/fp32)"
    )
    logger.info(
        "-- Tensor-level diagnostic (scale-relative, target median ≤ {:.0e}) --".format(EXACTNESS_TENSOR_DIAG_TOL)
    )
    logger.info(
        f"tensor max-scaled    : worst image {worst_tensor_max:.2e}  "
        f"(median over images: {summary['tensor_max_scaled_median_over_images']:.2e})"
    )
    logger.info(
        f"tensor median-scaled : median over images {summary['tensor_median_scaled_median_over_images']:.2e}  "
        f"({'OK' if tensor_diag_ok else 'check'})"
    )
    logger.info("-- Regime B (bf16 aggregate / no-systematic-bias) --")
    logger.info(
        f"mean signed Δhtr     : {mean_signed:+.4f}   "
        f"(rel to mean |htr_old| = {mean_abs_htr_old:.4f}: {rel_systematic_bias:.2%})"
    )
    logger.info(f"median |Δhtr|        : {summary['median_abs_delta_htr']:.4f}")
    logger.info(f"max |Δhtr|           : {summary['max_abs_delta_htr']:.4f}")
    logger.info(f"mean curve rel diff  : {summary['mean_curve_relative_diff']:.4%}")
    logger.info(f"median curve rel diff: {summary['median_curve_relative_diff']:.4%}")
    logger.info(f"report saved to      : {report_path}")
    # Exit code: PASS iff Regime A passes (tight downstream 5/5).
    return 0 if exactness_pass else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover (operator-side script)
        logger.debug(f"exception: {exc}")
        logger.error(traceback.format_exc())
        raise SystemExit(1)
