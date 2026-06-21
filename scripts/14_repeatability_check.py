#!/usr/bin/env python
"""Phase-1 sanity (item 3): FD repeatability vs SPARC effect, paired per image."""

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
    from vr_modality_bias.experiment.sparc import SparcHyperparams, enable_sparc
    from vr_modality_bias.metrics.kl import compute_kl_matrix
    from vr_modality_bias.metrics.residual import head_tail_ratio
    from vr_modality_bias.models.registry import build_model
    from vr_modality_bias.utils.config import load_config
    from vr_modality_bias.utils.device import resolve_dtype, select_device
    from vr_modality_bias.utils.seeds import derive_image_seed
except ModuleNotFoundError:
    sys.path.insert(0, str(here()))

    from src.vr_modality_bias.data.perturbations import noise_image_uniform
    from src.vr_modality_bias.data.prompts import get_prompt
    from src.vr_modality_bias.experiment.forced_decoding import collect_forced_decoding
    from src.vr_modality_bias.experiment.sparc import SparcHyperparams, enable_sparc
    from src.vr_modality_bias.metrics.kl import compute_kl_matrix
    from src.vr_modality_bias.metrics.residual import head_tail_ratio
    from src.vr_modality_bias.models.registry import build_model
    from src.vr_modality_bias.utils.config import load_config
    from src.vr_modality_bias.utils.device import resolve_dtype, select_device
    from src.vr_modality_bias.utils.seeds import derive_image_seed


def _iso_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def _htr_for_pair(lm_head, result_A, result_B, *, top_k: int, t0: int) -> float:
    kl = compute_kl_matrix(
        lm_head,
        result_A.hidden_states, result_B.hidden_states,
        caption_start=int(result_A.caption_start),
        caption_len=int(result_A.caption_len),
        top_k=top_k,
    )
    return float(head_tail_ratio(kl, t0=t0))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--limit", type=int, default=3,
        help="Number of images for the sanity. Default 3 — enough to see the ratio.",
    )
    parser.add_argument("--alpha", type=float, default=1.3, help="SPARC alpha.")
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--selected-layer", type=int, default=15)
    parser.add_argument("--se-layers", type=int, nargs=2, default=(0, 31))
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument(
        "--model-key", type=str, default=None,
        help="Override cfg['model']['key'] (e.g. qwen2.5-vl-7b for bf16 default).",
    )
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--dtype", type=str, default=None)
    parser.add_argument(
        "--report-path", type=Path, default=None,
        help="Where to write the JSON report.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
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

    # --- images + generation params ---
    images_dir = cfg["dataset"]["images_dir"]
    image_files = sorted(glob.glob(f"{images_dir}{os.sep}*.jpg"))[: args.limit]
    if not image_files:
        logger.error(f"No images under {images_dir}")
        return 1
    logger.info(f"Sanity-checking on {len(image_files)} image(s).")

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
    t0 = int(cfg["residual"]["t0"])

    hparams = SparcHyperparams(
        alpha=args.alpha,
        tau=args.tau,
        selected_layer=args.selected_layer,
        se_layers=tuple(args.se_layers),
        beta=args.beta,
    )

    rows: list[dict] = []
    for image_path in image_files:
        image_id = Path(image_path).stem
        with Image.open(image_path) as raw:
            image = raw.convert("RGB")
        noise_seed = derive_image_seed(seed_global, image_id)
        noise_img = noise_image_uniform(image, seed=int(noise_seed))

        caption_ref = model_wrapper.generate_caption(
            image=image, prompt=prompt,
            max_new_tokens=max_new_tokens,
            seed=noise_seed,
            generation_kwargs=gen_kwargs,
        )
        if not caption_ref.strip():
            logger.warning(f"[{image_id}] empty caption — skipping.")
            continue

        # 1. FD-OFF run #1 (A, B)
        try:
            off1_A = collect_forced_decoding(model_wrapper, image, prompt, caption_ref)
            off1_B = collect_forced_decoding(model_wrapper, noise_img, prompt, caption_ref)
            htr_off_1 = _htr_for_pair(lm_head, off1_A, off1_B, top_k=top_k, t0=t0)
            del off1_A, off1_B
        except Exception as exc:
            logger.error(f"[{image_id}] FD-OFF run 1 failed: {exc}")
            logger.error(traceback.format_exc())
            continue

        # 2. FD-OFF run #2 (A, B) — exact same path, exact same inputs.
        try:
            off2_A = collect_forced_decoding(model_wrapper, image, prompt, caption_ref)
            off2_B = collect_forced_decoding(model_wrapper, noise_img, prompt, caption_ref)
            htr_off_2 = _htr_for_pair(lm_head, off2_A, off2_B, top_k=top_k, t0=t0)
            del off2_A, off2_B
        except Exception as exc:
            logger.error(f"[{image_id}] FD-OFF run 2 failed: {exc}")
            logger.error(traceback.format_exc())
            continue

        # 3. FD-ON (SPARC) (A, B)
        try:
            with enable_sparc(
                model_wrapper,
                hparams=hparams,
                probe_image=image,
                prompt=prompt,
            ) as buffer:
                on_A = collect_forced_decoding(
                    model_wrapper, image, prompt, caption_ref, sparc_buffer=buffer,
                )
                on_B = collect_forced_decoding(
                    model_wrapper, noise_img, prompt, caption_ref, sparc_buffer=buffer,
                )
                htr_on = _htr_for_pair(lm_head, on_A, on_B, top_k=top_k, t0=t0)
            del on_A, on_B
        except Exception as exc:
            logger.error(f"[{image_id}] FD-ON failed: {exc}")
            logger.error(traceback.format_exc())
            continue

        delta_repeat = htr_off_2 - htr_off_1
        delta_sparc = htr_on - htr_off_1
        ratio = abs(delta_sparc) / max(abs(delta_repeat), 1e-6)

        rows.append({
            "image_id": image_id,
            "htr_off_1": htr_off_1,
            "htr_off_2": htr_off_2,
            "htr_on": htr_on,
            "delta_repeat": delta_repeat,
            "delta_sparc": delta_sparc,
            "ratio_sparc_over_repeat": ratio,
        })
        logger.info(
            f"[{image_id}] htr_off1={htr_off_1:.4f}  htr_off2={htr_off_2:.4f}  htr_on={htr_on:.4f}  "
            f"Δrepeat={delta_repeat:+.4f}  Δsparc={delta_sparc:+.4f}  ratio={ratio:.1f}×"
        )

    if not rows:
        logger.error("No usable rows.")
        return 1

    # --- aggregate ---
    abs_repeats = [abs(r["delta_repeat"]) for r in rows]
    abs_sparcs = [abs(r["delta_sparc"]) for r in rows]
    ratios = [r["ratio_sparc_over_repeat"] for r in rows]

    summary = {
        "n_images": len(rows),
        "model_id": model_wrapper.model_id,
        "dtype": dtype_str,
        "sparc_hparams": hparams.as_dict(),
        "median_abs_delta_repeat": statistics.median(abs_repeats),
        "max_abs_delta_repeat": max(abs_repeats),
        "median_abs_delta_sparc": statistics.median(abs_sparcs),
        "max_abs_delta_sparc": max(abs_sparcs),
        "median_ratio": statistics.median(ratios),
        "min_ratio": min(ratios),
        "timestamp_iso": _iso_now(),
    }

    report = {"summary": summary, "rows": rows}
    report_path = args.report_path or Path(
        f"results/repeatability_{Path(args.config).stem}_alpha{args.alpha}_{_iso_now().replace(':', '-')}.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    logger.info("=" * 70)
    logger.info("REPEATABILITY vs SPARC EFFECT")
    logger.info("=" * 70)
    logger.info(f"model / dtype             : {model_wrapper.model_id} / {dtype_str}")
    logger.info(f"SPARC alpha               : {args.alpha}")
    logger.info(f"images                    : {summary['n_images']}")
    logger.info(f"median |Δrepeat| (noise)  : {summary['median_abs_delta_repeat']:.4f}")
    logger.info(f"max    |Δrepeat|          : {summary['max_abs_delta_repeat']:.4f}")
    logger.info(f"median |Δsparc|  (signal) : {summary['median_abs_delta_sparc']:.4f}")
    logger.info(f"max    |Δsparc|           : {summary['max_abs_delta_sparc']:.4f}")
    logger.info(f"median ratio (signal/noise): {summary['median_ratio']:.1f}×")
    logger.info(f"min    ratio              : {summary['min_ratio']:.1f}×")
    logger.info(f"report saved to           : {report_path}")
    # Heuristic: SPARC is detectable if signal ≥ 5× noise floor on at least
    # half the images.
    return 0 if summary["median_ratio"] >= 5.0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover (operator-side script)
        logger.debug(f"exception: {exc}")
        logger.error(traceback.format_exc())
        raise SystemExit(1)
