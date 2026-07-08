#!/usr/bin/env python
"""Phase-2 orchestrator: forced-decoding A/B collection, SPARC OFF and SPARC ON.

Designed to be run **after** the §4.4 equivalence check
(``scripts/equivalence_check.py``) passes. Conceptually:

    for each image:
        caption_ref = model.generate_caption(image)                # deterministic per (seed, image_id)
        # condition 1: SPARC OFF
        result_A_off = collect_forced_decoding(image,        prompt, caption_ref)
        result_B_off = collect_forced_decoding(noise_image,  prompt, caption_ref)
        # condition 2: SPARC ON (in a context manager — restores originals on exit)
        with enable_sparc(...) as buffer:
            result_A_on  = collect_forced_decoding(image,       ..., sparc_buffer=buffer)
            result_B_on  = collect_forced_decoding(noise_image, ..., sparc_buffer=buffer)
        # KL + head_tail_ratio per condition; write rows to the parquets.

Outputs (under ``<run_dir>``):
    metrics_sparc_off.parquet   one row per image, baseline (SPARC OFF)
    metrics_sparc_on.parquet    one row per image, SPARC ON
    summary_compare.json        side-by-side aggregate statistics (htr per condition)
    ref_captions.jsonl          the captions used as forced targets
    logs/run_sparc_eval.log  per-image trace

CLI
---
    python scripts/run_sparc_eval.py --config configs/baseline.yaml --limit 5
    python scripts/run_sparc_eval.py --config configs/baseline.yaml --alpha 1.3 --limit 50
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger
from PIL import Image
from pyprojroot import here

try:
    from vr_modality_bias.data.perturbations import noise_image_uniform
    from vr_modality_bias.data.prompts import get_prompt
    from vr_modality_bias.experiment.forced_decoding import collect_forced_decoding
    from vr_modality_bias.experiment.sparc import SparcHyperparams, enable_sparc
    from vr_modality_bias.io.results import (
        compute_summary_stats,
        write_metrics_table,
        write_summary_json,
    )
    from vr_modality_bias.metrics.cosine import compute_cosine_distance_matrix
    from vr_modality_bias.metrics.kl import compute_kl_matrix
    from vr_modality_bias.metrics.residual import head_tail_ratio, residual_drift_ratio
    from vr_modality_bias.models.registry import build_model
    from vr_modality_bias.utils.config import load_config, snapshot_config
    from vr_modality_bias.utils.device import resolve_dtype, select_device
    from vr_modality_bias.utils.runs import make_run_dir
    from vr_modality_bias.utils.seeds import derive_image_seed
except ModuleNotFoundError:
    sys.path.insert(0, str(here()))

    from src.vr_modality_bias.data.perturbations import noise_image_uniform
    from src.vr_modality_bias.data.prompts import get_prompt
    from src.vr_modality_bias.experiment.forced_decoding import collect_forced_decoding
    from src.vr_modality_bias.experiment.sparc import SparcHyperparams, enable_sparc
    from src.vr_modality_bias.io.results import (
        compute_summary_stats,
        write_metrics_table,
        write_summary_json,
    )
    from src.vr_modality_bias.metrics.cosine import compute_cosine_distance_matrix
    from src.vr_modality_bias.metrics.kl import compute_kl_matrix
    from src.vr_modality_bias.metrics.residual import head_tail_ratio, residual_drift_ratio
    from src.vr_modality_bias.models.registry import build_model
    from src.vr_modality_bias.utils.config import load_config, snapshot_config
    from src.vr_modality_bias.utils.device import resolve_dtype, select_device
    from src.vr_modality_bias.utils.runs import make_run_dir
    from src.vr_modality_bias.utils.seeds import derive_image_seed


def _iso_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def _row_for_pair(
    *,
    lm_head,
    result_A,
    result_B,
    image_id: str,
    caption_ref: str,
    model_id: str,
    prompt_key: str,
    seed_global: int,
    noise_seed: int,
    top_k: int,
    t0: int,
    condition: str,
) -> dict:
    """Compute metrics for a single (A, B) pair and produce one parquet row."""
    kl = compute_kl_matrix(
        lm_head,
        result_A.hidden_states, result_B.hidden_states,
        caption_start=int(result_A.caption_start),
        caption_len=int(result_A.caption_len),
        top_k=top_k,
    )
    cos = compute_cosine_distance_matrix(
        result_A.hidden_states, result_B.hidden_states,
        caption_start=int(result_A.caption_start),
        caption_len=int(result_A.caption_len),
    )
    rr = residual_drift_ratio(kl, t0=t0)
    htr = head_tail_ratio(kl, t0=t0)

    return {
        "image_id": image_id,
        "caption_len": int(result_A.caption_len),
        "n_layers": int(result_A.hidden_states.shape[0]),
        "hidden_dim": int(result_A.hidden_states.shape[-1]),
        "caption_ref": caption_ref,
        "kl": kl,
        "cos_dist": cos,
        "residual_ratio": float(rr),
        "head_tail_ratio": float(htr),
        "model_id": model_id,
        "prompt_key": prompt_key,
        "seed_global": int(seed_global),
        "noise_seed": int(noise_seed),
        "timestamp_iso": _iso_now(),
        "caption_tokens": None,
        # condition is metadata; not part of METRICS_SCHEMA so it's just for logging.
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--alpha", type=float, default=1.3,
        help="SPARC alpha. Defaults to 1.3 (middle of the 1.1-1.5 band). "
             "Run a mini-sweep in Phase 2 if the effect is weak.",
    )
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--selected-layer", type=int, default=15)
    parser.add_argument("--se-layers", type=int, nargs=2, default=(0, 31))
    parser.add_argument("--beta", type=float, default=0.0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_dir = make_run_dir(cfg["run"]["output_root"], cfg["run"]["name"])
    log_file = run_dir / "logs" / "run_sparc_eval.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_file))
    logger.info(f"Run dir: {run_dir}")
    snapshot_config(args.config, run_dir)

    # ---- SPARC config snapshot (immutable record of what we ran) ----
    hparams = SparcHyperparams(
        alpha=args.alpha,
        tau=args.tau,
        selected_layer=args.selected_layer,
        se_layers=tuple(args.se_layers),
        beta=args.beta,
    )
    (run_dir / "sparc_hparams.json").write_text(
        json.dumps(hparams.as_dict(), indent=2) + "\n", encoding="utf-8"
    )
    logger.info(f"SPARC hparams: {hparams.as_dict()}")

    # ---- model load ----
    model = build_model(cfg["model"]["key"])
    model.model_id = str(cfg["model"]["model_id"])
    dtype = resolve_dtype(str(cfg["model"]["dtype"]))
    device = select_device("cuda")
    if hasattr(model, "_dtype"):
        model._dtype = dtype  # noqa: SLF001
    logger.info(f"Loading model {model.model_id} on {device} (dtype={dtype})...")
    model.load(device)
    lm_head = model.get_lm_head()
    logger.info(f"Loaded. n_layers={model.n_layers}")

    # ---- dataset ----
    images_dir = cfg["dataset"]["images_dir"]
    image_files = sorted(glob.glob(f"{images_dir}{os.sep}*.jpg"))
    if args.limit:
        image_files = image_files[: args.limit]
    if not image_files:
        logger.error(f"No images under {images_dir}")
        return 1
    logger.info(f"Processing {len(image_files)} image(s).")

    # ---- generation params ----
    prompt_key = str(cfg["task"]["prompt_key"])
    prompt = get_prompt(prompt_key)
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

    # ---- iterate ----
    ref_path = run_dir / "ref_captions.jsonl"
    rows_off: list[dict] = []
    rows_on: list[dict] = []

    # Probe image — used by enable_sparc to discover image_token_index.
    with Image.open(image_files[0]) as probe_raw:
        probe_image = probe_raw.convert("RGB")

    for image_path in image_files:
        image_id = Path(image_path).stem
        with Image.open(image_path) as raw:
            image = raw.convert("RGB")
        noise_seed = derive_image_seed(seed_global, image_id)
        noise_img = noise_image_uniform(image, seed=int(noise_seed))

        # ---- caption_ref ----
        caption_ref = model.generate_caption(
            image=image, prompt=prompt,
            max_new_tokens=max_new_tokens,
            seed=int(noise_seed),
            generation_kwargs=gen_kwargs,
        )
        if not caption_ref.strip():
            logger.warning(f"[{image_id}] empty caption — skipping image.")
            continue
        with ref_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "image_id": image_id,
                "caption_ref": caption_ref,
                "model_id": model.model_id,
                "prompt_key": prompt_key,
                "noise_seed": int(noise_seed),
                "seed_global": int(seed_global),
                "timestamp": _iso_now(),
            }) + "\n")

        # ---- SPARC OFF ----
        try:
            off_A = collect_forced_decoding(model, image, prompt, caption_ref)
            off_B = collect_forced_decoding(model, noise_img, prompt, caption_ref)
        except Exception as exc:
            logger.error(f"[{image_id}] OFF collection failed: {exc}")
            continue
        row_off = _row_for_pair(
            lm_head=lm_head, result_A=off_A, result_B=off_B,
            image_id=image_id, caption_ref=caption_ref,
            model_id=model.model_id, prompt_key=prompt_key,
            seed_global=seed_global, noise_seed=int(noise_seed),
            top_k=top_k, t0=t0, condition="off",
        )
        rows_off.append(row_off)
        logger.info(
            f"[{image_id}] OFF htr={row_off['head_tail_ratio']:.4f} "
            f"rr={row_off['residual_ratio']:.4f} caption_len={row_off['caption_len']}"
        )

        # ---- SPARC ON ----
        try:
            with enable_sparc(model, hparams=hparams, probe_image=probe_image, prompt=prompt) as buffer:
                on_A = collect_forced_decoding(model, image, prompt, caption_ref, sparc_buffer=buffer)
                on_B = collect_forced_decoding(model, noise_img, prompt, caption_ref, sparc_buffer=buffer)
        except Exception as exc:
            logger.error(f"[{image_id}] ON collection failed: {exc}")
            continue
        row_on = _row_for_pair(
            lm_head=lm_head, result_A=on_A, result_B=on_B,
            image_id=image_id, caption_ref=caption_ref,
            model_id=model.model_id, prompt_key=prompt_key,
            seed_global=seed_global, noise_seed=int(noise_seed),
            top_k=top_k, t0=t0, condition="on",
        )
        rows_on.append(row_on)
        logger.info(
            f"[{image_id}] ON  htr={row_on['head_tail_ratio']:.4f} "
            f"rr={row_on['residual_ratio']:.4f}  "
            f"Δhtr={row_on['head_tail_ratio'] - row_off['head_tail_ratio']:+.4f}"
        )

        # Flush after every image — crash-safe.
        write_metrics_table(rows_off, run_dir / "metrics_sparc_off.parquet")
        write_metrics_table(rows_on, run_dir / "metrics_sparc_on.parquet")

    # ---- side-by-side summary ----
    stats_off = compute_summary_stats(rows_off)
    stats_on = compute_summary_stats(rows_on)
    htr_old_med = stats_off["head_tail_ratio"].get("median")
    htr_new_med = stats_on["head_tail_ratio"].get("median")
    summary = {
        "n_images": len(rows_off),
        "sparc_hparams": hparams.as_dict(),
        "sparc_off": stats_off,
        "sparc_on": stats_on,
        "delta_htr_median": (
            float(htr_new_med - htr_old_med)
            if htr_old_med is not None and htr_new_med is not None
            else None
        ),
        "timestamp_iso": _iso_now(),
        "config_path": str(args.config),
    }
    write_summary_json(summary, run_dir / "summary_compare.json")
    logger.info("=" * 70)
    logger.info(
        f"DONE. n={len(rows_off)}  "
        f"OFF htr={htr_old_med}  ON htr={htr_new_med}  "
        f"Δhtr median={summary['delta_htr_median']}"
    )
    logger.info(f"results under {run_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover (operator-side script)
        logger.debug(f"exception: {exc}")
        logger.error(traceback.format_exc())
        raise SystemExit(1)
