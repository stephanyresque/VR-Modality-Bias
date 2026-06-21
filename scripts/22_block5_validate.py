#!/usr/bin/env python
"""Block-5 functional validation — LLaVA-1.5-7B end-to-end, small sample.

Pipeline validated, in a single process:

    1. Audit: confirm the SPARC ``image_positions`` mask is 100%
       <image>-tokens for LLaVA-1.5 (contiguous block, no Idefics3-style
       separators).
    2. For each of ``--limit`` images (default 3), length=long:
       (a) Generate ``caption_baseline`` (free generation, greedy).
       (b) Generate ``caption_sparc`` (free generation, greedy, with
           ``enable_sparc`` context — official COCO config).
       (c) Run ``collect_forced_decoding`` PAIRED:
             * SPARC OFF on (real_image, noise_image)
             * SPARC ON  on (real_image, noise_image)
           This is the apples-to-apples ``share_tail`` comparison (both
           sides use the FD code path so the TF↔FD bf16 noise floor
           doesn't pollute the delta).
       (d) Compute KL, deep_curve and share_tail for each condition.
    3. Write rows:
       * ``results/diagnostico/llava-1.5-7b/long/<run>_<ts>/metrics.parquet``
         — baseline (FD-OFF) rows, post-Block-4 layout.
       * ``results/avaliacao/llava-1.5-7b/long/<run>_<ts>/metrics.parquet``
         — SPARC (FD-ON) rows, same schema.
    4. Print everything to stdout: audit verdict, side-by-side captions,
       per-image share_tail table, and bounded-check sanity (both
       conditions must produce values in [0, 1]).

This is a FUNCTIONAL validation, not a scientific run. Two-to-three
images is enough to prove every code path lights up and writes to the
new layout. Scale to 50 images for the real diagnostic.

CLI
---
    python scripts/22_block5_validate.py
    python scripts/22_block5_validate.py --limit 2 --length long
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from pyprojroot import here

try:
    from vr_modality_bias.data.perturbations import noise_image_uniform
    from vr_modality_bias.data.prompts import get_prompt
    from vr_modality_bias.experiment.forced_decoding import collect_forced_decoding
    from vr_modality_bias.experiment.sparc import SparcHyperparams, enable_sparc
    from vr_modality_bias.io.results import write_metrics_table
    from vr_modality_bias.metrics.cosine import compute_cosine_distance_matrix
    from vr_modality_bias.metrics.kl import compute_kl_matrix
    from vr_modality_bias.metrics.residual import (
        deep_block,
        residual_drift_ratio,
        share_tail,
    )
    from vr_modality_bias.models.registry import build_model
    from vr_modality_bias.utils.config import load_config
    from vr_modality_bias.utils.device import resolve_dtype, select_device
    from vr_modality_bias.utils.runs import (
        area_root,
        length_from_prompt_key,
        make_run_dir,
    )
    from vr_modality_bias.utils.seeds import derive_image_seed
except ModuleNotFoundError:
    sys.path.insert(0, str(here()))

    from src.vr_modality_bias.data.perturbations import noise_image_uniform
    from src.vr_modality_bias.data.prompts import get_prompt
    from src.vr_modality_bias.experiment.forced_decoding import collect_forced_decoding
    from src.vr_modality_bias.experiment.sparc import SparcHyperparams, enable_sparc
    from src.vr_modality_bias.io.results import write_metrics_table
    from src.vr_modality_bias.metrics.cosine import compute_cosine_distance_matrix
    from src.vr_modality_bias.metrics.kl import compute_kl_matrix
    from src.vr_modality_bias.metrics.residual import (
        deep_block,
        residual_drift_ratio,
        share_tail,
    )
    from src.vr_modality_bias.models.registry import build_model
    from src.vr_modality_bias.utils.config import load_config
    from src.vr_modality_bias.utils.device import resolve_dtype, select_device
    from src.vr_modality_bias.utils.runs import (
        area_root,
        length_from_prompt_key,
        make_run_dir,
    )
    from src.vr_modality_bias.utils.seeds import derive_image_seed


_LENGTH_CONFIGS = {
    "short":  "configs/run_llava_short.yaml",
    "medium": "configs/run_llava_medium.yaml",
    "long":   "configs/run_llava_long.yaml",
}


def _iso_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


# ----------------------------------------------------------------- audit


def _audit_llava_indexing(model_wrapper, image, prompt: str) -> dict:
    """Reproduce scripts/19_audit_image_token_layout for one (model, image).

    The full diagnostic script is fine for stand-alone use; here we
    inline the essential check so the block-5 output has it without a
    second subprocess.
    """
    proc = model_wrapper._processor  # noqa: SLF001
    msgs = model_wrapper._build_messages(prompt, image)
    # LLaVA's chat template renders this to "USER: <image>\n{prompt} ASSISTANT:"
    chat_template = getattr(proc, "chat_template", None)
    if chat_template:
        prefix_text = proc.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False,
        )
    else:
        prefix_text = f"USER: <image>\n{prompt} ASSISTANT:"
    prefix_inputs = proc(text=[prefix_text], images=[image], return_tensors="pt")
    input_ids = prefix_inputs["input_ids"][0]
    image_token_id = int(model_wrapper._model.config.image_token_id)  # noqa: SLF001
    positions = (input_ids == image_token_id).nonzero(as_tuple=True)[0]
    n_patches = int(positions.numel())
    if n_patches == 0:
        return {
            "image_token_id": image_token_id,
            "image_token_index": None,
            "num_image_patches": 0,
            "input_len": int(input_ids.shape[-1]),
            "n_image_in_range": 0,
            "contiguous": False,
        }
    image_token_index = int(positions[0])
    contiguous = bool(int(positions[-1]) - image_token_index + 1 == n_patches)
    # Legacy contiguous-range audit (what the SPARC code WOULD have
    # done before the per-id mask fix in Block-4 wave). For LLaVA's
    # contiguous layout this MUST give 100%.
    n_image_in_range = int(
        (input_ids[image_token_index:image_token_index + n_patches] == image_token_id)
        .sum()
        .item()
    )
    return {
        "image_token_id": image_token_id,
        "image_token_index": image_token_index,
        "num_image_patches": n_patches,
        "input_len": int(input_ids.shape[-1]) - n_patches,
        "n_image_in_range": n_image_in_range,
        "contiguous": contiguous,
        "total_seq_len": int(input_ids.shape[-1]),
    }


# --------------------------------------------------------- write helpers


def _build_row(
    *,
    lm_head,
    result_A,
    result_B,
    image_id: str,
    caption_ref: str,
    condition_meta: dict,
    top_k: int,
    t0: int,
    model_id: str,
    prompt_key: str,
    seed_global: int,
    noise_seed: int,
) -> dict:
    """Compute the full metrics row (matches METRICS_SCHEMA post-Block-4)."""
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
    rr = float(residual_drift_ratio(kl, t0=t0))
    st = float(share_tail(kl))
    l0, l1 = deep_block(int(kl.shape[0]))
    deep_curve_arr = kl[l0:l1, :].astype(np.float32).mean(axis=0)

    return {
        "image_id": image_id,
        "caption_len": int(result_A.caption_len),
        "n_layers": int(result_A.hidden_states.shape[0]),
        "hidden_dim": int(result_A.hidden_states.shape[-1]),
        "caption_ref": caption_ref,
        "kl": kl,
        "cos_dist": cos,
        "deep_curve": deep_curve_arr,
        "residual_ratio": rr,
        "share_tail": st,
        "head_tail_ratio": None,  # deprecated; intentionally not computed
        "model_id": model_id,
        "prompt_key": prompt_key,
        "seed_global": int(seed_global),
        "noise_seed": int(noise_seed),
        "timestamp_iso": _iso_now(),
        "caption_tokens": None,
        # Carries the SPARC settings + free-generation caption for the
        # JSON dump under each run dir.
        **condition_meta,
    }


# ----------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--length", choices=("short", "medium", "long"), default="long",
        help="Which LLaVA config to use. Default 'long' (max_new_tokens=512)."
    )
    parser.add_argument(
        "--limit", type=int, default=3,
        help="Number of images for the validation. Default 3 — the spec is 2-3."
    )
    args = parser.parse_args()

    cfg_path = Path(_LENGTH_CONFIGS[args.length])
    cfg = load_config(cfg_path)

    model_key = str(cfg["model"]["key"])
    model_id = str(cfg["model"]["model_id"])
    dtype_str = str(cfg["model"]["dtype"])
    prompt_key = str(cfg["task"]["prompt_key"])
    prompt = get_prompt(prompt_key)
    seed_global = int(cfg["run"]["seed_global"])
    max_new_tokens = int(cfg["generation"]["max_new_tokens"])
    top_k = int(cfg["metrics"]["logits_top_k"])
    t0 = int(cfg["residual"]["t0"])
    sparc_cfg = cfg.get("sparc", {})

    # Greedy across the board (free generation, baseline AND SPARC).
    gen_kwargs = {
        "do_sample": False,
        "num_beams": 1,
        "repetition_penalty": float(cfg["generation"].get("repetition_penalty", 1.0)),
    }

    print("=" * 78)
    print(f"BLOCK 5 VALIDATION — {model_id} ({dtype_str})")
    print(f"length={args.length}  limit={args.limit}  prompt_key={prompt_key}")
    print(f"decoding: GREEDY  max_new_tokens={max_new_tokens}  rep_pen={gen_kwargs['repetition_penalty']}")
    print(f"SPARC (official COCO): {dict(sparc_cfg)}")
    print("=" * 78)

    # --- model load ----
    device = select_device("cuda")
    dtype = resolve_dtype(dtype_str)
    model_wrapper = build_model(model_key)
    model_wrapper.model_id = model_id
    if hasattr(model_wrapper, "_dtype"):
        model_wrapper._dtype = dtype  # noqa: SLF001
    print(f"Loading {model_id} on {device}...")
    model_wrapper.load(device)
    print(f"Loaded. n_layers={model_wrapper.n_layers}  "
          f"lm_head={type(model_wrapper.get_lm_head()).__name__}")
    lm_head = model_wrapper.get_lm_head()

    sparc_hparams = SparcHyperparams(
        alpha=float(sparc_cfg.get("alpha", 1.1)),
        beta=float(sparc_cfg.get("beta", 0.1)),
        tau=float(sparc_cfg.get("tau", 1.5)),
        selected_layer=int(sparc_cfg.get("selected_layer", 20)),
        se_layers=tuple(sparc_cfg.get("se_layers", (0, 31))),
    )

    # --- images ----
    images_dir = Path(cfg["dataset"]["images_dir"])
    image_files = sorted(glob.glob(f"{images_dir}{os.sep}*.jpg"))[: args.limit]
    if not image_files:
        print(f"ERROR: no images under {images_dir}", file=sys.stderr)
        return 1
    print(f"\nWill validate on {len(image_files)} image(s).")

    # --- audit indexing on the FIRST image ----
    audit_image = Image.open(image_files[0]).convert("RGB")
    audit = _audit_llava_indexing(model_wrapper, audit_image, prompt)
    print()
    print("─" * 78)
    print("AUDIT — LLaVA image-token layout (1st validation image)")
    print("─" * 78)
    print(f"  config.image_token_id : {audit['image_token_id']}")
    print(f"  image_token_index     : {audit['image_token_index']}")
    print(f"  num_image_patches     : {audit['num_image_patches']}")
    print(f"  input_len (no patches): {audit['input_len']}")
    print(f"  total seq length      : {audit['total_seq_len']}")
    print(f"  contiguous block?     : {audit['contiguous']}")
    print(
        f"  positions [{audit['image_token_index']}, "
        f"{audit['image_token_index'] + audit['num_image_patches']}) "
        f"that are <image>: {audit['n_image_in_range']}/{audit['num_image_patches']}"
    )
    audit_ok = (
        audit["num_image_patches"] > 0
        and audit["contiguous"]
        and audit["n_image_in_range"] == audit["num_image_patches"]
    )
    print(f"  VERDICT: {'PASS — 100% <image> tokens' if audit_ok else 'FAIL — investigate'}")

    # --- prepare output dirs (BOTH areas) ----
    length = length_from_prompt_key(prompt_key)
    diag_root = area_root(
        cfg["run"]["output_root"], area="diagnostico",
        model_key=model_key, length=length,
    )
    eval_root = area_root(
        cfg["run"]["output_root"], area="avaliacao",
        model_key=model_key, length=length,
    )
    diag_run_dir = make_run_dir(diag_root, "block5_baseline")
    eval_run_dir = make_run_dir(eval_root, "block5_sparc")
    print()
    print(f"diagnostico run dir  : {diag_run_dir}")
    print(f"avaliacao   run dir  : {eval_run_dir}")

    # --- per-image loop ----
    baseline_rows: list[dict] = []
    sparc_rows: list[dict] = []
    per_image_summary: list[dict] = []

    for image_path in image_files:
        image_id = Path(image_path).stem
        image = Image.open(image_path).convert("RGB")
        noise_seed = derive_image_seed(seed_global, image_id)
        noise_img = noise_image_uniform(image, seed=int(noise_seed))

        print()
        print("=" * 78)
        print(f"IMAGE: {image_id}")
        print("=" * 78)

        # (a) free baseline caption (greedy, NO SPARC)
        try:
            t0_b = time.time()
            caption_baseline = model_wrapper.generate_caption(
                image=image, prompt=prompt,
                max_new_tokens=max_new_tokens,
                seed=int(noise_seed),
                generation_kwargs=gen_kwargs,
            )
            t_baseline = time.time() - t0_b
            print(f"\n[BASELINE caption] ({t_baseline:.1f}s, {len(caption_baseline.split())} words)")
            print(f"  {caption_baseline}")
        except Exception as exc:
            print(f"ERROR baseline generate: {exc}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            return 1

        # (b) free SPARC caption (greedy, WITH SPARC)
        try:
            t0_s = time.time()
            with enable_sparc(
                model_wrapper, hparams=sparc_hparams,
                probe_image=image, prompt=prompt,
            ) as buffer:
                # Per-image SPARC bookkeeping (same as forced_decoding).
                proc = model_wrapper._processor  # noqa: SLF001
                msgs = model_wrapper._build_messages(prompt, image)
                chat_template = getattr(proc, "chat_template", None)
                if chat_template:
                    prefix_text = proc.apply_chat_template(
                        msgs, add_generation_prompt=True, tokenize=False,
                    )
                else:
                    prefix_text = f"USER: <image>\n{prompt} ASSISTANT:"
                prefix_inputs = proc(text=[prefix_text], images=[image], return_tensors="pt")
                image_token_id = int(model_wrapper._model.config.image_token_id)  # noqa: SLF001
                image_positions = (
                    prefix_inputs["input_ids"][0] == image_token_id
                ).nonzero(as_tuple=True)[0]
                num_image_patches = int(image_positions.numel())
                input_len = int(prefix_inputs["input_ids"].shape[-1]) - num_image_patches
                buffer.reset()
                buffer.update_input_len(input_len)
                buffer.update_image_positions(image_positions)
                caption_sparc = model_wrapper.generate_caption(
                    image=image, prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    seed=int(noise_seed),
                    generation_kwargs=gen_kwargs,
                )
            t_sparc = time.time() - t0_s
            print(f"\n[SPARC    caption] ({t_sparc:.1f}s, {len(caption_sparc.split())} words)")
            print(f"  {caption_sparc}")
        except Exception as exc:
            print(f"ERROR SPARC generate: {exc}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            return 1

        # (c) paired FD measurements — both go through collect_forced_decoding
        #     so the comparison isn't biased by the TF↔FD bf16 floor.
        try:
            # Baseline: SPARC OFF (no buffer).
            off_A = collect_forced_decoding(model_wrapper, image, prompt, caption_baseline)
            off_B = collect_forced_decoding(model_wrapper, noise_img, prompt, caption_baseline)
            row_off = _build_row(
                lm_head=lm_head, result_A=off_A, result_B=off_B,
                image_id=image_id, caption_ref=caption_baseline,
                condition_meta={"_condition": "off", "_free_caption": caption_baseline},
                top_k=top_k, t0=t0,
                model_id=model_id, prompt_key=prompt_key,
                seed_global=seed_global, noise_seed=int(noise_seed),
            )
            baseline_rows.append(row_off)
            del off_A, off_B
        except Exception as exc:
            print(f"ERROR baseline FD: {exc}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            return 1

        try:
            # SPARC: enable + FD with sparc_buffer + per-id mask.
            with enable_sparc(
                model_wrapper, hparams=sparc_hparams,
                probe_image=image, prompt=prompt,
            ) as buffer:
                on_A = collect_forced_decoding(
                    model_wrapper, image, prompt, caption_baseline,
                    sparc_buffer=buffer,
                )
                on_B = collect_forced_decoding(
                    model_wrapper, noise_img, prompt, caption_baseline,
                    sparc_buffer=buffer,
                )
                row_on = _build_row(
                    lm_head=lm_head, result_A=on_A, result_B=on_B,
                    image_id=image_id, caption_ref=caption_baseline,
                    condition_meta={
                        "_condition": "on",
                        "_free_caption": caption_sparc,
                        **sparc_hparams.as_dict(),
                    },
                    top_k=top_k, t0=t0,
                    model_id=model_id, prompt_key=prompt_key,
                    seed_global=seed_global, noise_seed=int(noise_seed),
                )
                sparc_rows.append(row_on)
                del on_A, on_B
        except Exception as exc:
            print(f"ERROR SPARC FD: {exc}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            return 1

        st_off = row_off["share_tail"]
        st_on = row_on["share_tail"]
        per_image_summary.append({
            "image_id": image_id,
            "caption_baseline": caption_baseline,
            "caption_sparc": caption_sparc,
            "share_tail_off": st_off,
            "share_tail_on": st_on,
            "delta": st_on - st_off,
        })
        print(f"\n[share_tail] OFF={st_off:.4f}   ON={st_on:.4f}   Δ={st_on - st_off:+.4f}")

    # --- persist parquets ----
    # Drop the underscore-prefixed metadata fields before write — they
    # are not in METRICS_SCHEMA (which is shared across runs).
    def _clean_for_schema(rows):
        out = []
        for r in rows:
            out.append({k: v for k, v in r.items() if not k.startswith("_")
                        and k not in {"alpha", "beta", "tau", "selected_layer", "se_layers"}})
        return out

    write_metrics_table(_clean_for_schema(baseline_rows), diag_run_dir / "metrics.parquet")
    write_metrics_table(_clean_for_schema(sparc_rows),    eval_run_dir / "metrics.parquet")

    # --- final printout ----
    print()
    print("=" * 78)
    print("SUMMARY — Block 5 validation")
    print("=" * 78)
    print(f"audit               : {'PASS' if audit_ok else 'FAIL'}")
    print(f"diagnostico written : {diag_run_dir / 'metrics.parquet'}  "
          f"({len(baseline_rows)} rows)")
    print(f"avaliacao   written : {eval_run_dir / 'metrics.parquet'}  "
          f"({len(sparc_rows)} rows)")
    print()
    print("Per-image share_tail (paired FD-OFF vs FD-ON, both bounded [0, 1]):")
    print(f"  {'image_id':<14} {'OFF':>8} {'ON':>8} {'Δ':>9}  bounded?")
    for s in per_image_summary:
        bounded = (0.0 <= s["share_tail_off"] <= 1.0) and (0.0 <= s["share_tail_on"] <= 1.0)
        print(f"  {s['image_id']:<14} {s['share_tail_off']:>8.4f} "
              f"{s['share_tail_on']:>8.4f} {s['delta']:>+9.4f}  "
              f"{'yes' if bounded else 'NO'}")

    all_bounded = all(
        0.0 <= s["share_tail_off"] <= 1.0 and 0.0 <= s["share_tail_on"] <= 1.0
        for s in per_image_summary
    )
    print()
    print(f"all share_tail bounded in [0, 1]: {'yes' if all_bounded else 'NO'}")
    print(f"VERDICT: {'BLOCK 5 OK' if (audit_ok and all_bounded) else 'BLOCK 5 FAIL'}")
    return 0 if (audit_ok and all_bounded) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover (operator-side)
        print(f"top-level failure: {exc}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        raise SystemExit(1)
