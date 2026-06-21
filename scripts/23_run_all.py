#!/usr/bin/env python
"""Single resumable orchestrator — full LLaVA-1.5-7B diagnostic + SPARC run."""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from loguru import logger
from PIL import Image
from pyprojroot import here

try:
    from vr_modality_bias.data.captions import detect_loop
    from vr_modality_bias.data.perturbations import noise_image_uniform
    from vr_modality_bias.data.prompts import get_prompt
    from vr_modality_bias.experiment.forced_decoding import collect_forced_decoding
    from vr_modality_bias.experiment.sparc import SparcHyperparams, enable_sparc
    from vr_modality_bias.io.results import METRICS_SCHEMA, write_metrics_table
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
    from vr_modality_bias.utils.runs import area_root, length_from_prompt_key
    from vr_modality_bias.utils.seeds import derive_image_seed
except ModuleNotFoundError:
    sys.path.insert(0, str(here()))

    from src.vr_modality_bias.data.captions import detect_loop
    from src.vr_modality_bias.data.perturbations import noise_image_uniform
    from src.vr_modality_bias.data.prompts import get_prompt
    from src.vr_modality_bias.experiment.forced_decoding import collect_forced_decoding
    from src.vr_modality_bias.experiment.sparc import SparcHyperparams, enable_sparc
    from src.vr_modality_bias.io.results import METRICS_SCHEMA, write_metrics_table
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
    from src.vr_modality_bias.utils.runs import area_root, length_from_prompt_key
    from src.vr_modality_bias.utils.seeds import derive_image_seed


_LENGTH_CONFIGS = {
    "short":  "configs/run_llava_short.yaml",
    "medium": "configs/run_llava_medium.yaml",
    "long":   "configs/run_llava_long.yaml",
}


def _iso_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------- atomic IO


def _atomic_write_parquet(rows: list[dict], path: Path) -> None:
    """Atomic parquet write — tmp + rename. Survives kills mid-write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    write_metrics_table(rows, tmp)
    os.replace(tmp, path)


def _load_existing_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return pq.read_table(path).to_pylist()


def _cell_key(row: dict) -> tuple:
    """Stable identity of a cell — (image_id, condition)."""
    return (row["image_id"], row.get("condition"))


def _upsert(rows: list[dict], new_row: dict) -> list[dict]:
    """Replace the row with the same cell key, or append if new."""
    key = _cell_key(new_row)
    return [r for r in rows if _cell_key(r) != key] + [new_row]


# ------------------------------------------------------- caption cache


def _read_caption_cache(jsonl_path: Path) -> dict[str, str]:
    cache: dict[str, str] = {}
    if not jsonl_path.exists():
        return cache
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            iid = entry.get("image_id")
            cap = entry.get("caption_baseline")
            if iid and cap:
                cache[str(iid)] = str(cap)
    return cache


def _append_caption(jsonl_path: Path, entry: dict) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


# --------------------------------------------------------- per-image helpers


def _probe_sparc_layout(model_wrapper, image, prompt: str):
    """Return ``(input_len, image_positions)`` for an image — mirrors
    scripts/22's helper. Used to set up the per-id mask on the SPARC
    buffer for both free generation and forced decoding."""
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
    n_patches = int(image_positions.numel())
    input_len = int(prefix_inputs["input_ids"].shape[-1]) - n_patches
    return input_len, image_positions


def _build_row(
    *,
    lm_head,
    result_A,
    result_B,
    image_id: str,
    length: str,
    condition: str,
    caption_ref: str,
    free_caption: str,
    degenerated: bool,
    degeneration_reason: str,
    top_k: int,
    t0: int,
    model_id: str,
    prompt_key: str,
    seed_global: int,
    noise_seed: int,
    sparc_hparams: SparcHyperparams | None,
) -> dict:
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

    row = {
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
        "head_tail_ratio": None,  # deprecated
        "model_id": model_id,
        "prompt_key": prompt_key,
        "seed_global": int(seed_global),
        "noise_seed": int(noise_seed),
        "timestamp_iso": _iso_now(),
        "caption_tokens": None,
        "condition": condition,
        "free_caption": free_caption,
        "degenerated": bool(degenerated),
        "degeneration_reason": degeneration_reason if degenerated else None,
        "sparc_alpha": float(sparc_hparams.alpha) if sparc_hparams else None,
        "sparc_beta": float(sparc_hparams.beta) if sparc_hparams else None,
        "sparc_tau": float(sparc_hparams.tau) if sparc_hparams else None,
        "sparc_selected_layer": int(sparc_hparams.selected_layer) if sparc_hparams else None,
        "sparc_se_layer_lo": int(sparc_hparams.se_layers[0]) if sparc_hparams else None,
        "sparc_se_layer_hi": int(sparc_hparams.se_layers[1]) if sparc_hparams else None,
    }
    # Pad missing schema fields to None so write_metrics_table doesn't
    # silently drop a column on a partial row.
    for field in METRICS_SCHEMA:
        row.setdefault(field.name, None)
    return row


def _write_unit_case(
    *,
    run_dir: Path,
    image_id: str,
    image_path: Path,
    prompt: str,
    row_off: dict | None,
    row_on: dict | None,
    overwrite: bool,
) -> Path:
    """Write the Fig-3 data JSON for one image, with both OFF and ON slots.

    Mirrors the format from script 08 but adds the SPARC ON pairing and
    the free-caption + degeneration flags from Block 6.
    """
    target_dir = run_dir / "unit_cases"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{image_id}.json"
    if target.exists() and not overwrite:
        return target

    def _coerce(v):
        if isinstance(v, np.generic):
            v = v.item()
        if isinstance(v, float) and not np.isfinite(v):
            return None
        return v

    def _slot(row: dict | None) -> dict | None:
        if row is None:
            return None
        return {
            "caption_ref": row.get("caption_ref"),
            "free_caption": row.get("free_caption"),
            "share_tail": _coerce(row.get("share_tail")),
            "residual_ratio": _coerce(row.get("residual_ratio")),
            "degenerated": bool(row.get("degenerated", False)),
            "degeneration_reason": row.get("degeneration_reason"),
            "caption_len": row.get("caption_len"),
            "n_layers": row.get("n_layers"),
        }

    payload = {
        "image_id": image_id,
        "image_path": str(image_path),
        "prompt": prompt,
        "metrics_parquet_off": (
            f"results/diagnostico/llava-1.5-7b/{run_dir.parent.name}/"
            f"{run_dir.name}/metrics.parquet"
        ),
        "metrics_parquet_on": (
            f"results/avaliacao/llava-1.5-7b/{run_dir.parent.name}/"
            f"{run_dir.name}/metrics.parquet"
        ),
        "baseline": _slot(row_off),
        "sparc": _slot(row_on),
        # Slots populated by a future CHAIR-on-diagnostic pass.
        "hallucinated_objects_baseline": [],
        "hallucinated_objects_sparc": [],
        "notes": "kl / cos_dist matrices live in metrics.parquet (one row per condition).",
    }
    target.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    return target


# --------------------------------------------------------- per-length loop


def _process_length(
    *,
    length: str,
    cfg: dict,
    model_wrapper,
    lm_head,
    image_files: list[Path],
    sparc_hparams: SparcHyperparams,
    overwrite: bool,
    cells_total: int,
    cells_done_total: list[int],  # mutable for ETA bookkeeping across lengths
    t_start: float,
) -> tuple[int, int]:
    """Process all images for one length. Returns ``(n_done, n_skipped)``."""
    model_key = str(cfg["model"]["key"])
    model_id = str(cfg["model"]["model_id"])
    prompt_key = str(cfg["task"]["prompt_key"])
    prompt = get_prompt(prompt_key)
    seed_global = int(cfg["run"]["seed_global"])
    max_new_tokens = int(cfg["generation"]["max_new_tokens"])
    top_k = int(cfg["metrics"]["logits_top_k"])
    t0 = int(cfg["residual"]["t0"])

    rep_pen = float(cfg["generation"].get("repetition_penalty", 1.15))
    gen_kwargs = {
        "do_sample": False,
        "num_beams": 1,
        "repetition_penalty": rep_pen,
    }

    # Run dirs (NO timestamp — resumable).
    diag_root = area_root(
        cfg["run"]["output_root"], area="diagnostico",
        model_key=model_key, length=length,
    )
    eval_root = area_root(
        cfg["run"]["output_root"], area="avaliacao",
        model_key=model_key, length=length,
    )
    run_name = cfg["__run_name__"]
    diag_run = diag_root / run_name
    eval_run = eval_root / run_name
    diag_run.mkdir(parents=True, exist_ok=True)
    eval_run.mkdir(parents=True, exist_ok=True)

    diag_parq = diag_run / "metrics.parquet"
    eval_parq = eval_run / "metrics.parquet"
    captions_jsonl = diag_run / "captions.jsonl"

    # Load existing rows (resume).
    rows_off = _load_existing_rows(diag_parq) if not overwrite else []
    rows_on = _load_existing_rows(eval_parq) if not overwrite else []
    done_off = {r["image_id"] for r in rows_off if r.get("condition") == "off"}
    done_on = {r["image_id"] for r in rows_on if r.get("condition") == "on"}
    caption_cache = _read_caption_cache(captions_jsonl) if not overwrite else {}

    if overwrite:
        # Wipe artifacts so the run starts clean.
        for p in (diag_parq, eval_parq, captions_jsonl):
            if p.exists():
                p.unlink()

    logger.info(
        f"[{length}] resume: diag={len(done_off)} rows, eval={len(done_on)} rows, "
        f"captions_cached={len(caption_cache)}"
    )

    n_done = 0
    n_skipped = 0
    for image_path in image_files:
        image_id = Path(image_path).stem

        # Both cells already done?
        if image_id in done_off and image_id in done_on:
            n_skipped += 2
            cells_done_total[0] += 2
            continue

        # Load image + caption_baseline (cached or freshly generated).
        image = Image.open(image_path).convert("RGB")
        noise_seed = derive_image_seed(seed_global, image_id)
        noise_img = noise_image_uniform(image, seed=int(noise_seed))

        caption_baseline = caption_cache.get(image_id)
        if caption_baseline is None:
            t_g = time.time()
            try:
                caption_baseline = model_wrapper.generate_caption(
                    image=image, prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    seed=int(noise_seed),
                    generation_kwargs=gen_kwargs,
                )
            except Exception as exc:
                logger.error(f"[{length}|{image_id}] caption_baseline FAILED: {exc}")
                logger.error(traceback.format_exc())
                continue
            _append_caption(captions_jsonl, {
                "image_id": image_id, "length": length,
                "caption_baseline": caption_baseline,
                "timestamp_iso": _iso_now(),
                "generate_seconds": round(time.time() - t_g, 2),
            })
            caption_cache[image_id] = caption_baseline

        if not caption_baseline.strip():
            logger.warning(f"[{length}|{image_id}] empty baseline caption — skipping")
            n_skipped += 2
            cells_done_total[0] += 2
            continue

        base_loop, base_why = detect_loop(caption_baseline)

        # --- OFF cell ----
        row_off: dict | None = next(
            (r for r in rows_off if r["image_id"] == image_id and r.get("condition") == "off"),
            None,
        )
        if image_id not in done_off:
            t_cell = time.time()
            try:
                off_A = collect_forced_decoding(model_wrapper, image, prompt, caption_baseline)
                off_B = collect_forced_decoding(model_wrapper, noise_img, prompt, caption_baseline)
                row_off = _build_row(
                    lm_head=lm_head, result_A=off_A, result_B=off_B,
                    image_id=image_id, length=length, condition="off",
                    caption_ref=caption_baseline, free_caption=caption_baseline,
                    degenerated=base_loop, degeneration_reason=base_why,
                    top_k=top_k, t0=t0, model_id=model_id, prompt_key=prompt_key,
                    seed_global=seed_global, noise_seed=int(noise_seed),
                    sparc_hparams=None,
                )
                rows_off = _upsert(rows_off, row_off)
                _atomic_write_parquet(rows_off, diag_parq)
                cells_done_total[0] += 1
                eta = _eta(cells_done_total[0], cells_total, t_start)
                logger.info(
                    f"[{length}|{image_id}|off] OK share_tail={row_off['share_tail']:.4f} "
                    f"deg={base_loop} ({time.time() - t_cell:.1f}s) "
                    f"progress {cells_done_total[0]}/{cells_total} ETA {eta}"
                )
                del off_A, off_B
                n_done += 1
            except Exception as exc:
                logger.error(f"[{length}|{image_id}|off] FAILED: {exc}")
                logger.error(traceback.format_exc())
                continue

        # --- ON cell ----
        if image_id not in done_on:
            t_cell = time.time()
            try:
                # SPARC free caption (same seed/prompt as baseline).
                input_len, image_positions = _probe_sparc_layout(
                    model_wrapper, image, prompt,
                )
                with enable_sparc(
                    model_wrapper, hparams=sparc_hparams,
                    probe_image=image, prompt=prompt,
                ) as buffer:
                    buffer.reset()
                    buffer.update_input_len(input_len)
                    buffer.update_image_positions(image_positions)
                    caption_sparc = model_wrapper.generate_caption(
                        image=image, prompt=prompt,
                        max_new_tokens=max_new_tokens,
                        seed=int(noise_seed),
                        generation_kwargs=gen_kwargs,
                    )
                sparc_loop, sparc_why = detect_loop(caption_sparc)

                # SPARC FD on the SAME caption_baseline target — clean OFF↔ON delta.
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
                        image_id=image_id, length=length, condition="on",
                        caption_ref=caption_baseline, free_caption=caption_sparc,
                        degenerated=sparc_loop, degeneration_reason=sparc_why,
                        top_k=top_k, t0=t0, model_id=model_id, prompt_key=prompt_key,
                        seed_global=seed_global, noise_seed=int(noise_seed),
                        sparc_hparams=sparc_hparams,
                    )
                    rows_on = _upsert(rows_on, row_on)
                    _atomic_write_parquet(rows_on, eval_parq)
                del on_A, on_B
                cells_done_total[0] += 1
                eta = _eta(cells_done_total[0], cells_total, t_start)
                logger.info(
                    f"[{length}|{image_id}|on ] OK share_tail={row_on['share_tail']:.4f} "
                    f"deg={sparc_loop} ({time.time() - t_cell:.1f}s) "
                    f"progress {cells_done_total[0]}/{cells_total} ETA {eta}"
                )
                n_done += 1
            except Exception as exc:
                logger.error(f"[{length}|{image_id}|on] FAILED: {exc}")
                logger.error(traceback.format_exc())
                continue

        # Unit case JSON (overwrite each pass — small file, both slots).
        if row_off is not None or (row_on := next(
            (r for r in rows_on if r["image_id"] == image_id and r.get("condition") == "on"), None,
        )) is not None:
            row_on_for_json = next(
                (r for r in rows_on if r["image_id"] == image_id and r.get("condition") == "on"),
                None,
            )
            _write_unit_case(
                run_dir=diag_run, image_id=image_id, image_path=image_path,
                prompt=prompt, row_off=row_off, row_on=row_on_for_json,
                overwrite=True,
            )

    return n_done, n_skipped


def _eta(done: int, total: int, t_start: float) -> str:
    if done == 0:
        return "?"
    elapsed = time.time() - t_start
    per_cell = elapsed / done
    remaining = max(0, total - done)
    seconds = per_cell * remaining
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}min"
    return f"{seconds / 3600:.2f}h"


# ----------------------------------------------------------------- summary


def _final_summary(*, output_root: Path, model_key: str, lengths: list[str], run_name: str) -> dict:
    """Read back the parquets and compute the per-length OFF/ON medians."""
    summary: dict[str, Any] = {"per_length": {}, "totals": {}}
    total_off = total_on = 0
    deg_off = deg_on = 0
    for length in lengths:
        diag = (area_root(output_root, area="diagnostico",
                          model_key=model_key, length=length)
                / run_name / "metrics.parquet")
        eval_p = (area_root(output_root, area="avaliacao",
                            model_key=model_key, length=length)
                  / run_name / "metrics.parquet")
        rows_off = _load_existing_rows(diag) if diag.exists() else []
        rows_on = _load_existing_rows(eval_p) if eval_p.exists() else []
        st_off = [r["share_tail"] for r in rows_off
                  if r.get("share_tail") is not None and np.isfinite(r["share_tail"])]
        st_on = [r["share_tail"] for r in rows_on
                 if r.get("share_tail") is not None and np.isfinite(r["share_tail"])]
        n_deg_off = sum(1 for r in rows_off if r.get("degenerated"))
        n_deg_on = sum(1 for r in rows_on if r.get("degenerated"))
        summary["per_length"][length] = {
            "n_off": len(rows_off),
            "n_on": len(rows_on),
            "median_share_tail_off": (statistics.median(st_off) if st_off else None),
            "median_share_tail_on": (statistics.median(st_on) if st_on else None),
            "pct_degenerated_off": (
                100 * n_deg_off / len(rows_off) if rows_off else None
            ),
            "pct_degenerated_on": (
                100 * n_deg_on / len(rows_on) if rows_on else None
            ),
        }
        total_off += len(rows_off)
        total_on += len(rows_on)
        deg_off += n_deg_off
        deg_on += n_deg_on

    summary["totals"] = {
        "n_off": total_off, "n_on": total_on,
        "n_degenerated_off": deg_off, "n_degenerated_on": deg_on,
    }
    return summary


# ----------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-name", type=str, default="run_all_v1",
        help="Run name (also the folder name). Same name = resume.",
    )
    parser.add_argument(
        "--limit", type=int, default=50,
        help="Max images per length. Default 50 (the full diagnostic set).",
    )
    parser.add_argument(
        "--lengths", nargs="+", default=["short", "medium", "long"],
        choices=("short", "medium", "long"),
        help="Which length(s) to run. Default: all three.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Wipe existing parquets / captions cache and recompute from scratch.",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke: limit=1, lengths=['long']. Confirms entrypoint + IO + log path.",
    )
    args = parser.parse_args()

    if args.smoke:
        args.limit = 1
        args.lengths = ["long"]
        args.run_name = f"{args.run_name}_smoke"

    # --- log sink ---
    # We pin the log to the FIRST length's run dir so tail -f works from a
    # single, predictable path. (All lengths share the same run_name.)
    first_cfg = load_config(Path(_LENGTH_CONFIGS[args.lengths[0]]))
    first_cfg["__run_name__"] = args.run_name
    model_key = str(first_cfg["model"]["key"])
    output_root = first_cfg["run"]["output_root"]
    log_run_dir = (
        area_root(output_root, area="diagnostico",
                  model_key=model_key,
                  length=length_from_prompt_key(str(first_cfg["task"]["prompt_key"])))
        / args.run_name
    )
    log_run_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_run_dir / "logs" / "run_all.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    # ``enqueue=True`` makes the writer signal-safe (tmux detach / SIGTERM).
    logger.add(str(log_file), enqueue=True, level="INFO")

    logger.info("=" * 70)
    logger.info(f"Block-6 orchestrator — run_name={args.run_name}")
    logger.info(
        f"lengths={args.lengths} limit={args.limit} "
        f"overwrite={args.overwrite} smoke={args.smoke}"
    )
    logger.info(f"log file: {log_file}")
    logger.info("=" * 70)

    # --- load model ONCE; shared across lengths ---
    device = select_device("cuda")
    model_wrapper = build_model(model_key)
    model_wrapper.model_id = str(first_cfg["model"]["model_id"])
    if hasattr(model_wrapper, "_dtype"):
        model_wrapper._dtype = resolve_dtype(str(first_cfg["model"]["dtype"]))  # noqa: SLF001
    logger.info(f"Loading {model_wrapper.model_id} on {device}...")
    model_wrapper.load(device)
    logger.info(f"Model loaded. n_layers={model_wrapper.n_layers}")
    lm_head = model_wrapper.get_lm_head()

    sparc_cfg = first_cfg.get("sparc", {})
    sparc_hparams = SparcHyperparams(
        alpha=float(sparc_cfg.get("alpha", 1.1)),
        beta=float(sparc_cfg.get("beta", 0.1)),
        tau=float(sparc_cfg.get("tau", 1.5)),
        selected_layer=int(sparc_cfg.get("selected_layer", 20)),
        se_layers=tuple(sparc_cfg.get("se_layers", (0, 31))),
    )
    logger.info(f"SPARC config: {sparc_hparams.as_dict()}")

    # --- plan ---
    # Per length we plan args.limit images × 2 conditions.
    total_cells = 2 * args.limit * len(args.lengths)
    logger.info(f"Planned cells: {total_cells} "
                f"({args.limit} imgs × {len(args.lengths)} lengths × 2 conditions)")

    cells_done_total = [0]
    t_start = time.time()
    n_done_grand = 0
    n_skip_grand = 0

    for length in args.lengths:
        cfg = load_config(Path(_LENGTH_CONFIGS[length]))
        cfg["__run_name__"] = args.run_name
        images_dir = Path(cfg["dataset"]["images_dir"])
        image_files = [Path(p) for p in
                       sorted(glob.glob(f"{images_dir}{os.sep}*.jpg"))[: args.limit]]
        if not image_files:
            logger.error(f"[{length}] no images under {images_dir}; skipping length.")
            continue

        n_done, n_skipped = _process_length(
            length=length, cfg=cfg, model_wrapper=model_wrapper, lm_head=lm_head,
            image_files=image_files, sparc_hparams=sparc_hparams,
            overwrite=args.overwrite, cells_total=total_cells,
            cells_done_total=cells_done_total, t_start=t_start,
        )
        n_done_grand += n_done
        n_skip_grand += n_skipped

    elapsed = time.time() - t_start
    logger.info("=" * 70)
    logger.info(
        f"DONE. cells_executed={n_done_grand} cells_skipped={n_skip_grand} "
        f"elapsed={elapsed / 60:.1f}min"
    )

    summary = _final_summary(
        output_root=Path(output_root), model_key=model_key,
        lengths=args.lengths, run_name=args.run_name,
    )
    summary_path = log_run_dir / "run_all_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )

    logger.info("Per-length results:")
    for length, s in summary["per_length"].items():
        logger.info(
            f"  [{length}] off n={s['n_off']} median_share_tail={s['median_share_tail_off']!s}  "
            f"on n={s['n_on']} median_share_tail={s['median_share_tail_on']!s}"
        )
        logger.info(
            f"  [{length}] degenerated off={s['pct_degenerated_off']!s}%  "
            f"on={s['pct_degenerated_on']!s}%"
        )
    logger.info(f"summary saved to: {summary_path}")
    logger.info(f"log file        : {log_file}")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover (operator-side)
        logger.error(f"top-level failure: {exc}")
        logger.error(traceback.format_exc())
        raise SystemExit(1)
