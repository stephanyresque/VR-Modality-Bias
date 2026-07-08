#!/usr/bin/env python
"""Phase 2 sweep — 50 images × 3 lengths × {SPARC OFF + α-sweep ON}. Resumable.

Run: make phase2  (smoke: make phase2-smoke; override run dir: make phase2 PHASE2_RUN_NAME=my_run)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger
from PIL import Image
from pyprojroot import here

try:
    from vr_modality_bias.data.perturbations import noise_image_uniform
    from vr_modality_bias.data.prompts import get_prompt
    from vr_modality_bias.experiment.forced_decoding import collect_forced_decoding
    from vr_modality_bias.experiment.sparc import SparcHyperparams, enable_sparc
    from vr_modality_bias.metrics.cosine import compute_cosine_distance_matrix
    from vr_modality_bias.metrics.kl import compute_kl_matrix
    from vr_modality_bias.metrics.residual import head_tail_ratio, residual_drift_ratio
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
    from src.vr_modality_bias.metrics.cosine import compute_cosine_distance_matrix
    from src.vr_modality_bias.metrics.kl import compute_kl_matrix
    from src.vr_modality_bias.metrics.residual import head_tail_ratio, residual_drift_ratio
    from src.vr_modality_bias.models.registry import build_model
    from src.vr_modality_bias.utils.config import load_config
    from src.vr_modality_bias.utils.device import resolve_dtype, select_device
    from src.vr_modality_bias.utils.seeds import derive_image_seed


PHASE2_SCHEMA = pa.schema(
    [
        pa.field("image_id", pa.string()),
        pa.field("length", pa.string()),
        pa.field("condition", pa.string()),  # "off" or "on"
        pa.field("alpha", pa.float32(), nullable=True),
        pa.field("tau", pa.float32(), nullable=True),
        pa.field("selected_layer", pa.int32(), nullable=True),
        pa.field("se_layer_lo", pa.int32(), nullable=True),
        pa.field("se_layer_hi", pa.int32(), nullable=True),
        pa.field("beta", pa.float32(), nullable=True),
        pa.field("caption_len", pa.int32()),
        pa.field("caption_ref", pa.string()),
        pa.field("kl", pa.list_(pa.list_(pa.float32()))),
        pa.field("cos_dist", pa.list_(pa.list_(pa.float32()))),
        pa.field("residual_ratio", pa.float32()),
        pa.field("head_tail_ratio", pa.float32(), nullable=True),
        pa.field("model_id", pa.string()),
        pa.field("prompt_key", pa.string()),
        pa.field("seed_global", pa.int32()),
        pa.field("noise_seed", pa.int64()),
        pa.field("timestamp_iso", pa.string()),
    ]
)


def _iso_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def _matrix_to_nested(arr) -> list[list[float]]:
    if arr is None:
        return []
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim != 2:
        raise ValueError(f"expected 2-D matrix, got shape {a.shape}")
    return a.tolist()


def _cell_key(row: dict) -> tuple:
    """Stable identity of a cell. Alpha is rounded to 4dp for set membership."""
    a = row.get("alpha")
    return (
        row["image_id"],
        row["condition"],
        round(float(a), 4) if a is not None else None,
    )


def _load_existing(parquet_path: Path) -> tuple[set, list[dict]]:
    """Return (done_set, rows) from an existing parquet (or empty if missing)."""
    if not parquet_path.exists():
        return set(), []
    table = pq.read_table(parquet_path)
    rows = table.to_pylist()
    done = {_cell_key(r) for r in rows}
    return done, rows


def _atomic_write_parquet(rows: list[dict], path: Path) -> None:
    """Write parquet atomically (write to tmp, then rename). Survives a kill mid-write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    columns: dict[str, list[Any]] = {f.name: [] for f in PHASE2_SCHEMA}
    for row in rows:
        for f in PHASE2_SCHEMA:
            v = row.get(f.name)
            if f.name in ("kl", "cos_dist"):
                v = _matrix_to_nested(v) if v is not None else []
            columns[f.name].append(v)
    table = pa.Table.from_pydict(columns, schema=PHASE2_SCHEMA)
    pq.write_table(table, tmp)
    os.replace(tmp, path)


def _upsert(rows: list[dict], new_row: dict) -> list[dict]:
    """Replace the row with the same cell key, or append if new."""
    key = _cell_key(new_row)
    out = [r for r in rows if _cell_key(r) != key]
    out.append(new_row)
    return out


def _load_caption_cache(jsonl_path: Path) -> dict[str, str]:
    """Read ``ref_captions.jsonl`` → ``{image_id: caption_ref}``.

    Lines without a usable caption (empty / corrupted) are silently skipped.
    """
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
            image_id = entry.get("image_id")
            caption = entry.get("caption_ref")
            if image_id and caption:
                cache[str(image_id)] = str(caption)
    return cache


def _append_caption(jsonl_path: Path, entry: dict) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def _build_row(
    *,
    lm_head,
    result_A,
    result_B,
    image_id: str,
    length: str,
    condition: str,
    hparams: SparcHyperparams | None,
    caption_ref: str,
    model_id: str,
    prompt_key: str,
    seed_global: int,
    noise_seed: int,
    top_k: int,
    t0: int,
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
    htr = float(head_tail_ratio(kl, t0=t0))

    row = {
        "image_id": image_id,
        "length": length,
        "condition": condition,
        "alpha": float(hparams.alpha) if hparams else None,
        "tau": float(hparams.tau) if hparams else None,
        "selected_layer": int(hparams.selected_layer) if hparams else None,
        "se_layer_lo": int(hparams.se_layers[0]) if hparams else None,
        "se_layer_hi": int(hparams.se_layers[1]) if hparams else None,
        "beta": float(hparams.beta) if hparams else None,
        "caption_len": int(result_A.caption_len),
        "caption_ref": caption_ref,
        "kl": kl,
        "cos_dist": cos,
        "residual_ratio": rr,
        "head_tail_ratio": htr,
        "model_id": model_id,
        "prompt_key": prompt_key,
        "seed_global": int(seed_global),
        "noise_seed": int(noise_seed),
        "timestamp_iso": _iso_now(),
    }
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-name", type=str, default="phase2_alpha_sweep",
        help="Run directory name under results/runs/<run-name>.",
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("results/runs"),
    )
    parser.add_argument(
        "--lengths", nargs="+", default=["short", "medium", "long"],
        choices=["short", "medium", "long"],
    )
    parser.add_argument(
        "--alphas", nargs="+", type=float, default=[1.1, 1.2, 1.3, 1.4, 1.5],
    )
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--selected-layer", type=int, default=15)
    parser.add_argument("--se-layers", type=int, nargs=2, default=(0, 31))
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Truncate image list (e.g. --limit 50 for the full sweep, --limit 1 for smoke).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Recompute already-done cells (default: skip).",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Quick smoke: 1 image, short length, α=1.3 only. Confirms entrypoint + IO + log path.",
    )
    parser.add_argument(
        "--config-dir", type=Path, default=Path("configs"),
        help="Where to find run_qwen7b_<length>.yaml configs.",
    )
    args = parser.parse_args()

    if args.smoke:
        args.limit = 1
        args.lengths = ["short"]
        args.alphas = [1.3]

    run_dir = Path(args.output_root) / args.run_name
    log_file = run_dir / "logs" / "phase2.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    # File sink: every message lands in phase2.log, with timestamps. enqueue=True
    # makes the writer thread-safe and resilient under tmux + signal interruption.
    logger.add(str(log_file), enqueue=True, level="INFO")
    logger.info("=" * 70)
    logger.info(f"Phase 2 sweep — run_name={args.run_name}")
    logger.info(f"lengths={args.lengths}  alphas={args.alphas}  limit={args.limit}  overwrite={args.overwrite}  smoke={args.smoke}")
    logger.info(f"run dir : {run_dir}")
    logger.info(f"log file: {log_file}")
    logger.info("=" * 70)

    # Snapshot the run params for reproducibility.
    snapshot = {
        "run_name": args.run_name,
        "lengths": args.lengths,
        "alphas": args.alphas,
        "tau": args.tau,
        "selected_layer": args.selected_layer,
        "se_layers": list(args.se_layers),
        "beta": args.beta,
        "limit": args.limit,
        "overwrite": args.overwrite,
        "smoke": args.smoke,
        "timestamp_iso": _iso_now(),
    }
    (run_dir / "run_params.json").write_text(
        json.dumps(snapshot, indent=2) + "\n", encoding="utf-8"
    )

    # Pull model spec from the first length config (they share the model
    # across short/medium/long — only prompt and gen-params change).
    first_cfg = load_config(args.config_dir / f"run_qwen7b_{args.lengths[0]}.yaml")
    model = build_model(str(first_cfg["model"]["key"]))
    model.model_id = str(first_cfg["model"]["model_id"])
    dtype = resolve_dtype(str(first_cfg["model"]["dtype"]))
    device = select_device("cuda")
    if hasattr(model, "_dtype"):
        model._dtype = dtype  # noqa: SLF001
    logger.info(f"Loading model {model.model_id} on {device} (dtype={dtype})...")
    model.load(device)
    lm_head = model.get_lm_head()
    logger.info(f"Loaded. n_layers={model.n_layers}")

    images_dir = first_cfg["dataset"]["images_dir"]
    image_files = sorted(glob.glob(f"{images_dir}{os.sep}*.jpg"))
    if args.limit is not None:
        image_files = image_files[: args.limit]
    if not image_files:
        logger.error(f"No images under {images_dir}")
        return 1
    n_images = len(image_files)
    logger.info(f"{n_images} image(s) to process.")

    # Probe image — used by enable_sparc to discover image_token_index
    # (constant for a given prompt template, so first image is fine).
    with Image.open(image_files[0]) as probe_raw:
        probe_image = probe_raw.convert("RGB")

    cells_per_image = 1 + len(args.alphas)  # 1 OFF + N ON
    total_cells = len(args.lengths) * n_images * cells_per_image
    logger.info(f"Total cells to evaluate: {total_cells} (per length: {n_images * cells_per_image})")

    t_start = time.time()
    cells_done = 0
    cells_skipped = 0
    cells_failed = 0

    for length_name in args.lengths:
        cfg = load_config(args.config_dir / f"run_qwen7b_{length_name}.yaml")
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

        length_dir = run_dir / length_name
        length_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = length_dir / "metrics_sweep.parquet"
        ref_jsonl = length_dir / "ref_captions.jsonl"

        done_cells, rows = _load_existing(parquet_path)
        cached_captions = _load_caption_cache(ref_jsonl)
        logger.info(
            f"[{length_name}] resume state: {len(done_cells)} cells in parquet, "
            f"{len(cached_captions)} captions cached"
        )

        for image_path in image_files:
            image_id = Path(image_path).stem
            with Image.open(image_path) as raw:
                image = raw.convert("RGB")
            noise_seed = derive_image_seed(seed_global, image_id)
            noise_img = noise_image_uniform(image, seed=int(noise_seed))

            # caption_ref — generate once per (length, image), cache to disk.
            if image_id in cached_captions:
                caption_ref = cached_captions[image_id]
            else:
                caption_ref = model.generate_caption(
                    image=image, prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    seed=int(noise_seed),
                    generation_kwargs=gen_kwargs,
                )
                if not caption_ref.strip():
                    logger.warning(f"[{length_name}|{image_id}] empty caption — skipping image (all cells).")
                    cells_skipped += cells_per_image
                    continue
                _append_caption(ref_jsonl, {
                    "image_id": image_id,
                    "length": length_name,
                    "caption_ref": caption_ref,
                    "model_id": model.model_id,
                    "prompt_key": prompt_key,
                    "noise_seed": int(noise_seed),
                    "seed_global": int(seed_global),
                    "timestamp": _iso_now(),
                })
                cached_captions[image_id] = caption_ref

            # ---- OFF cell ----
            off_key = (image_id, "off", None)
            if off_key in done_cells and not args.overwrite:
                cells_skipped += 1
                logger.info(
                    f"[{length_name}|{image_id}|off] SKIP (cached)  "
                    f"progress {cells_done + cells_skipped + cells_failed}/{total_cells}"
                )
            else:
                t_cell = time.time()
                try:
                    off_A = collect_forced_decoding(model, image, prompt, caption_ref)
                    off_B = collect_forced_decoding(model, noise_img, prompt, caption_ref)
                    row = _build_row(
                        lm_head=lm_head, result_A=off_A, result_B=off_B,
                        image_id=image_id, length=length_name,
                        condition="off", hparams=None,
                        caption_ref=caption_ref,
                        model_id=model.model_id, prompt_key=prompt_key,
                        seed_global=seed_global, noise_seed=int(noise_seed),
                        top_k=top_k, t0=t0,
                    )
                    rows = _upsert(rows, row)
                    _atomic_write_parquet(rows, parquet_path)
                    done_cells.add(off_key)
                    cells_done += 1
                    dt = time.time() - t_cell
                    eta_s = (time.time() - t_start) / max(cells_done, 1) * (total_cells - cells_done - cells_skipped - cells_failed)
                    logger.info(
                        f"[{length_name}|{image_id}|off] OK  "
                        f"htr={row['head_tail_ratio']:.4f} rr={row['residual_ratio']:.4f} "
                        f"caption_len={row['caption_len']}  ({dt:.1f}s)  "
                        f"progress {cells_done + cells_skipped + cells_failed}/{total_cells}  ETA {eta_s/60:.1f}min"
                    )
                except Exception as exc:
                    cells_failed += 1
                    logger.error(f"[{length_name}|{image_id}|off] FAILED: {exc}")
                    logger.error(traceback.format_exc())
                    # Don't skip α loop — ON might succeed; OFF will be retried next run.

            # ---- ON cells (one per α) ----
            for alpha in args.alphas:
                on_key = (image_id, "on", round(float(alpha), 4))
                if on_key in done_cells and not args.overwrite:
                    cells_skipped += 1
                    logger.info(
                        f"[{length_name}|{image_id}|on α={alpha}] SKIP (cached)  "
                        f"progress {cells_done + cells_skipped + cells_failed}/{total_cells}"
                    )
                    continue
                t_cell = time.time()
                try:
                    hparams = SparcHyperparams(
                        alpha=float(alpha),
                        tau=float(args.tau),
                        selected_layer=int(args.selected_layer),
                        se_layers=tuple(args.se_layers),
                        beta=float(args.beta),
                    )
                    with enable_sparc(
                        model, hparams=hparams,
                        probe_image=probe_image, prompt=prompt,
                    ) as buffer:
                        on_A = collect_forced_decoding(
                            model, image, prompt, caption_ref, sparc_buffer=buffer,
                        )
                        on_B = collect_forced_decoding(
                            model, noise_img, prompt, caption_ref, sparc_buffer=buffer,
                        )
                    row = _build_row(
                        lm_head=lm_head, result_A=on_A, result_B=on_B,
                        image_id=image_id, length=length_name,
                        condition="on", hparams=hparams,
                        caption_ref=caption_ref,
                        model_id=model.model_id, prompt_key=prompt_key,
                        seed_global=seed_global, noise_seed=int(noise_seed),
                        top_k=top_k, t0=t0,
                    )
                    rows = _upsert(rows, row)
                    _atomic_write_parquet(rows, parquet_path)
                    done_cells.add(on_key)
                    cells_done += 1
                    dt = time.time() - t_cell
                    eta_s = (time.time() - t_start) / max(cells_done, 1) * (total_cells - cells_done - cells_skipped - cells_failed)
                    logger.info(
                        f"[{length_name}|{image_id}|on α={alpha}] OK  "
                        f"htr={row['head_tail_ratio']:.4f} rr={row['residual_ratio']:.4f}  "
                        f"({dt:.1f}s)  "
                        f"progress {cells_done + cells_skipped + cells_failed}/{total_cells}  ETA {eta_s/60:.1f}min"
                    )
                except Exception as exc:
                    cells_failed += 1
                    logger.error(f"[{length_name}|{image_id}|on α={alpha}] FAILED: {exc}")
                    logger.error(traceback.format_exc())
                    continue

    elapsed_min = (time.time() - t_start) / 60
    logger.info("=" * 70)
    logger.info(
        f"Phase 2 sweep DONE.  "
        f"cells_done={cells_done}  skipped={cells_skipped}  failed={cells_failed}  "
        f"elapsed={elapsed_min:.1f}min"
    )
    logger.info(f"Output dir: {run_dir}")
    logger.info(f"Log file  : {log_file}")
    logger.info("=" * 70)
    return 0 if cells_failed == 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover (operator-side script)
        logger.error(f"Top-level exception: {exc}")
        logger.error(traceback.format_exc())
        raise SystemExit(1)
