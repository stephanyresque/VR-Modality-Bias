#!/usr/bin/env python
"""Per-image caption-sweep cell: TF + metrics + (optional) discard, one image at a time."""

from __future__ import annotations

import argparse
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from vr_modality_bias.data.manifests import iter_manifest
from vr_modality_bias.data.prompts import get_prompt
from vr_modality_bias.experiment.teacher_forcing import run_paired_for_image
from vr_modality_bias.io.results import (
    compute_summary_stats,
    write_metrics_table,
    write_summary_csv,
    write_summary_json,
)
from vr_modality_bias.io.storage import hidden_states_filename, load_hidden_states
from vr_modality_bias.metrics.cosine import compute_cosine_distance_matrix
from vr_modality_bias.metrics.kl import compute_kl_matrix
from vr_modality_bias.metrics.residual import residual_drift_ratio
from vr_modality_bias.models.registry import build_model
from vr_modality_bias.utils.config import load_config, snapshot_config
from vr_modality_bias.utils.device import resolve_dtype, select_device
from vr_modality_bias.utils.logging import configure_logging, get_logger
from vr_modality_bias.utils.profiling import (
    Timer,
    cuda_peak_bytes,
    dir_size_bytes,
    format_bytes,
    format_seconds,
    reset_cuda_peak,
    summarize_seconds,
)
from vr_modality_bias.utils.runs import make_run_dir
from vr_modality_bias.utils.seeds import derive_image_seed, set_global_seeds
from vr_modality_bias.utils.tokens import decode_caption_tokens


def _iso_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    run_dir = make_run_dir(cfg["run"]["output_root"], cfg["run"]["name"])
    log_file = run_dir / "logs" / "09_run_caption_sweep.log"
    configure_logging(log_file=log_file)
    log = get_logger(__name__)
    log.info("Run dir: %s", run_dir)
    log.info("Config: %s", args.config)
    snapshot_config(args.config, run_dir)

    seed_global = int(cfg["run"]["seed_global"])
    set_global_seeds(seed_global)

    prompt_key = str(cfg["task"]["prompt_key"])
    prompt = get_prompt(prompt_key)
    log.info("Prompt key: %s", prompt_key)

    # ---- model load ----
    model = build_model(cfg["model"]["key"])
    model.model_id = str(cfg["model"]["model_id"])
    dtype = resolve_dtype(str(cfg["model"]["dtype"]))
    device = select_device("cuda")
    log.info("Loading model %s on %s (dtype=%s)…", model.model_id, device, dtype)
    if hasattr(model, "_dtype"):
        model._dtype = dtype  
    model.load(device)
    lm_head = model.get_lm_head()
    log.info("Loaded. n_layers=%d", model.n_layers)

    # ---- io config ----
    discard_h5 = bool(cfg.get("io", {}).get("discard_hidden_states_after_metrics", False))
    compression = str(cfg["io"]["compression"])
    compression_level = int(cfg["io"]["compression_level"])
    log.info("discard_hidden_states_after_metrics = %s", discard_h5)

    # ---- generation config ----
    gen_kwargs = {
        "do_sample": bool(cfg["generation"]["do_sample"]),
        "temperature": float(cfg["generation"]["temperature"]),
        "top_p": float(cfg["generation"]["top_p"]),
        "repetition_penalty": float(cfg["generation"]["repetition_penalty"]),
    }
    max_new_tokens = int(cfg["generation"]["max_new_tokens"])

    top_k = int(cfg["metrics"]["logits_top_k"])
    t0 = int(cfg["residual"]["t0"])

    # ---- manifest ----
    manifest_path = Path(cfg["dataset"]["manifest_path"])
    images_dir = Path(cfg["dataset"]["images_dir"])
    manifest = list(iter_manifest(manifest_path))
    # Respect cfg.dataset.n_images: the shared manifest can hold more entries
    # (e.g. the baseline kept 100), but each sweep cell should process exactly
    # the first n_images for cross-cell comparability.
    n_images = int(cfg["dataset"].get("n_images", len(manifest)))
    if n_images < len(manifest):
        manifest = manifest[:n_images]
    if args.limit:
        manifest = manifest[: args.limit]
    log.info("Processing %d image(s) (cfg.n_images=%d).", len(manifest), n_images)

    # ---- output paths ----
    hidden_states_dir = run_dir / "hidden_states"
    hidden_states_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.parquet"
    ref_captions_path = run_dir / "ref_captions.jsonl"
    diagnostics_path = run_dir / "diagnostics_collect.jsonl"

    rows: list[dict] = []
    per_pair_stats: list[dict] = []

    for record in manifest:
        image_path = images_dir / record.file_name
        with Image.open(image_path) as raw:
            image = raw.convert("RGB")

        noise_seed = derive_image_seed(seed_global, record.image_id)

        caption_ref = model.generate_caption(
            image=image,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            seed=noise_seed,
            generation_kwargs=gen_kwargs,
        )
        with ref_captions_path.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "image_id": record.image_id,
                        "caption_ref": caption_ref,
                        "model_id": model.model_id,
                        "prompt_key": prompt_key,
                        "noise_seed": int(noise_seed),
                        "seed_global": int(seed_global),
                        "timestamp": _iso_now(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        log.info(
            "[%s] caption_ref: %s",
            record.image_id,
            (caption_ref[:80] + "...") if len(caption_ref) > 80 else caption_ref,
        )

        reset_cuda_peak()
        with Timer() as timer:
            path_A, path_B = run_paired_for_image(
                model=model,
                image_id=record.image_id,
                image=image,
                prompt=prompt,
                prompt_key=prompt_key,
                caption_ref=caption_ref,
                out_dir=hidden_states_dir,
                seed_global=seed_global,
                noise_seed=noise_seed,
                compression=compression,
                compression_level=compression_level,
            )
        peak_bytes = cuda_peak_bytes()
        bytes_pair = path_A.stat().st_size + path_B.stat().st_size

        result_A = load_hidden_states(path_A)
        result_B = load_hidden_states(path_B)
        kl = compute_kl_matrix(
            lm_head,
            result_A.hidden_states,
            result_B.hidden_states,
            caption_start=int(result_A.caption_start),
            caption_len=int(result_A.caption_len),
            top_k=top_k,
        )
        cos = compute_cosine_distance_matrix(
            result_A.hidden_states,
            result_B.hidden_states,
            caption_start=int(result_A.caption_start),
            caption_len=int(result_A.caption_len),
        )
        rr = residual_drift_ratio(kl, t0=t0)

        caption_tokens = decode_caption_tokens(
            model,
            result_A.input_ids,
            caption_start=int(result_A.caption_start),
        )

        meta = result_A.metadata
        rows.append(
            {
                "image_id": record.image_id,
                "caption_len": int(result_A.caption_len),
                "n_layers": int(result_A.hidden_states.shape[0]),
                "hidden_dim": int(result_A.hidden_states.shape[-1]),
                "caption_ref": str(caption_ref),
                "kl": kl,
                "cos_dist": cos,
                "residual_ratio": float(rr),
                "model_id": str(meta.get("model_id", model.model_id)),
                "prompt_key": prompt_key,
                "seed_global": int(seed_global),
                "noise_seed": int(noise_seed),
                "timestamp_iso": str(meta.get("timestamp_iso", _iso_now())),
                "caption_tokens": caption_tokens,
            }
        )

        entry = {
            "image_id": record.image_id,
            "seconds": float(timer.seconds),
            "vram_peak_bytes": int(peak_bytes),
            "bytes_pair": int(bytes_pair),
        }
        per_pair_stats.append(entry)
        with diagnostics_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

        log.info(
            "[%s] saved (time=%s, vram_peak=%s, disk=%s, rr=%.4f, caption_len=%d)",
            record.image_id,
            format_seconds(timer.seconds),
            format_bytes(peak_bytes),
            format_bytes(bytes_pair),
            rr if rr == rr else float("nan"),  
            int(result_A.caption_len),
        )

        if discard_h5:
            path_A.unlink(missing_ok=True)
            path_B.unlink(missing_ok=True)

        write_metrics_table(rows, metrics_path)

    log.info("Wrote %d row(s) to %s.", len(rows), metrics_path)

    summary_csv_path = run_dir / "summary.csv"
    summary_json_path = run_dir / "summary.json"
    diagnostics_summary_path = run_dir / "diagnostics.json"

    write_summary_csv(rows, summary_csv_path)
    stats = compute_summary_stats(rows)
    write_summary_json(stats, summary_json_path)
    log.info(
        "residual_ratio summary — n_finite_in_range=%d/%d, median=%s, iqr=%s",
        stats["n_residual_ratio_finite_in_range"],
        stats["n_images"],
        stats["residual_ratio"].get("median"),
        stats["residual_ratio"].get("iqr"),
    )

    if per_pair_stats:
        secs = [p["seconds"] for p in per_pair_stats]
        seconds_stats = summarize_seconds(secs)
        bytes_pair = [p["bytes_pair"] for p in per_pair_stats]
        diagnostics = {
            "n_pairs": len(per_pair_stats),
            "time_total_seconds": float(seconds_stats["total"]),
            "time_median_seconds_per_pair": seconds_stats["median"],
            "time_mean_seconds_per_pair": seconds_stats["mean"],
            "vram_peak_bytes": int(max(p["vram_peak_bytes"] for p in per_pair_stats)),
            "disk_total_bytes_pairs": int(sum(bytes_pair)),
            "disk_median_bytes_per_pair": (
                float(sorted(bytes_pair)[len(bytes_pair) // 2]) if bytes_pair else None
            ),
            "disk_total_bytes_run": int(dir_size_bytes(run_dir)),
            "discard_hidden_states_after_metrics": discard_h5,
        }
        with diagnostics_summary_path.open("w", encoding="utf-8") as fh:
            json.dump(diagnostics, fh, indent=2)
            fh.write("\n")
        log.info(
            "Diagnostics — pairs=%d, total=%s, median/pair=%s, VRAM peak=%s, disk run=%s",
            diagnostics["n_pairs"],
            format_seconds(diagnostics["time_total_seconds"]),
            format_seconds(diagnostics["time_median_seconds_per_pair"]),
            format_bytes(diagnostics["vram_peak_bytes"]),
            format_bytes(diagnostics["disk_total_bytes_run"]),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
