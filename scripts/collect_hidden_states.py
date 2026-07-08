#!/usr/bin/env python
"""Collect paired hidden states (A: real image, B: uniform noise) for every manifest entry.

Run: make baseline  (or: python scripts/collect_hidden_states.py --config configs/baseline.yaml)
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from PIL import Image

from vr_modality_bias.data.manifests import iter_manifest
from vr_modality_bias.data.prompts import get_prompt
from vr_modality_bias.experiment.reference import read_reference_captions
from vr_modality_bias.experiment.teacher_forcing import run_paired_for_image
from vr_modality_bias.io.storage import hidden_states_filename
from vr_modality_bias.models.registry import build_model
from vr_modality_bias.utils.config import load_config
from vr_modality_bias.utils.device import resolve_dtype, select_device
from vr_modality_bias.utils.logging import configure_logging, get_logger
from vr_modality_bias.utils.profiling import (
    Timer,
    cuda_peak_bytes,
    format_bytes,
    format_seconds,
    reset_cuda_peak,
    summarize_seconds,
)
from vr_modality_bias.utils.runs import current_run_dir
from vr_modality_bias.utils.seeds import derive_image_seed, set_global_seeds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_dir = current_run_dir(cfg["run"]["output_root"], cfg["run"]["name"])
    log_file = run_dir / "logs" / "collect_hidden_states.log"
    configure_logging(log_file=log_file)
    log = get_logger(__name__)
    log.info("Run dir: %s", run_dir)

    seed_global = int(cfg["run"]["seed_global"])
    set_global_seeds(seed_global)

    prompt_key = str(cfg["task"]["prompt_key"])
    prompt = get_prompt(prompt_key)

    model = build_model(cfg["model"]["key"])
    model.model_id = str(cfg["model"]["model_id"])
    dtype = resolve_dtype(str(cfg["model"]["dtype"]))
    device = select_device("cuda")
    log.info(
        "Loading model %s on %s (dtype=%s)…", model.model_id, device, dtype
    )
    if hasattr(model, "_dtype"):
        model._dtype = dtype  # noqa: SLF001
    model.load(device)
    log.info("Loaded. n_layers=%d", model.n_layers)

    captions = read_reference_captions(run_dir / "ref_captions.jsonl")
    log.info("Loaded %d reference captions.", len(captions))

    manifest_path = Path(cfg["dataset"]["manifest_path"])
    images_dir = Path(cfg["dataset"]["images_dir"])
    manifest = iter_manifest(manifest_path)
    if args.limit:
        manifest = itertools.islice(manifest, args.limit)

    hidden_states_dir = run_dir / "hidden_states"
    hidden_states_dir.mkdir(parents=True, exist_ok=True)

    diagnostics_path = run_dir / "diagnostics_collect.jsonl"
    if args.overwrite and diagnostics_path.exists():
        diagnostics_path.unlink()
    compression = str(cfg["io"]["compression"])
    compression_level = int(cfg["io"]["compression_level"])

    per_pair: list[dict] = []
    for record in manifest:
        path_A = hidden_states_dir / hidden_states_filename(record.image_id, "A")
        path_B = hidden_states_dir / hidden_states_filename(record.image_id, "B")

        if path_A.exists() and path_B.exists() and not args.overwrite:
            log.info("[%s] already present, skipping.", record.image_id)
            continue

        if record.image_id not in captions:
            raise KeyError(
                f"No reference caption for image_id={record.image_id!r}. "
                "Run scripts/generate_refs.py first."
            )
        caption_ref = str(captions[record.image_id]["caption_ref"])
        noise_seed = derive_image_seed(seed_global, record.image_id)

        image_path = images_dir / record.file_name
        with Image.open(image_path) as raw:
            image = raw.convert("RGB")

        reset_cuda_peak()
        with Timer() as timer:
            run_paired_for_image(
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

        entry = {
            "image_id": record.image_id,
            "seconds": float(timer.seconds),
            "vram_peak_bytes": int(peak_bytes),
            "bytes_pair": int(bytes_pair),
        }
        per_pair.append(entry)
        with diagnostics_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

        log.info(
            "[%s] saved (time=%s, vram_peak=%s, disk=%s)",
            record.image_id,
            format_seconds(timer.seconds),
            format_bytes(peak_bytes),
            format_bytes(bytes_pair),
        )

    stats = summarize_seconds(p["seconds"] for p in per_pair)
    peak_vram_run = max((p["vram_peak_bytes"] for p in per_pair), default=0)
    total_bytes_pairs = sum(p["bytes_pair"] for p in per_pair)
    median_bytes_pair: float | None = None
    if per_pair:
        sorted_sizes = sorted(p["bytes_pair"] for p in per_pair)
        median_bytes_pair = (
            sorted_sizes[len(sorted_sizes) // 2]
            if len(sorted_sizes) % 2 == 1
            else (
                sorted_sizes[len(sorted_sizes) // 2 - 1]
                + sorted_sizes[len(sorted_sizes) // 2]
            )
            / 2.0
        )

    log.info(
        "scripts/04 rollup — %d pair(s), total=%s, median/pair=%s, "
        "VRAM peak=%s, disk total=%s (median/pair=%s)",
        stats["n"],
        format_seconds(stats["total"]),
        format_seconds(stats["median"]),
        format_bytes(peak_vram_run),
        format_bytes(total_bytes_pairs),
        format_bytes(median_bytes_pair) if median_bytes_pair is not None else "n/a",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
