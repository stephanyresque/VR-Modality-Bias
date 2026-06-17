#!/usr/bin/env python
"""Phase 3 — free caption generation: baseline (SPARC OFF) vs SPARC α=1.1.

Why this exists
---------------
Fase 2 showed htr is inflatable by SPARC itself, so CHAIR is the actual
evaluation metric for hallucination. α=1.1 was the only regime that didn't
degenerate the text in the fluency sample. This script generates the 50-image
× 3-length × 2-condition (off, on α=1.1) caption matrix that scripts/17 reads
to compute CHAIR.

Properties
----------
* Same seed/prompt per image across OFF and ON, so the pair is directly
  comparable for the side-by-side panel.
* Idempotent + resumable. Each generation appends a single line to
  captions.jsonl. Re-running picks up where it stopped (skips done cells).
* tmux-safe: file logging via loguru with ``enqueue=True``, no terminal
  dependence.
* Loads the model exactly once and reuses it across all 3 lengths and
  both conditions.

Output layout
-------------
    results/runs/<run-name>/
        captions.jsonl           — one JSON line per generation
        logs/phase3.log          — full log
        run_params.json          — snapshot of the CLI args

CLI
---
    make phase3-smoke              # 1 image, short, both conditions
    make phase3                    # full 50 × 3 × 2
    python scripts/18_phase3_generate.py --run-name X --limit 5 --lengths short
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

from loguru import logger
from PIL import Image
from pyprojroot import here

try:
    from vr_modality_bias.data.prompts import get_prompt
    from vr_modality_bias.experiment.sparc import SparcHyperparams, enable_sparc
    from vr_modality_bias.models.registry import build_model
    from vr_modality_bias.utils.config import load_config
    from vr_modality_bias.utils.device import resolve_dtype, select_device
    from vr_modality_bias.utils.seeds import derive_image_seed
except ModuleNotFoundError:
    sys.path.insert(0, str(here()))

    from src.vr_modality_bias.data.prompts import get_prompt
    from src.vr_modality_bias.experiment.sparc import SparcHyperparams, enable_sparc
    from src.vr_modality_bias.models.registry import build_model
    from src.vr_modality_bias.utils.config import load_config
    from src.vr_modality_bias.utils.device import resolve_dtype, select_device
    from src.vr_modality_bias.utils.seeds import derive_image_seed


# Map length name → length-specific config file. The configs share the
# model (Qwen2.5-VL-7B / bf16) — only prompt and max_new_tokens differ.
LENGTH_CONFIGS = {
    "short":  "configs/run_qwen7b_short.yaml",
    "medium": "configs/run_qwen7b_medium.yaml",
    "long":   "configs/run_qwen7b_long.yaml",
}


def _iso_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def _read_done(jsonl_path: Path) -> set[tuple[str, str, str]]:
    """Read captions.jsonl → set of (image_id, length, condition) keys done."""
    done: set[tuple[str, str, str]] = set()
    if not jsonl_path.exists():
        return done
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((entry["image_id"], entry["length"], entry["condition"]))
    return done


def _append(jsonl_path: Path, entry: dict) -> None:
    """Append a JSON line atomically (line is a single write)."""
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _probe_input_len(model_wrapper, image, prompt: str) -> int:
    """Compute SPARC ``input_len`` (= prompt length excluding image patches).

    Same logic as collect_forced_decoding and the fluency section of the
    Phase 2 report: tokenise the prefix, count <|image_pad|> tokens,
    subtract.
    """
    processor = model_wrapper._processor  # noqa: SLF001
    messages = model_wrapper._build_messages(prompt, image)
    prefix_text = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )
    prefix_inputs = processor(text=[prefix_text], images=[image], return_tensors="pt")
    caption_start = int(prefix_inputs["input_ids"].shape[-1])
    image_token_id = int(model_wrapper._model.config.image_token_id)  # noqa: SLF001
    image_positions = (
        prefix_inputs["input_ids"][0] == image_token_id
    ).nonzero(as_tuple=True)[0]
    num_image_patches = int(image_positions.numel())
    return caption_start - num_image_patches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", type=str, default="phase3",
        help="Output goes to results/runs/<run-name>/.")
    parser.add_argument("--output-root", type=Path, default=Path("results/runs"),
        help="Parent dir for run directories (default: results/runs).")
    parser.add_argument("--limit", type=int, default=50,
        help="Max images per length (default 50).")
    parser.add_argument("--lengths", nargs="+",
        choices=list(LENGTH_CONFIGS.keys()),
        default=list(LENGTH_CONFIGS.keys()),
        help="Which lengths to generate (default: all three).")
    parser.add_argument("--alpha", type=float, default=1.1,
        help="SPARC α for the ON condition. Default 1.1 (the only safe regime per Phase 2).")
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--selected-layer", type=int, default=15)
    parser.add_argument("--se-layers", type=int, nargs=2, default=(0, 31))
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true",
        help="Delete an existing captions.jsonl before starting (re-runs everything).")
    parser.add_argument("--smoke", action="store_true",
        help="Smoke: --limit 1, --lengths short. Confirms entrypoint + IO + log path.")
    args = parser.parse_args()

    if args.smoke:
        args.limit = 1
        args.lengths = ["short"]

    run_dir = args.output_root / args.run_name
    log_file = run_dir / "logs" / "phase3.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    # File sink in addition to stderr. ``enqueue=True`` makes the writer
    # thread-safe and resilient to signal interrupts (tmux detach, etc.).
    logger.add(str(log_file), enqueue=True, level="INFO")

    logger.info("=" * 70)
    logger.info(f"Phase 3 generation — run_name={args.run_name}")
    logger.info(f"lengths={args.lengths}  alpha={args.alpha}  limit={args.limit}  "
                f"overwrite={args.overwrite}  smoke={args.smoke}")
    logger.info(f"run dir : {run_dir}")
    logger.info(f"log file: {log_file}")
    logger.info("=" * 70)

    # Snapshot args for reproducibility.
    snapshot = {
        "run_name": args.run_name,
        "lengths": args.lengths,
        "alpha": args.alpha,
        "tau": args.tau,
        "selected_layer": args.selected_layer,
        "se_layers": list(args.se_layers),
        "beta": args.beta,
        "limit": args.limit,
        "overwrite": args.overwrite,
        "smoke": args.smoke,
        "timestamp_iso": _iso_now(),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_params.json").write_text(
        json.dumps(snapshot, indent=2) + "\n", encoding="utf-8",
    )

    jsonl_path = run_dir / "captions.jsonl"
    if args.overwrite and jsonl_path.exists():
        logger.info(f"--overwrite: removing existing {jsonl_path}")
        jsonl_path.unlink()
    done = _read_done(jsonl_path)
    logger.info(f"Resume state: {len(done)} cells already in {jsonl_path}")

    # Load model once from the first length config (they share the model spec).
    first_cfg_path = Path(LENGTH_CONFIGS[args.lengths[0]])
    cfg_first = load_config(first_cfg_path)
    model_key = str(cfg_first["model"]["key"])
    model_id = str(cfg_first["model"]["model_id"])
    dtype_str = str(cfg_first["model"]["dtype"])

    model_wrapper = build_model(model_key)
    model_wrapper.model_id = model_id
    dtype = resolve_dtype(dtype_str)
    device = select_device("cuda")
    if hasattr(model_wrapper, "_dtype"):
        model_wrapper._dtype = dtype  # noqa: SLF001
    logger.info(f"Loading {model_id} ({dtype_str}) on {device}...")
    model_wrapper.load(device)
    logger.info(f"Model loaded. n_layers={model_wrapper.n_layers}")

    # Plan: count planned cells (informational; we still skip-on-disk live).
    planned_per_length = 2 * args.limit  # off + on per image
    total_planned = planned_per_length * len(args.lengths)
    logger.info(f"Planned cells: {total_planned} (per length: {planned_per_length})")

    sparc_hparams = SparcHyperparams(
        alpha=args.alpha, tau=args.tau,
        selected_layer=args.selected_layer,
        se_layers=tuple(args.se_layers),
        beta=args.beta,
    )

    cells_done = 0
    cells_skipped = 0
    cells_failed = 0
    t_start = time.time()

    for length in args.lengths:
        cfg = load_config(LENGTH_CONFIGS[length])
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

        images_dir = cfg["dataset"]["images_dir"]
        image_files = sorted(glob.glob(f"{images_dir}{os.sep}*.jpg"))[: args.limit]
        if not image_files:
            logger.error(f"[{length}] no images under {images_dir}; skipping length.")
            continue
        logger.info(
            f"[{length}] {len(image_files)} image(s), prompt_key={prompt_key}, "
            f"max_new_tokens={max_new_tokens}"
        )

        for image_path in image_files:
            image_id = Path(image_path).stem
            with Image.open(image_path) as raw:
                image = raw.convert("RGB")
            # Same seed used for OFF and ON so the pair is directly comparable.
            seed = int(derive_image_seed(seed_global, image_id))

            # ---------------- OFF (baseline) ----------------
            key_off = (image_id, length, "off")
            if key_off in done:
                cells_skipped += 1
            else:
                t_cell = time.time()
                try:
                    caption = model_wrapper.generate_caption(
                        image=image, prompt=prompt,
                        max_new_tokens=max_new_tokens,
                        seed=seed,
                        generation_kwargs=gen_kwargs,
                    )
                    entry = {
                        "image_id": image_id,
                        "length": length,
                        "condition": "off",
                        "alpha": None,
                        "caption": caption,
                        "seed": seed,
                        "prompt_key": prompt_key,
                        "model_id": model_id,
                        "dtype": dtype_str,
                        "max_new_tokens": max_new_tokens,
                        "timestamp_iso": _iso_now(),
                    }
                    _append(jsonl_path, entry)
                    done.add(key_off)
                    cells_done += 1
                    dt = time.time() - t_cell
                    n_words = len(caption.split())
                    rate = cells_done / max(time.time() - t_start, 0.1)
                    remaining = max(total_planned - cells_done - cells_skipped, 0)
                    eta_min = remaining / max(rate, 1e-6) / 60
                    logger.info(
                        f"[{length}|{image_id}|off] OK  words={n_words}  "
                        f"({dt:.1f}s)  progress {cells_done + cells_skipped}/{total_planned}  "
                        f"ETA {eta_min:.1f}min"
                    )
                except Exception as exc:
                    cells_failed += 1
                    logger.error(f"[{length}|{image_id}|off] FAILED: {exc}")
                    logger.error(traceback.format_exc())

            # ---------------- ON (SPARC α) ----------------
            key_on = (image_id, length, "on")
            if key_on in done:
                cells_skipped += 1
                continue
            t_cell = time.time()
            try:
                input_len = _probe_input_len(model_wrapper, image, prompt)
                with enable_sparc(
                    model_wrapper, hparams=sparc_hparams,
                    probe_image=image, prompt=prompt,
                ) as buffer:
                    buffer.reset()
                    buffer.update_input_len(input_len)
                    caption = model_wrapper.generate_caption(
                        image=image, prompt=prompt,
                        max_new_tokens=max_new_tokens,
                        seed=seed,
                        generation_kwargs=gen_kwargs,
                    )
                entry = {
                    "image_id": image_id,
                    "length": length,
                    "condition": "on",
                    "alpha": float(args.alpha),
                    "tau": float(args.tau),
                    "selected_layer": int(args.selected_layer),
                    "se_layers": list(args.se_layers),
                    "beta": float(args.beta),
                    "caption": caption,
                    "seed": seed,
                    "prompt_key": prompt_key,
                    "model_id": model_id,
                    "dtype": dtype_str,
                    "max_new_tokens": max_new_tokens,
                    "timestamp_iso": _iso_now(),
                }
                _append(jsonl_path, entry)
                done.add(key_on)
                cells_done += 1
                dt = time.time() - t_cell
                n_words = len(caption.split())
                rate = cells_done / max(time.time() - t_start, 0.1)
                remaining = max(total_planned - cells_done - cells_skipped, 0)
                eta_min = remaining / max(rate, 1e-6) / 60
                logger.info(
                    f"[{length}|{image_id}|on α={args.alpha:.1f}] OK  words={n_words}  "
                    f"({dt:.1f}s)  progress {cells_done + cells_skipped}/{total_planned}  "
                    f"ETA {eta_min:.1f}min"
                )
            except Exception as exc:
                cells_failed += 1
                logger.error(f"[{length}|{image_id}|on α={args.alpha:.1f}] FAILED: {exc}")
                logger.error(traceback.format_exc())

    elapsed_min = (time.time() - t_start) / 60
    logger.info("=" * 70)
    logger.info(
        f"Phase 3 generation DONE.  cells_done={cells_done}  "
        f"skipped={cells_skipped}  failed={cells_failed}  "
        f"elapsed={elapsed_min:.1f}min"
    )
    logger.info(f"Output dir: {run_dir}")
    logger.info(f"Captions  : {jsonl_path}")
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
