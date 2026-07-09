#!/usr/bin/env python
"""Phase 3 — free caption generation, baseline (SPARC OFF) vs SPARC ON with the
official COCO recipe (α=1.1, β=0.1, τ=1.5, layer 20, greedy — sampling is what
destabilised SPARC on long captions, not the implementation).

Run: make phase3  (smoke: make phase3-smoke; coherence check: make phase3-coherence)
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
# model (Qwen2.5-VL-7B / bf16 OR SmolVLM-2.2B) — only prompt and
# max_new_tokens differ. Use ``--length-config-pattern`` to swap families
# (e.g. ``configs/run_smolvlm22_{length}.yaml`` for the SmolVLM smoke).
LENGTH_CONFIGS = {
    "short":  "configs/run_qwen7b_short.yaml",
    "medium": "configs/run_qwen7b_medium.yaml",
    "long":   "configs/run_qwen7b_long.yaml",
}


def _resolve_length_configs(pattern: str | None) -> dict[str, str]:
    """Build the {length: cfg_path} dict from a pattern with ``{length}``."""
    if pattern is None:
        return dict(LENGTH_CONFIGS)
    if "{length}" not in pattern:
        raise ValueError(
            f"--length-config-pattern must contain '{{length}}', got {pattern!r}"
        )
    return {name: pattern.format(length=name) for name in LENGTH_CONFIGS}


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


def _probe_sparc_layout(model_wrapper, image, prompt):
    """Return ``(input_len, image_positions)`` for one image's prefill.

    * ``input_len`` is the prompt length excluding image-placeholder tokens
      (the value SPARC's ``update_input_len`` expects).
    * ``image_positions`` is the per-id mask — a 1-D LongTensor of global
      positions in the prefill input_ids where the token equals
      ``model.config.image_token_id``. SPARC's ``update_image_positions``
      consumes this. For Qwen the mask IS contiguous; for SmolVLM/Idefics3
      it skips over <fake_token_around_image> / <row_X_col_Y> separators
      that the contiguous-block assumption would otherwise miscalibrate.
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
    return caption_start - num_image_patches, image_positions


def build_parser() -> argparse.ArgumentParser:
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
    # SPARC defaults match the OFFICIAL SPARC paper's COCO recipe
    # (captioning_coco.sh). Earlier defaults (β=0, τ=2, selected_layer=15)
    # were not what the authors validated and were partly responsible for
    # the long-caption degeneration. Do NOT override these unless you're
    # intentionally probing the SPARC hparams.
    parser.add_argument("--alpha", type=float, default=1.1,
        help="SPARC α. Official COCO value: 1.1.")
    parser.add_argument("--tau", type=float, default=1.5,
        help="SPARC τ. Official COCO value: 1.5.")
    parser.add_argument("--selected-layer", type=int, default=20,
        help="SPARC selected_layer. Official COCO value: 20.")
    parser.add_argument("--se-layers", type=int, nargs=2, default=(0, 31),
        help="SPARC se_layers (lo hi). Official COCO value: (0, 31).")
    parser.add_argument("--beta", type=float, default=0.1,
        help="SPARC β. Official COCO value: 0.1 — smooths the reference "
             "attention used for selection; β=0 makes selection erratic.")
    parser.add_argument("--sampling", action="store_true",
        help="Override greedy decoding with the sampling params in the "
             "length config. Default: greedy (do_sample=False, num_beams=1) "
             "— matches the official SPARC COCO setup and is REQUIRED for "
             "SPARC stability on long captions.")
    parser.add_argument("--repetition-penalty", type=float, default=None,
        help="Override repetition_penalty in gen_kwargs (default: 1.0 in "
             "greedy mode). Use ~1.1-1.3 to test whether Qwen-7B greedy "
             "repetition is what's bootstrapping SPARC's degeneration.")
    parser.add_argument("--no-repeat-ngram-size", type=int, default=None,
        help="If set, pass no_repeat_ngram_size to generate() — bans any "
             "n-gram from appearing twice. Hard guard against 'the room. "
             "the room.' style loops in greedy.")
    parser.add_argument("--image-ids", type=str, nargs="+", default=None,
        help="Specific COCO image IDs (zero-padded 12-digit stems) to use, "
             "in order. Overrides the auto-pick of the first --limit images. "
             "Used by the isolation tests in the SPARC degeneration audit.")
    parser.add_argument("--length-config-pattern", type=str, default=None,
        help="Pattern with '{length}' placeholder for length-specific configs. "
             "Default: configs/run_qwen7b_{length}.yaml. For the SmolVLM smoke, "
             "use 'configs/run_smolvlm22_{length}.yaml'.")
    parser.add_argument("--print-captions", action="store_true",
        help="Print each generated caption to stdout (in addition to the log). "
             "Useful for eyeball coherence checks.")
    parser.add_argument("--overwrite", action="store_true",
        help="Delete an existing captions.jsonl before starting (re-runs everything).")
    parser.add_argument("--smoke", action="store_true",
        help="Smoke: --limit 1, --lengths short. Confirms entrypoint + IO + log path. "
             "Implies --print-captions.")
    parser.add_argument("--coherence-smoke", action="store_true",
        help="Coherence smoke: --limit 2, --lengths long, --print-captions. "
             "Used to verify SPARC produces fluent text on long generation "
             "after the official-config switch.")
    # Adaptive intensity (Ponto 1). Absent => the original α^c reinforcement,
    # so every existing invocation keeps its behaviour untouched.
    parser.add_argument("--adaptive", action="store_true",
        help="Replace the accumulating α^c reinforcement with the deficit-driven "
             "target factor capped by --ceiling. α is unused in this mode.")
    parser.add_argument("--lam", type=float, default=0.0,
        help="SPARC λ, the deficit sensitivity of --adaptive. λ=0 is the "
             "neutrality gate (no intervention at all).")
    parser.add_argument("--ceiling", type=float, default=2.0,
        help="SPARC saturation ceiling for --adaptive. The effective factor of "
             "any visual token never exceeds it.")
    return parser


def sparc_hparams_from_args(args) -> SparcHyperparams:
    """Build the SPARC hyperparameters from a parsed argparse namespace."""
    return SparcHyperparams(
        alpha=args.alpha,
        tau=args.tau,
        selected_layer=args.selected_layer,
        se_layers=tuple(args.se_layers),
        beta=args.beta,
        adaptive=args.adaptive,
        lam=args.lam,
        ceiling=args.ceiling,
    )


def main() -> int:
    args = build_parser().parse_args()

    if args.smoke:
        args.limit = 1
        args.lengths = ["short"]
        args.print_captions = True
    if args.coherence_smoke:
        args.limit = 2
        args.lengths = ["long"]
        args.print_captions = True

    # Built before the model load so an invalid combination (α<=1 without
    # --adaptive, negative λ, ceiling<=1) fails in a second, not after the
    # checkpoint is on the GPU.
    sparc_hparams = sparc_hparams_from_args(args)

    run_dir = args.output_root / args.run_name
    log_file = run_dir / "logs" / "phase3.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    # File sink in addition to stderr. ``enqueue=True`` makes the writer
    # thread-safe and resilient to signal interrupts (tmux detach, etc.).
    logger.add(str(log_file), enqueue=True, level="INFO")

    logger.info("=" * 70)
    logger.info(f"Phase 3 generation — run_name={args.run_name}")
    logger.info(f"lengths={args.lengths}  limit={args.limit}  "
                f"overwrite={args.overwrite}  smoke={args.smoke}  "
                f"coherence_smoke={args.coherence_smoke}")
    logger.info(
        f"SPARC official-COCO config: alpha={args.alpha} beta={args.beta} "
        f"tau={args.tau} selected_layer={args.selected_layer} "
        f"se_layers={tuple(args.se_layers)}"
    )
    logger.info(
        f"SPARC intensity: {'ADAPTIVE' if args.adaptive else 'ORIGINAL α^c'} "
        f"lam={args.lam} ceiling={args.ceiling}"
    )
    logger.info(
        f"decoding: {'GREEDY (do_sample=False, num_beams=1)' if not args.sampling else 'SAMPLING (from config)'}"
    )
    logger.info(f"run dir : {run_dir}")
    logger.info(f"log file: {log_file}")
    logger.info("=" * 70)

    # Snapshot args for reproducibility. The SPARC block comes straight from
    # ``as_dict`` so any hyperparameter added there lands here automatically.
    snapshot = {
        "run_name": args.run_name,
        "lengths": args.lengths,
        **sparc_hparams.as_dict(),
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

    length_configs = _resolve_length_configs(args.length_config_pattern)
    logger.info(f"length_configs: {length_configs}")

    # Load model once from the first length config (they share the model spec).
    first_cfg_path = Path(length_configs[args.lengths[0]])
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

    cells_done = 0
    cells_skipped = 0
    cells_failed = 0
    t_start = time.time()

    for length in args.lengths:
        cfg = load_config(length_configs[length])
        prompt_key = str(cfg["task"]["prompt_key"])
        prompt = get_prompt(prompt_key)
        seed_global = int(cfg["run"]["seed_global"])
        max_new_tokens = int(cfg["generation"]["max_new_tokens"])
        # Greedy by default (Phase 3 spec): matches the official SPARC COCO
        # config and is required for SPARC stability on long captions.
        # Use --sampling to fall back to the config's sampling params (for
        # diagnostic re-runs only — Phase 3 results must be greedy).
        if args.sampling:
            gen_kwargs = {
                "do_sample": bool(cfg["generation"]["do_sample"]),
                "temperature": float(cfg["generation"]["temperature"]),
                "top_p": float(cfg["generation"]["top_p"]),
                "repetition_penalty": float(cfg["generation"]["repetition_penalty"]),
            }
        else:
            gen_kwargs = {
                "do_sample": False,
                "num_beams": 1,
                "repetition_penalty": 1.0,
            }
        # CLI overrides (used by the SPARC degeneration audit experiments).
        if args.repetition_penalty is not None:
            gen_kwargs["repetition_penalty"] = float(args.repetition_penalty)
        if args.no_repeat_ngram_size is not None:
            gen_kwargs["no_repeat_ngram_size"] = int(args.no_repeat_ngram_size)

        images_dir = cfg["dataset"]["images_dir"]
        if args.image_ids:
            # Explicit IDs override the auto-pick (used by the audit experiments).
            image_files = [str(Path(images_dir) / f"{i}.jpg") for i in args.image_ids]
            missing = [p for p in image_files if not Path(p).exists()]
            if missing:
                logger.error(f"[{length}] missing image files: {missing}")
                continue
        else:
            image_files = sorted(glob.glob(f"{images_dir}{os.sep}*.jpg"))[: args.limit]
        if not image_files:
            logger.error(f"[{length}] no images under {images_dir}; skipping length.")
            continue
        logger.info(
            f"[{length}] {len(image_files)} image(s), prompt_key={prompt_key}, "
            f"max_new_tokens={max_new_tokens}, gen_kwargs={gen_kwargs}"
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
                    if args.print_captions:
                        print(f"\n── [{length}|{image_id}|OFF] ─────────")
                        print(caption)
                        print()
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
                input_len, image_positions = _probe_sparc_layout(
                    model_wrapper, image, prompt,
                )
                with enable_sparc(
                    model_wrapper, hparams=sparc_hparams,
                    probe_image=image, prompt=prompt,
                ) as buffer:
                    buffer.reset()
                    buffer.update_input_len(input_len)
                    # Per-id mask of <image>-token positions for THIS image.
                    # Without this, SPARC falls back to the legacy contiguous
                    # slice — fine for Qwen, broken for Idefics3/SmolVLM where
                    # separator tokens sit inside that range.
                    buffer.update_image_positions(image_positions)
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
                if args.print_captions:
                    print(f"\n── [{length}|{image_id}|ON α={args.alpha:.1f}] ─────────")
                    print(caption)
                    print()
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
