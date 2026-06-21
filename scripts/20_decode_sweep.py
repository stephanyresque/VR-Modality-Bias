#!/usr/bin/env python
"""Decode-sweep tuner: pick repetition_penalty × max_new_tokens for SPARC."""

from __future__ import annotations

import argparse
import re
import sys
import time
import traceback
from pathlib import Path

from loguru import logger
from PIL import Image
from pyprojroot import here

try:
    from vr_modality_bias.data.captions import detect_loop
    from vr_modality_bias.data.prompts import get_prompt
    from vr_modality_bias.experiment.sparc import SparcHyperparams, enable_sparc
    from vr_modality_bias.models.registry import build_model
    from vr_modality_bias.utils.config import load_config
    from vr_modality_bias.utils.device import resolve_dtype, select_device
    from vr_modality_bias.utils.seeds import derive_image_seed
except ModuleNotFoundError:
    sys.path.insert(0, str(here()))

    from src.vr_modality_bias.data.captions import detect_loop
    from src.vr_modality_bias.data.prompts import get_prompt
    from src.vr_modality_bias.experiment.sparc import SparcHyperparams, enable_sparc
    from src.vr_modality_bias.models.registry import build_model
    from src.vr_modality_bias.utils.config import load_config
    from src.vr_modality_bias.utils.device import resolve_dtype, select_device
    from src.vr_modality_bias.utils.seeds import derive_image_seed


# ---------------------------------------------------------------- helpers
# detect_loop lives in src/vr_modality_bias/data/captions.py (Block 6
# moved it there so the orchestrator can import without going through a
# script). It's imported above in the try/except.


def _word_count(text: str) -> int:
    # Same definition the Phase 2 / Phase 3 reports use.
    return len(re.findall(r"\S+", text))


def _print_caption(label: str, image_id: str, rp: float, max_tok: int, text: str) -> None:
    bar = "─" * 78
    print(bar)
    print(f"  [long|{image_id}|{label}]  rp={rp}  max_tok={max_tok}  words={_word_count(text)}")
    loop_flag, why = detect_loop(text)
    if loop_flag:
        print(f"  LOOP DETECTED: {why}")
    print(bar)
    print(text)
    print()


# ---------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--length-config-pattern", type=str,
        default="configs/run_smolvlm22_{length}.yaml",
        help="Defaults to SmolVLM-2.2B. Pass the Qwen pattern to repurpose.",
    )
    parser.add_argument(
        "--length", choices=("short", "medium", "long"), default="long",
        help="Long is the only one that exposes the SPARC tail-degeneration "
             "we're tuning for — keep this default unless diagnosing something else.",
    )
    parser.add_argument(
        "--image-ids", type=str, nargs="+",
        default=["000000000139", "000000000285", "000000000632"],
        help="Image stems (no extension). Defaults: 3 longs that exposed the "
             "tail-loop in Phase 4 smoke.",
    )
    parser.add_argument(
        "--rps", type=float, nargs="+", default=[1.0, 1.1, 1.15],
        help="repetition_penalty grid. 1.0 = no penalty (the official SPARC "
             "COCO setup); 1.1/1.15 add light/medium tail damping.",
    )
    parser.add_argument(
        "--max-toks", type=int, nargs="+", default=[512, 256],
        help="max_new_tokens grid. 512 is the official long; 256 is the "
             "fallback if no rp stabilises 512.",
    )
    # SPARC hparams — frozen at official COCO values; expose only so a
    # mistake is loud (override deliberately, never by accident).
    parser.add_argument("--alpha", type=float, default=1.1)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=1.5)
    parser.add_argument("--selected-layer", type=int, default=15)
    parser.add_argument("--se-layers", type=int, nargs=2, default=(0, 31))
    args = parser.parse_args()

    # ---- 1. resolve config ---------------------------------------
    cfg_path = Path(args.length_config_pattern.format(length=args.length))
    cfg = load_config(cfg_path)
    model_key = str(cfg["model"]["key"])
    model_id = str(cfg["model"]["model_id"])
    dtype_str = str(cfg["model"]["dtype"])
    prompt_key = str(cfg["task"]["prompt_key"])
    prompt = get_prompt(prompt_key)
    seed_global = int(cfg["run"]["seed_global"])

    images_dir = Path(cfg["dataset"]["images_dir"])

    print("=" * 78)
    print(f"DECODE SWEEP — {model_id} ({dtype_str})")
    print(f"length={args.length}  prompt_key={prompt_key}")
    print(f"images={args.image_ids}")
    print(f"rp grid={args.rps}  max_tok grid={args.max_toks}")
    print(
        f"SPARC: alpha={args.alpha} beta={args.beta} tau={args.tau} "
        f"selected_layer={args.selected_layer} se_layers={tuple(args.se_layers)}"
    )
    print("decoding: GREEDY (do_sample=False, num_beams=1)")
    print("=" * 78)

    # ---- 2. load model once --------------------------------------
    device = select_device("cuda")
    dtype = resolve_dtype(dtype_str)
    model_wrapper = build_model(model_key)
    model_wrapper.model_id = model_id
    if hasattr(model_wrapper, "_dtype"):
        model_wrapper._dtype = dtype  # noqa: SLF001
    logger.info(f"Loading {model_id} ({dtype_str}) on {device}...")
    model_wrapper.load(device)
    logger.info(f"Model loaded. n_layers={model_wrapper.n_layers}")

    sparc_hparams = SparcHyperparams(
        alpha=args.alpha,
        beta=args.beta,
        tau=args.tau,
        selected_layer=args.selected_layer,
        se_layers=tuple(args.se_layers),
    )

    # ---- 3. sweep -------------------------------------------------
    # Per-image SPARC bookkeeping helper (mirrors what script 18 does;
    # the per-id mask is set per image after the buffer is constructed).
    def _probe_layout(image):
        proc = model_wrapper._processor  # noqa: SLF001
        msgs = model_wrapper._build_messages(prompt, image)
        prefix_text = proc.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False,
        )
        prefix_inputs = proc(text=[prefix_text], images=[image], return_tensors="pt")
        caption_start = int(prefix_inputs["input_ids"].shape[-1])
        image_token_id = int(model_wrapper._model.config.image_token_id)  # noqa: SLF001
        positions = (
            prefix_inputs["input_ids"][0] == image_token_id
        ).nonzero(as_tuple=True)[0]
        n_patches = int(positions.numel())
        return caption_start - n_patches, positions

    # Table rows: (image_id, rp, max_tok, off_words, off_loop, on_words, on_loop).
    rows: list[tuple] = []

    for image_id in args.image_ids:
        image_path = images_dir / f"{image_id}.jpg"
        if not image_path.exists():
            logger.error(f"missing {image_path}; skipping")
            continue
        image = Image.open(image_path).convert("RGB")
        # Deterministic seed (same definition as scripts/18); greedy
        # doesn't actually consume the seed, but we keep it for trace.
        seed = int(derive_image_seed(seed_global, image_id))

        for max_tok in args.max_toks:
            for rp in args.rps:
                gen_kwargs = {
                    "do_sample": False,
                    "num_beams": 1,
                    "repetition_penalty": float(rp),
                }
                cell_id = f"{image_id} | rp={rp} | max_tok={max_tok}"
                logger.info(f"--- {cell_id} ---")

                # ---- OFF ----
                try:
                    t0 = time.time()
                    off = model_wrapper.generate_caption(
                        image=image, prompt=prompt,
                        max_new_tokens=int(max_tok),
                        seed=seed,
                        generation_kwargs=gen_kwargs,
                    )
                    off_dt = time.time() - t0
                    _print_caption("OFF", image_id, rp, max_tok, off)
                except Exception as exc:
                    logger.error(f"OFF generation failed: {exc}")
                    logger.error(traceback.format_exc())
                    off, off_dt = "<FAILED>", 0.0

                # ---- ON ----
                try:
                    input_len, image_positions = _probe_layout(image)
                    t0 = time.time()
                    with enable_sparc(
                        model_wrapper,
                        hparams=sparc_hparams,
                        probe_image=image,
                        prompt=prompt,
                    ) as buffer:
                        buffer.reset()
                        buffer.update_input_len(input_len)
                        buffer.update_image_positions(image_positions)
                        on = model_wrapper.generate_caption(
                            image=image, prompt=prompt,
                            max_new_tokens=int(max_tok),
                            seed=seed,
                            generation_kwargs=gen_kwargs,
                        )
                    on_dt = time.time() - t0
                    _print_caption("ON  α=1.1", image_id, rp, max_tok, on)
                except Exception as exc:
                    logger.error(f"ON generation failed: {exc}")
                    logger.error(traceback.format_exc())
                    on, on_dt = "<FAILED>", 0.0

                off_loop, _ = detect_loop(off) if off != "<FAILED>" else (False, "")
                on_loop, _ = detect_loop(on) if on != "<FAILED>" else (False, "")
                rows.append((
                    image_id, rp, max_tok,
                    _word_count(off), "Y" if off_loop else ".",
                    _word_count(on),  "Y" if on_loop  else ".",
                ))

    # ---- 4. summary table ---------------------------------------
    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    # Format: keep it narrow so it fits in any terminal.
    headers = ("image_id", "rp", "max_tok", "off_words", "off_loop", "on_words", "on_loop")
    widths = [max(len(h), 12) for h in headers]
    widths[0] = 14  # image_id column wider
    fmt = "  ".join(f"%{w}s" for w in widths)
    print(fmt % headers)
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print(fmt % (r[0], f"{r[1]}", f"{r[2]}", f"{r[3]}", r[4], f"{r[5]}", r[6]))

    print()
    print(
        "loop heuristic: Y iff any non-stopword unigram occupies ≥4 of the last "
        "20 tokens, OR any token repeats ≥3 times consecutively in the last 30."
    )
    print(
        "decision is the user's: pick the lightest (rp, max_tok) where on_loop is "
        "'.' for all 3 images AND off_words isn't anomalously truncated."
    )
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
