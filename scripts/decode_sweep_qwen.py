#!/usr/bin/env python
"""Decode-stabilization sweep for Qwen2.5-VL-7B: SPARC fixed at the official
COCO config, sweeping (repetition_penalty × no_repeat_ngram_size) with the same
decoding applied to OFF and ON so the comparison stays fair.

Run: python scripts/decode_sweep_qwen.py --image-ids ID [ID ...] [--rep-penalties ...] [--no-repeat-ngrams ...]
"""

from __future__ import annotations

import argparse
import sys
import time
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


DEFAULT_LENGTH_CONFIG = "configs/run_qwen7b_long.yaml"

# Heuristic for "the caption has a repetition loop". Used only for the
# summary table — the actual judgement is yours after eyeballing the text.
LOOP_TRIGGER_WORDS = 4   # 4× the same word in a row
LOOP_TRIGGER_BIGRAM = 3  # same 2-gram appearing 3+ times


def _probe_input_len(model_wrapper, image, prompt: str) -> int:
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


def _detect_loop(caption: str) -> tuple[bool, str]:
    """Lightweight loop detector for the summary table. NOT the gate decision."""
    words = caption.split()
    if not words:
        return (True, "empty")
    # Word-level repetition.
    cur = 1
    for i in range(1, len(words)):
        if words[i].lower() == words[i - 1].lower():
            cur += 1
            if cur >= LOOP_TRIGGER_WORDS:
                return (True, f"word×{cur}: '{words[i].lower()}'")
        else:
            cur = 1
    # Bigram repetition.
    if len(words) >= 6:
        bigrams = [(words[i].lower(), words[i + 1].lower())
                   for i in range(len(words) - 1)]
        counts: dict[tuple, int] = {}
        for bg in bigrams:
            counts[bg] = counts.get(bg, 0) + 1
        worst = max(counts.items(), key=lambda kv: kv[1])
        if worst[1] >= LOOP_TRIGGER_BIGRAM:
            return (True, f"bigram×{worst[1]}: '{worst[0][0]} {worst[0][1]}'")
    return (False, "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-ids", type=str, nargs="+", required=True,
        help="COCO image IDs (zero-padded 12-digit) to probe.")
    parser.add_argument("--length-config", type=Path, default=Path(DEFAULT_LENGTH_CONFIG),
        help="Length-specific config to pull prompt + max_new_tokens + model from.")
    parser.add_argument("--rep-penalties", type=float, nargs="+",
        default=[1.0, 1.1, 1.15, 1.2],
        help="Repetition_penalty values to sweep.")
    parser.add_argument("--no-repeat-ngrams", type=int, nargs="+",
        default=[0, 3],
        help="no_repeat_ngram_size values to sweep (0 means disabled).")
    # SPARC official COCO config — DO NOT change without an explicit reason.
    parser.add_argument("--alpha", type=float, default=1.1)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=1.5)
    parser.add_argument("--selected-layer", type=int, default=20)
    parser.add_argument("--se-layers", type=int, nargs=2, default=(0, 31))
    args = parser.parse_args()

    cfg = load_config(args.length_config)
    prompt_key = str(cfg["task"]["prompt_key"])
    prompt = get_prompt(prompt_key)
    seed_global = int(cfg["run"]["seed_global"])
    max_new_tokens = int(cfg["generation"]["max_new_tokens"])

    model_wrapper = build_model(str(cfg["model"]["key"]))
    model_wrapper.model_id = str(cfg["model"]["model_id"])
    dtype = resolve_dtype(str(cfg["model"]["dtype"]))
    device = select_device("cuda")
    if hasattr(model_wrapper, "_dtype"):
        model_wrapper._dtype = dtype  # noqa: SLF001
    logger.info(f"Loading {model_wrapper.model_id} ({dtype})...")
    model_wrapper.load(device)
    logger.info(f"Loaded. n_layers={model_wrapper.n_layers}")

    sparc_hparams = SparcHyperparams(
        alpha=args.alpha, beta=args.beta, tau=args.tau,
        selected_layer=args.selected_layer,
        se_layers=tuple(args.se_layers),
    )

    images_dir = cfg["dataset"]["images_dir"]
    image_paths = [Path(images_dir) / f"{i}.jpg" for i in args.image_ids]
    for p in image_paths:
        if not p.exists():
            logger.error(f"missing image: {p}")
            return 1

    print()
    print("=" * 78)
    print("ETAPA 1 — DECODE SWEEP")
    print("=" * 78)
    print(f"  length_config   : {args.length_config}")
    print(f"  prompt_key      : {prompt_key}")
    print(f"  max_new_tokens  : {max_new_tokens}")
    print(f"  SPARC           : α={args.alpha} β={args.beta} τ={args.tau} "
          f"sel={args.selected_layer} se={tuple(args.se_layers)}")
    print(f"  decoding        : greedy (do_sample=False, num_beams=1)")
    print(f"  rep_penalties   : {args.rep_penalties}")
    print(f"  no_repeat_ngrams: {args.no_repeat_ngrams}")
    print(f"  images          : {args.image_ids}")
    print()

    summary_rows: list[dict] = []  # for the final table
    for image_id, image_path in zip(args.image_ids, image_paths):
        with Image.open(image_path) as raw:
            image = raw.convert("RGB")
        seed = int(derive_image_seed(seed_global, image_id))
        input_len = _probe_input_len(model_wrapper, image, prompt)

        print("=" * 78)
        print(f"IMAGE {image_id}  (seed={seed}, input_len={input_len})")
        print("=" * 78)

        for rp in args.rep_penalties:
            for ng in args.no_repeat_ngrams:
                gen_kwargs = {
                    "do_sample": False,
                    "num_beams": 1,
                    "repetition_penalty": float(rp),
                }
                if ng > 0:
                    gen_kwargs["no_repeat_ngram_size"] = int(ng)

                tag = f"rp={rp:.2f}  ng={ng}"
                print(f"\n── [{image_id} | {tag}] ────────────────────────────")

                # OFF
                t0 = time.time()
                off_caption = model_wrapper.generate_caption(
                    image=image, prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    seed=seed, generation_kwargs=gen_kwargs,
                )
                off_t = time.time() - t0
                off_words = len(off_caption.split())
                off_loop, off_loop_why = _detect_loop(off_caption)
                print(f"OFF [{off_words:3d} words  {off_t:.1f}s] "
                      f"{'⚠ LOOP: ' + off_loop_why if off_loop else 'ok'}")
                print(f"  {off_caption}")

                # ON
                t0 = time.time()
                with enable_sparc(
                    model_wrapper, hparams=sparc_hparams,
                    probe_image=image, prompt=prompt,
                ) as buffer:
                    buffer.reset()
                    buffer.update_input_len(input_len)
                    on_caption = model_wrapper.generate_caption(
                        image=image, prompt=prompt,
                        max_new_tokens=max_new_tokens,
                        seed=seed, generation_kwargs=gen_kwargs,
                    )
                on_t = time.time() - t0
                on_words = len(on_caption.split())
                on_loop, on_loop_why = _detect_loop(on_caption)
                print(f"ON  [{on_words:3d} words  {on_t:.1f}s] "
                      f"{'⚠ LOOP: ' + on_loop_why if on_loop else 'ok'}")
                print(f"  {on_caption}")

                summary_rows.append({
                    "image_id": image_id,
                    "rp": rp,
                    "ng": ng,
                    "off_words": off_words,
                    "off_loop": off_loop,
                    "on_words": on_words,
                    "on_loop": on_loop,
                })

    print()
    print("=" * 78)
    print("SUMMARY — word counts and loop heuristic")
    print("=" * 78)
    headers = ["image_id", "rp", "ng", "off_words", "off_loop", "on_words", "on_loop"]
    widths = [
        max(len(h), max(len(str(r[k])) for r in summary_rows))
        for h, k in zip(headers, headers)
    ]
    sep = "  "
    print("  " + sep.join(f"{h:<{w}}" for h, w in zip(headers, widths)))
    print("  " + sep.join("-" * w for w in widths))
    for r in summary_rows:
        print("  " + sep.join(f"{str(r[k]):<{w}}" for k, w in zip(headers, widths)))

    print()
    print("Pick: smallest rp where ON has off_loop=False AND off_words is "
          "comparable to the rp=1.0 baseline. Then validate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
