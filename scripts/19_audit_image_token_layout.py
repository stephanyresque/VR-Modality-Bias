#!/usr/bin/env python
"""Diagnostic: inspect the image-token layout in the prefill input_ids.

The SPARC mechanism (``forward_qwen25vl`` in
``src/vr_modality_bias/utils/attn.py``) assumes that the image-placeholder
tokens occupy a **contiguous block** at
``input_ids[image_token_index : image_token_index + num_image_patches]``,
and calibrates / selects only inside that block. That assumption is
inherited from LLaVA's layout, where it's correct. For Qwen2.5-VL and
SmolVLM (Idefics3), the layout is more complex (special
``<vision_start>`` / ``<vision_end>`` markers, multi-row image grids with
in-line separators, etc.). If the assumed block actually contains text
tokens, SPARC is silently calibrating the wrong things — which would
explain why generation degenerates over many tokens.

This script is **purely diagnostic**. It does NOT run SPARC, does NOT
generate, does NOT touch the GPU. It only loads the processor + config,
runs the same prefill tokenisation that ``probe_image_token_index`` does,
and prints what's actually in those positions.

For each (model, image, prompt) it prints:
    1. the model's ``config.image_token_id`` (the ground-truth
       image-placeholder id)
    2. the values the SPARC probe computes: ``image_token_index``,
       ``num_image_patches``, ``input_len``
    3. the token-id sequence in the assumed SPARC range, plus 10 tokens
       of context on each side, decoded to text
    4. a factual verdict — how many of the ``num_image_patches`` positions
       are actually the image-placeholder id, and how many are something
       else.

The verdict is the only thing that matters. If ANY model shows
"X of N positions are NOT image-placeholder", that's the bug. No
conclusion / fix is proposed here — the user reads the numbers.

CLI
---
    # Qwen2.5-VL-7B (long prompt, one COCO image)
    python scripts/19_audit_image_token_layout.py --model qwen

    # SmolVLM-2.2B (same)
    python scripts/19_audit_image_token_layout.py --model smol

    # Override defaults
    python scripts/19_audit_image_token_layout.py --model qwen \\
        --image-path data/processed/mscoco_baseline/images/000000000285.jpg \\
        --prompt-key caption_short
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image
from pyprojroot import here

try:
    from vr_modality_bias.data.prompts import get_prompt
except ModuleNotFoundError:
    sys.path.insert(0, str(here()))
    from src.vr_modality_bias.data.prompts import get_prompt


MODEL_SPECS = {
    "qwen": {
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "build_messages": lambda prompt, image: [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    },
    "smol": {
        "model_id": "HuggingFaceTB/SmolVLM-Instruct",
        "build_messages": lambda prompt, image: [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    },
    "llava": {
        # Post-Block-2 the study is LLaVA-1.5-only. Layout expected:
        # contiguous block of <image> tokens at a fixed index (no
        # row/col separators like Idefics3) — the audit should report
        # 100% match, identical to Qwen2.5-VL's verdict.
        "model_id": "llava-hf/llava-1.5-7b-hf",
        "build_messages": lambda prompt, image: [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    },
}


def _decode_one(tokenizer, token_id: int) -> str:
    """Decode a single token id to a human-readable repr.

    ``skip_special_tokens=False`` so we actually see the placeholders
    (otherwise ``<|image_pad|>`` would print as empty string).
    """
    try:
        s = tokenizer.decode([int(token_id)], skip_special_tokens=False)
    except Exception:  # tokenizer raised on a malformed id
        return f"<decode_err:{token_id}>"
    return repr(s)


def _print_window(label: str, ids, tokenizer, start: int, end: int) -> None:
    """Print a window of input_ids with their decoded forms, one per line."""
    print(f"  {label} (positions {start}..{end - 1}, n={end - start}):")
    for i in range(start, end):
        tid = int(ids[i])
        print(f"    [{i:5d}] id={tid:>7d}  →  {_decode_one(tokenizer, tid)}")


def _audit_model(model_key: str, image_path: Path, prompt_key: str) -> None:
    from transformers import AutoConfig, AutoProcessor

    spec = MODEL_SPECS[model_key]
    model_id = spec["model_id"]

    print()
    print("=" * 78)
    print(f"MODEL: {model_id}")
    print(f"IMAGE: {image_path.name}")
    print(f"PROMPT: prompt_key={prompt_key}")
    print("=" * 78)

    # ---- 1. config.image_token_id ----------------------------------
    config = AutoConfig.from_pretrained(model_id)
    # The placeholder id lives under different names across VLM families.
    candidates = (
        "image_token_id",
        "image_token_index",
        "image_token",
    )
    image_token_id = None
    image_token_id_name = None
    for name in candidates:
        if hasattr(config, name):
            v = getattr(config, name)
            if isinstance(v, int):
                image_token_id = int(v)
                image_token_id_name = name
                break
    if image_token_id is None:
        # Sometimes nested under config.text_config or similar.
        for sub in ("text_config", "vision_config"):
            sub_cfg = getattr(config, sub, None)
            if sub_cfg is None:
                continue
            for name in candidates:
                if hasattr(sub_cfg, name):
                    v = getattr(sub_cfg, name)
                    if isinstance(v, int):
                        image_token_id = int(v)
                        image_token_id_name = f"{sub}.{name}"
                        break
            if image_token_id is not None:
                break
    print()
    print(
        f"  config image-token-id field : "
        f"{image_token_id_name or '(NONE FOUND)'} = {image_token_id}"
    )

    # ---- 2. tokenise the prefill ------------------------------------
    processor = AutoProcessor.from_pretrained(model_id)
    image = Image.open(image_path).convert("RGB")
    prompt = get_prompt(prompt_key)
    messages = spec["build_messages"](prompt, image)
    prefix_text = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    inputs = processor(text=[prefix_text], images=[image], return_tensors="pt")
    input_ids = inputs["input_ids"][0]
    total_len = int(input_ids.shape[-1])

    # ---- 3. SPARC probe (replicates probe_image_token_index) ---------
    if image_token_id is None:
        print("  ! cannot run SPARC probe — no image_token_id on config.")
        return

    image_positions = (input_ids == image_token_id).nonzero(as_tuple=True)[0]
    num_image_patches = int(image_positions.numel())
    if num_image_patches == 0:
        print("  ! no image-placeholder tokens found in input_ids.")
        return
    sparc_image_token_index = int(image_positions[0])
    sparc_input_len = total_len - num_image_patches

    print()
    print("  SPARC probe values (from probe_image_token_index):")
    print(f"    image_token_index : {sparc_image_token_index}")
    print(f"    num_image_patches : {num_image_patches}")
    print(f"    input_len         : {sparc_input_len}")
    print(f"    total seq length  : {total_len}")

    # ---- 4. inspect the assumed SPARC range ------------------------
    lo = sparc_image_token_index
    hi = sparc_image_token_index + num_image_patches
    range_ids = input_ids[lo:hi].tolist()
    n_match = sum(1 for tid in range_ids if int(tid) == image_token_id)
    n_mismatch = num_image_patches - n_match

    print()
    print(f"  SPARC ASSUMES positions [{lo}, {hi}) are all image patches.")
    print(f"    of {num_image_patches} positions in that range:")
    print(f"      {n_match:6d}  ARE  image_token_id={image_token_id}")
    print(f"      {n_mismatch:6d}  are NOT (something else — text? separators?)")

    if n_mismatch > 0:
        # Show which positions inside the range are wrong + what's there.
        wrong_positions = [
            (lo + i, int(tid))
            for i, tid in enumerate(range_ids)
            if int(tid) != image_token_id
        ]
        # Print up to 20 wrong entries so the output stays small.
        print(f"    first {min(20, len(wrong_positions))} mismatches:")
        for pos, tid in wrong_positions[:20]:
            print(f"      pos {pos:5d}: id={tid}  →  {_decode_one(processor.tokenizer, tid)}")

    # ---- 5. context windows ----------------------------------------
    print()
    print("  CONTEXT (decoded, special tokens preserved):")
    pre_lo = max(0, lo - 10)
    _print_window("  before SPARC range", input_ids, processor.tokenizer, pre_lo, lo)
    # Inside the range we just show the FIRST 5 and LAST 5 positions
    # (printing all ~thousand patch tokens would be useless noise).
    if num_image_patches <= 12:
        _print_window("  inside SPARC range", input_ids, processor.tokenizer, lo, hi)
    else:
        print(f"  inside SPARC range (first 5 of {num_image_patches}):")
        for i in range(lo, lo + 5):
            tid = int(input_ids[i])
            tag = "✓" if tid == image_token_id else "✗"
            print(f"    [{i:5d}] id={tid:>7d} {tag}  →  {_decode_one(processor.tokenizer, tid)}")
        print(f"  inside SPARC range (last 5 of {num_image_patches}):")
        for i in range(hi - 5, hi):
            tid = int(input_ids[i])
            tag = "✓" if tid == image_token_id else "✗"
            print(f"    [{i:5d}] id={tid:>7d} {tag}  →  {_decode_one(processor.tokenizer, tid)}")
    post_hi = min(total_len, hi + 10)
    _print_window("  after SPARC range", input_ids, processor.tokenizer, hi, post_hi)

    # ---- 6. ALSO scan the ENTIRE input_ids for image_token_id ---
    # to detect "non-contiguous" cases the SPARC probe will silently
    # miss (e.g. Idefics3's per-row image tokens interleaved with
    # separator tokens).
    all_image_positions = image_positions.tolist()
    if len(all_image_positions) > 1:
        diffs = [
            all_image_positions[i + 1] - all_image_positions[i]
            for i in range(len(all_image_positions) - 1)
        ]
        non_contiguous = [d for d in diffs if d != 1]
        print()
        print(
            f"  CONTIGUITY: among {len(all_image_positions)} image-placeholder "
            f"positions, {len(non_contiguous)} adjacent pairs have a gap > 1."
        )
        if non_contiguous:
            # Where are the gaps? Show the first 10 gap-positions.
            gap_positions = [
                (all_image_positions[i], all_image_positions[i + 1])
                for i in range(len(diffs))
                if diffs[i] != 1
            ][:10]
            print("    first gaps (a, b) where b > a + 1:")
            for a, b in gap_positions:
                # What's sitting in the gap?
                gap_ids = input_ids[a + 1: b].tolist()
                decoded = " ".join(
                    _decode_one(processor.tokenizer, t) for t in gap_ids[:5]
                )
                more = "..." if len(gap_ids) > 5 else ""
                print(f"      gap [{a+1}..{b-1}] ({b - a - 1} tokens): {decoded}{more}")

    # ---- 7. factual verdicts ---------------------------------------
    # Two verdicts side by side so the user can compare what SPARC USED to
    # do (legacy contiguous slice — what 19_audit reported originally)
    # against what SPARC NOW DOES after the per-id-mask fix.
    print()
    print("  VERDICT — legacy contiguous slice (pre-fix SPARC):")
    if n_mismatch == 0:
        print(
            f"    all {num_image_patches}/{num_image_patches} positions in "
            f"[{lo}, {hi}) ARE image-placeholders. SPARC was consistent here."
        )
    else:
        print(
            f"    {n_match}/{num_image_patches} positions in [{lo}, {hi}) "
            f"are image-placeholders; {n_mismatch} are something else "
            f"(separators/text). The contiguous-block assumption MISSES "
            f"the layout."
        )

    print()
    print("  VERDICT — per-id mask (post-fix SPARC, current code):")
    print(
        f"    SPARC now operates on the explicit list of {len(all_image_positions)} "
        f"positions where input_ids == image_token_id ({image_token_id}). "
        f"By construction, 100% of those positions are image-placeholders "
        f"(no separators / text mixed in)."
    )
    if len(all_image_positions) == num_image_patches:
        print(
            f"    Layout is contiguous → per-id mask gives IDENTICAL result "
            f"to the legacy slice (e.g. Qwen 2.5-VL)."
        )
    else:
        # Shouldn't happen — we built num_image_patches = len(image_positions).
        # Kept for defensive logging.
        print(
            f"    (Note: num_image_patches={num_image_patches} != "
            f"len(image_positions)={len(all_image_positions)}.)"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=list(MODEL_SPECS.keys()), required=True,
        help="Which model family to audit: 'qwen' or 'smol'.",
    )
    parser.add_argument(
        "--image-path", type=Path,
        default=Path("data/processed/mscoco_baseline/images/000000000139.jpg"),
        help="Image to use for the prefill probe.",
    )
    parser.add_argument(
        "--prompt-key", type=str, default="caption_long",
        help="Prompt key (from data/prompts). Default: caption_long.",
    )
    args = parser.parse_args()

    if not args.image_path.exists():
        print(f"image not found: {args.image_path}", file=sys.stderr)
        return 1

    _audit_model(args.model, args.image_path, args.prompt_key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
