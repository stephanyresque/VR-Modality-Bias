#!/usr/bin/env python
"""Block-1 smoke for the LLaVA-1.5-7B migration.

Confirms three things, in order:
    1. ``LlavaWrapper`` loads — processor + model + dtype on CUDA, no
       crash, no missing-class shenanigans.
    2. The wrapper's introspection works — ``n_layers`` resolves to 32
       (LLaVA-1.5-7B's LlamaModel depth) and ``get_lm_head()`` returns a
       real ``torch.nn.Module`` (the LM head linear).
    3. Free generation runs end to end — greedy, 64 tokens, on a single
       image already in the dataset (000000000139 by default). Print
       the caption so a human can confirm it's coherent prose, not the
       empty string or a salad of tokens.

This is intentionally minimal. No SPARC, no metrics, no sweep, no log
file. Just a print to stdout. If it passes, we're cleared to start
Block 2 (retiring other families / wiring LLaVA into the experiment
orchestration). If it fails, we stop here and diagnose.

CLI
---
    python scripts/llava_smoke.py
    python scripts/llava_smoke.py --image-id 000000000285
    python scripts/llava_smoke.py --max-new-tokens 96
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image
from pyprojroot import here

try:
    from vr_modality_bias.models.registry import build_model
    from vr_modality_bias.utils.device import select_device
except ModuleNotFoundError:
    sys.path.insert(0, str(here()))

    from src.vr_modality_bias.models.registry import build_model
    from src.vr_modality_bias.utils.device import select_device


_DEFAULT_PROMPT = "Describe this image in one or two sentences."


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-id", type=str, default="000000000139",
        help="COCO image stem (zero-padded). Default 139 — already on disk.",
    )
    parser.add_argument(
        "--images-dir", type=Path,
        default=Path("data/processed/mscoco_baseline/images"),
        help="Where to look up the .jpg.",
    )
    parser.add_argument(
        "--prompt", type=str, default=_DEFAULT_PROMPT,
        help="Free-form prompt. Default asks for one-or-two-sentence caption.",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=64,
        help="Greedy decode budget. Default 64 — just enough to confirm prose.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Generation seed (greedy doesn't actually need it; kept for trace).",
    )
    args = parser.parse_args()

    image_path = args.images_dir / f"{args.image_id}.jpg"
    if not image_path.exists():
        print(f"image not found: {image_path}", file=sys.stderr)
        return 1

    print("=" * 78)
    print("LLaVA-1.5-7B smoke (Block 1 — wrapper + load + generate)")
    print("=" * 78)
    print(f"model_key  : llava-1.5-7b")
    print(f"image      : {image_path}")
    print(f"prompt     : {args.prompt!r}")
    print(f"decoding   : greedy, max_new_tokens={args.max_new_tokens}, seed={args.seed}")
    print("=" * 78)

    # ---- 1. instantiate + load -----------------------------------
    wrapper = build_model("llava-1.5-7b")
    device = select_device("cuda")
    print(f"loading {wrapper.model_id} on {device}...")
    wrapper.load(device)
    print(f"loaded. model class = {type(wrapper._model).__name__}")  # noqa: SLF001

    # ---- 2. introspection ----------------------------------------
    n_layers = wrapper.n_layers
    lm_head = wrapper.get_lm_head()
    print()
    print(f"n_layers          : {n_layers}  (expected 32 for LLaVA-1.5-7B)")
    print(f"lm_head type      : {type(lm_head).__name__}")
    print(f"lm_head out_dim   : {getattr(lm_head, 'out_features', '?')}  "
          f"(expected 32064 — LLaVA-1.5 vocab)")

    # ---- 3. free generation --------------------------------------
    image = Image.open(image_path).convert("RGB")
    # Greedy override — keep the smoke deterministic.
    gen_kwargs = {
        "do_sample": False,
        "num_beams": 1,
        "repetition_penalty": 1.0,
    }
    caption = wrapper.generate_caption(
        image=image,
        prompt=args.prompt,
        max_new_tokens=int(args.max_new_tokens),
        seed=int(args.seed),
        generation_kwargs=gen_kwargs,
    )

    print()
    print("─" * 78)
    print("CAPTION (greedy):")
    print("─" * 78)
    print(caption if caption else "<EMPTY>")
    print("─" * 78)
    print()
    print(
        "Block-1 pass criteria: n_layers == 32, lm_head resolved, caption is "
        "coherent prose (not empty, not a salad). Eyeball the text above."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
