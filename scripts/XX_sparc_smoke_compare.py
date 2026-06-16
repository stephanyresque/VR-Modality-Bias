#!/usr/bin/env python
"""Phase-0 SPARC smoke: prove that ``alpha > 1`` actually changes the caption.

Reads a config (e.g. ``configs/baseline.yaml`` — the one Arthur reoriented
for ``baseline_qwen_sparc``), iterates the first ``--limit`` images of the
dataset, and for each image generates the caption **twice** with the **same
seed**:

    1. SPARC OFF — vanilla ``model.generate_caption(...)`` on the unmodified
       Qwen-2.5-VL forward.
    2. SPARC ON  — after ``add_custom_attention_layers(...)`` with the
       requested ``--alpha`` (defaults to ``1.5``). The selection layer,
       ``tau``, ``beta`` and ``se_layers`` keep Arthur's defaults; the only
       parameter being tuned here is ``alpha`` (the paper's calibration
       coefficient — ``alpha=1`` is no-op).

At the end the script reports how many captions changed. Acceptance for
Phase 0 (per Stephany's instruction): **at least one caption must differ**
between OFF and ON, otherwise SPARC is still inert and the rest of the
pipeline is meaningless. The script exits with returncode 1 when nothing
changes.

Notes for the operator
----------------------
* Same probe + buffer reset cadence as ``scripts/XX_inference_sparc.py``.
* All generations are deterministic per ``(seed_global, image_id)`` thanks
  to ``derive_image_seed`` — the OFF/ON pair therefore samples from
  identical RNG state, isolating the effect of the calibration.
* SPARC is installed *after* the OFF passes so we don't have to undo the
  monkey-patch (this script doesn't restore the original forwards). If you
  need to call ``generate_caption`` again with SPARC OFF after this
  smoke test, reload the model.

CLI
---
    python scripts/XX_sparc_smoke_compare.py --config configs/baseline.yaml
    python scripts/XX_sparc_smoke_compare.py --config configs/baseline.yaml \
        --alpha 2.0 --limit 4
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import os
import sys
import traceback
from pathlib import Path

from loguru import logger
from PIL import Image
from pyprojroot import here

try:
    from vr_modality_bias.data.prompts import get_prompt
    from vr_modality_bias.models.registry import build_model
    from vr_modality_bias.utils.attn import (
        SelectedIndexBuffer,
        add_custom_attention_layers,
    )
    from vr_modality_bias.utils.config import load_config
    from vr_modality_bias.utils.device import resolve_dtype, select_device
    from vr_modality_bias.utils.seeds import derive_image_seed
except ModuleNotFoundError:
    sys.path.insert(0, str(here()))

    from src.vr_modality_bias.data.prompts import get_prompt
    from src.vr_modality_bias.models.registry import build_model
    from src.vr_modality_bias.utils.attn import (
        SelectedIndexBuffer,
        add_custom_attention_layers,
    )
    from src.vr_modality_bias.utils.config import load_config
    from src.vr_modality_bias.utils.device import resolve_dtype, select_device
    from src.vr_modality_bias.utils.seeds import derive_image_seed


def _probe_image_tokens(
    model,
    image: Image.Image,
    prompt: str,
    image_token_id: int,
) -> tuple[int, int, int]:
    """Return ``(image_token_index, input_len, num_image_patches)``.

    Mirrors the helper in ``scripts/XX_inference_sparc.py``. The input
    length here means *prompt length minus the image patch tokens* — that
    is what ``SelectedIndexBuffer.update_input_len`` expects so the
    ``num_image_patches`` can be derived at prefill time.
    """
    messages = model._build_messages(prompt, image)
    prompt_text = model._processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    inputs = model._processor(text=[prompt_text], images=[image], return_tensors="pt")
    input_ids = inputs["input_ids"][0]
    image_positions = (input_ids == image_token_id).nonzero(as_tuple=True)[0]
    total_len = int(input_ids.shape[-1])
    num_image_patches = int(image_positions.numel())
    return int(image_positions[0]), total_len - num_image_patches, num_image_patches


def _short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:8]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.5,
        help=(
            "SPARC calibration coefficient (1.0 is no-op). The paper's "
            "recalibration is a := a * alpha^c, where c is the selection "
            "count from previous steps. Pick a soft value > 1 (default 1.5)."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Number of images to compare (default: 3).",
    )
    args = parser.parse_args()

    if args.alpha == 1.0:
        logger.warning(
            "alpha=1.0 means no-op calibration — SPARC will produce identical "
            "captions by design. Pass --alpha > 1 to make this test meaningful."
        )

    cfg = load_config(args.config)

    # ---- model load ----
    model = build_model(cfg["model"]["key"])
    model.model_id = str(cfg["model"]["model_id"])
    dtype = resolve_dtype(str(cfg["model"]["dtype"]))
    device = select_device("cuda")
    if hasattr(model, "_dtype"):
        model._dtype = dtype  # noqa: SLF001
    logger.info(f"Loading model {model.model_id} on {device} (dtype={dtype})…")
    model.load(device)
    logger.info(f"Model loaded. n_layers={model.n_layers}")

    # ---- dataset ----
    images_dir = cfg["dataset"]["images_dir"]
    images_files = sorted(glob.glob(f"{images_dir}{os.sep}*.jpg"))[: args.limit]
    if not images_files:
        logger.error(f"No .jpg files found under {images_dir}")
        return 1
    logger.info(f"Comparing on {len(images_files)} image(s).")

    # ---- prompt + generation kwargs (same for OFF and ON) ----
    prompt = get_prompt(str(cfg["task"]["prompt_key"]))
    seed_global = int(cfg["run"]["seed_global"])
    max_new_tokens = int(cfg["generation"]["max_new_tokens"])
    gen_kwargs = {
        "do_sample": bool(cfg["generation"]["do_sample"]),
        "temperature": float(cfg["generation"]["temperature"]),
        "top_p": float(cfg["generation"]["top_p"]),
        "repetition_penalty": float(cfg["generation"]["repetition_penalty"]),
    }

    image_token_id = int(model._model.config.image_token_id)

    # ---- (1) SPARC OFF ----
    logger.info("=" * 70)
    logger.info("PASS 1/2 — SPARC OFF (baseline)")
    logger.info("=" * 70)
    captions_off: dict[str, str] = {}
    for image_path in images_files:
        image_id = Path(image_path).stem
        with Image.open(image_path) as raw:
            image = raw.convert("RGB")
        caption = model.generate_caption(
            image=image,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            seed=derive_image_seed(seed_global, image_id),
            generation_kwargs=gen_kwargs,
        )
        captions_off[image_id] = caption
        logger.info(
            f"[{image_id}] OFF sha1={_short_hash(caption)} len={len(caption)}"
        )
        logger.info(f"            {caption[:160]}{'…' if len(caption) > 160 else ''}")

    # ---- (2) install SPARC and re-generate ----
    indices_buffer = SelectedIndexBuffer()
    with Image.open(images_files[0]) as raw:
        probe_image = raw.convert("RGB")
    image_token_index, _, _ = _probe_image_tokens(
        model, probe_image, prompt, image_token_id
    )
    add_custom_attention_layers(
        model._model,
        alpha=args.alpha,
        indices_buffer=indices_buffer,
        image_token_index=image_token_index,
    )
    logger.info("=" * 70)
    logger.info(
        f"PASS 2/2 — SPARC ON (alpha={args.alpha}, tau=2, selected_layer=15, "
        "se_layers=(0,31), beta=0 — Arthur's defaults except alpha)"
    )
    logger.info(f"image_token_index={image_token_index}")
    logger.info("=" * 70)

    captions_on: dict[str, str] = {}
    for image_path in images_files:
        image_id = Path(image_path).stem
        with Image.open(image_path) as raw:
            image = raw.convert("RGB")
        # SPARC bookkeeping per image: re-probe input_len/num_image_patches
        # (image dimensions vary) and reset the index buffer.
        _, input_len, _ = _probe_image_tokens(model, image, prompt, image_token_id)
        indices_buffer.reset()
        indices_buffer.update_input_len(input_len)

        caption = model.generate_caption(
            image=image,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            seed=derive_image_seed(seed_global, image_id),  # SAME seed as OFF
            generation_kwargs=gen_kwargs,
        )
        captions_on[image_id] = caption
        logger.info(
            f"[{image_id}] ON  sha1={_short_hash(caption)} len={len(caption)}"
        )
        logger.info(f"            {caption[:160]}{'…' if len(caption) > 160 else ''}")

    # ---- comparison rollup ----
    logger.info("=" * 70)
    logger.info("COMPARISON")
    logger.info("=" * 70)
    n_diff = 0
    for image_id in captions_off:
        off = captions_off[image_id]
        on = captions_on[image_id]
        identical = off == on
        if not identical:
            n_diff += 1
        flag = "SAME" if identical else "DIFFERS"
        logger.info(
            f"[{image_id}] {flag:8s}  off_sha1={_short_hash(off)} "
            f"on_sha1={_short_hash(on)}"
        )
        if not identical:
            logger.info(f"  OFF: {off[:200]}")
            logger.info(f"  ON : {on[:200]}")

    logger.info("")
    logger.info(
        f"Result: {n_diff}/{len(captions_off)} caption(s) changed between "
        f"SPARC OFF and SPARC ON (alpha={args.alpha})."
    )
    if n_diff == 0:
        logger.error(
            "PHASE 0 FAIL: SPARC produced no caption change. Either alpha is "
            "still effectively neutral (try a larger value), or the SPARC "
            "hooks are not firing — inspect indices_buffer.indices1/indices2 "
            "and the gen_new_token branch in utils/attn.py::forward_qwen25vl."
        )
        return 1

    logger.info("PHASE 0 OK: SPARC calibration is active on the Qwen-2.5-VL path.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover (operator-side script)
        logger.debug(f"exception: {exc}")
        logger.error(traceback.format_exc())
        raise SystemExit(1)
