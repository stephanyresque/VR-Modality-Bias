#!/usr/bin/env python
"""Step 6.3 -- exactness gate for the InternVL3 SPARC patch.

Idea
====

With ``alpha=1.0`` the SPARC forward MUST be numerically identical to
the unpatched attention:

    * ``calibrate(value, alpha=1.0)`` is ``value *= 1.0`` -- a no-op.
    * The other SPARC side-effects (``image_attention`` bookkeeping,
      selection buffer) don't touch the OUTPUT tensor.

So: patched forward at ``alpha=1.0`` returns exactly what the
unpatched forward returns, bit-for-bit (fp16/bf16 tolerance).

Why we still run this on the native InternVL3-8B-hf
---------------------------------------------------

The native checkpoint (``InternVLForConditionalGeneration``) uses a
Qwen2.5 backbone with SEPARATE ``q_proj``/``k_proj``/``v_proj`` and
``o_proj``, so ``detect_model_family`` picks ``forward_qwen25vl`` or
``forward_llama`` (both already validated by scripts/13). The QKV-split
bug that motivated this gate for the legacy InternVL2 remote path is
NOT a concern here. What we ARE still checking:

    1. That ``detect_model_family`` picks the right forward for the
       InternVL native class (routing via ``config.text_config.model_type``).
    2. That the picked forward's rotary-embedding math matches how the
       InternVL wrapper feeds ``position_embeddings`` into the decoder
       layers.
    3. That the ``se_layers`` / selection bookkeeping doesn't alter the
       output tensor path at ``alpha=1.0, tau=1e9``.

If the gate fails, do NOT run CHAIR. Inspect the wrapper / attn family
routing first (Passo 0 dump).

What it does
============

1. Loads InternVL3-8B-hf (bf16, eager) via the InternVLWrapper.
2. Builds proper multimodal inputs via the processor (image + prompt),
   captures per-layer attention output on an unpatched pass.
3. Installs SPARC via ``add_custom_attention_layers`` with
   ``alpha=1.0``, ``beta=0.0``, ``tau=1e9`` (so no selection fires),
   and a ``SelectedIndexBuffer`` pre-populated with the same
   ``image_positions`` a real run would have.
4. Re-runs the same inputs, captures the same layers' outputs.
5. Reports ``max |patched - reference|`` and ``|| patched - reference ||_2``
   per layer, plus a global PASS/FAIL under a tight tolerance
   (1e-3 abs in bf16).

CLI
---

    python scripts/internvl_exactness_gate.py
    python scripts/internvl_exactness_gate.py --image PATH.jpg
    python scripts/internvl_exactness_gate.py --n-layers 4    # smoke: only first 4 layers
    python scripts/internvl_exactness_gate.py --family qwen   # override auto-detect
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    from vr_modality_bias.models.registry import build_model
    from vr_modality_bias.utils.attn import (
        SelectedIndexBuffer,
        _attention_module_of,
        add_custom_attention_layers,
        decoder_of,
        detect_model_family,
    )
    from vr_modality_bias.utils.device import select_device
except ModuleNotFoundError:
    from pyprojroot import here
    sys.path.insert(0, str(here()))

    from src.vr_modality_bias.models.registry import build_model
    from src.vr_modality_bias.utils.attn import (
        SelectedIndexBuffer,
        _attention_module_of,
        add_custom_attention_layers,
        decoder_of,
        detect_model_family,
    )
    from src.vr_modality_bias.utils.device import select_device


def _snapshot_forwards(decoder) -> list:
    """Snapshot every layer's attention.forward so we can restore later."""
    return [_attention_module_of(layer).forward for layer in decoder.layers]


def _restore_forwards(decoder, originals: list) -> None:
    for layer, orig in zip(decoder.layers, originals):
        _attention_module_of(layer).forward = orig


def _l1_l2_max(patched: torch.Tensor, reference: torch.Tensor) -> dict:
    diff = (patched.detach().float() - reference.detach().float()).abs()
    return {
        "max_abs": float(diff.max().item()),
        "l2":      float(diff.pow(2).sum().sqrt().item()),
        "mean_abs": float(diff.mean().item()),
        "shape":   tuple(patched.shape),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", default="internvl3-8b-hf")
    parser.add_argument(
        "--image", type=Path,
        default=Path("data/processed/mscoco_baseline/images/000000000139.jpg"),
        help="Any image; content doesn't matter for the identity check.",
    )
    parser.add_argument(
        "--prompt", default="Describe this image.",
        help="Any prompt; content doesn't matter for the identity check.",
    )
    parser.add_argument(
        "--abs-tol", type=float, default=1e-3,
        help="Maximum |patched - reference| that we still call a PASS. "
             "bf16 accumulates a little; 1e-3 is a reasonable ceiling. "
             "If the actual number is orders of magnitude larger "
             "(e.g. 1e-1), something in the SPARC forward doesn't match "
             "the unpatched forward on THIS backbone.",
    )
    parser.add_argument(
        "--n-layers", type=int, default=None,
        help="Limit the gate to the first N layers (smoke). Default: all.",
    )
    parser.add_argument(
        "--family", default=None, choices=("qwen", "llama", "internlm2"),
        help="Override the SPARC family. Default: auto-detect. Use this "
             "if Passo 0 says mRoPE is present but auto-detect picked llama.",
    )
    args = parser.parse_args()

    if not args.image.exists():
        print(f"ERROR: image not found -- {args.image}", file=sys.stderr)
        return 1

    print("=" * 78)
    print(f"InternVL3 SPARC exactness gate  (alpha=1.0 == identity check)")
    print("=" * 78)

    # ---- load model ----
    device = select_device("cuda")
    print(f"loading {args.model_key} on {device}...")
    wrapper = build_model(args.model_key)
    wrapper.load(device)
    print(f"loaded. n_layers={wrapper.n_layers}")

    detected = detect_model_family(wrapper._model)  # noqa: SLF001
    family = args.family or detected
    print(f"detected family : {detected!r}   using : {family!r}")
    if args.family and args.family != detected:
        print(f"(overriding auto-detect via --family)")

    # ---- build inputs via the processor (native chat template + image) ----
    from PIL import Image
    img = Image.open(args.image).convert("RGB")
    processor = wrapper._processor  # noqa: SLF001

    messages = wrapper._build_messages(args.prompt, img)  # noqa: SLF001
    prompt_text = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    inputs = processor(
        text=[prompt_text], images=[img], return_tensors="pt",
    ).to(device)
    print(f"input_ids shape : {tuple(inputs['input_ids'].shape)}")

    decoder = decoder_of(wrapper._model)  # noqa: SLF001
    layers_to_check = list(range(wrapper.n_layers))
    if args.n_layers is not None:
        layers_to_check = layers_to_check[: args.n_layers]

    # ---- Capture unpatched attention output per layer via a hook ----
    captured_inputs: dict[int, tuple] = {}
    captured_reference_outputs: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int, attn_module):
        original_forward = attn_module.forward

        def _capture(*args, **kwargs):
            captured_inputs[layer_idx] = (args, kwargs)
            out = original_forward(*args, **kwargs)
            captured_reference_outputs[layer_idx] = (
                out[0] if isinstance(out, tuple) else out
            ).detach().clone()
            return out
        return _capture

    originals = _snapshot_forwards(decoder)
    for i, layer in enumerate(decoder.layers):
        if i not in layers_to_check:
            continue
        attn = _attention_module_of(layer)
        attn.forward = make_hook(i, attn)

    print()
    print(f"reference pass (unpatched, capturing {len(layers_to_check)} layer(s))...")
    with torch.no_grad():
        _ = wrapper._model(  # noqa: SLF001
            **inputs,
            use_cache=False,
        )
    _restore_forwards(decoder, originals)
    print(f"  captured inputs / outputs for {len(captured_reference_outputs)} layer(s)")

    # ---- patched: SPARC with alpha=1.0, tau=1e9 (nothing selected) ----
    input_ids = inputs["input_ids"][0]
    # Try to discover the image token id from the model config; fall back
    # to whatever the buffer sees at reset if config doesn't carry it.
    image_token_id = getattr(wrapper._model.config, "image_token_id", None)  # noqa: SLF001
    if image_token_id is None:
        image_token_id = getattr(
            wrapper._model.config, "image_token_index", None,  # noqa: SLF001
        )

    buffer = SelectedIndexBuffer()
    buffer.reset()
    if image_token_id is not None:
        image_positions = (input_ids == int(image_token_id)).nonzero(as_tuple=True)[0]
        buffer.image_positions = image_positions
        buffer.update_input_len(int(input_ids.shape[-1]) - int(image_positions.numel()))
        print(f"image_token_id={int(image_token_id)}  "
              f"n_image_positions={int(image_positions.numel())}")
    else:
        buffer.image_positions = torch.zeros(0, dtype=torch.long)
        buffer.update_input_len(int(input_ids.shape[-1]))
        print("WARN: no image_token_id on config -- Step 0 should confirm.")

    add_custom_attention_layers(
        wrapper._model,  # noqa: SLF001
        alpha=1.0, beta=0.0, tau=1e9,
        selected_layer=-1,
        se_layers=(0, wrapper.n_layers - 1),
        image_token_index=(
            int(buffer.image_positions[0])
            if buffer.image_positions is not None and buffer.image_positions.numel() > 0
            else 0
        ),
        indices_buffer=buffer,
        family=family,
    )

    print()
    print(f"patched pass (family={family!r}, alpha=1.0, tau=1e9 -- nothing selected)...")
    diffs: list[dict] = []
    for i in layers_to_check:
        if i not in captured_inputs:
            continue
        args_i, kwargs_i = captured_inputs[i]
        attn = _attention_module_of(decoder.layers[i])
        with torch.no_grad():
            out = attn.forward(*args_i, **kwargs_i)
        patched = out[0] if isinstance(out, tuple) else out
        diff = _l1_l2_max(patched, captured_reference_outputs[i])
        diff["layer"] = i
        diffs.append(diff)

    _restore_forwards(decoder, originals)

    # ---- report ----
    print()
    print("-" * 78)
    print(f"{'layer':>5}  {'shape':<28}  {'max_abs':>12}  {'mean_abs':>12}  {'l2':>12}")
    print("-" * 78)
    global_max = 0.0
    for d in diffs:
        print(f"{d['layer']:>5}  {str(d['shape']):<28}  "
              f"{d['max_abs']:>12.3e}  {d['mean_abs']:>12.3e}  {d['l2']:>12.3e}")
        global_max = max(global_max, d["max_abs"])

    print("-" * 78)
    passed = global_max <= args.abs_tol
    verdict = "PASS" if passed else "FAIL"
    print(f"global max |patched - reference| : {global_max:.3e}  (tol={args.abs_tol:.0e})")
    print(f"VERDICT : {verdict}")
    if not passed:
        print()
        print(
            "The patched forward does not match the unpatched forward at "
            "alpha=1.0. On the native InternVL3-8B-hf, the QKV split is NOT "
            "the culprit (backbone has separate q_proj/k_proj/v_proj). Most "
            "likely: (a) family auto-detect picked the wrong forward (retry "
            "with --family={qwen,llama}), or (b) the rotary-embedding path "
            "differs from what forward_qwen25vl/forward_llama expects on "
            "this backbone. Do NOT run CHAIR until this passes."
        )
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover
        import traceback
        print(f"top-level failure: {exc}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        raise SystemExit(1)
