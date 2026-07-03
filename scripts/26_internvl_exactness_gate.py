#!/usr/bin/env python
"""Step 6.3 -- exactness gate for the InternVL2 SPARC patch.

Idea
====

With ``alpha=1.0`` the SPARC forward MUST be numerically identical to
the unpatched attention:

    * ``calibrate(value, alpha=1.0)`` is ``value *= 1.0`` -- a no-op.
    * The other SPARC side-effects (``image_attention`` bookkeeping,
      selection buffer) don't touch the OUTPUT tensor.

So: patched forward at ``alpha=1.0`` returns exactly what the
unpatched forward returns, bit-for-bit (fp16/bf16 tolerance).

If the patched forward drifts from the unpatched one at ``alpha=1.0``,
the QKV split, reshape, or ``wo`` wiring in ``forward_internlm2`` is
wrong. That is the #1 risk (fused-wqkv + GQA head interleaving is not
the same as separate q/k/v_proj). This script is the specific check
that catches it BEFORE we trust any downstream CHAIR number.

What it does
============

1. Loads InternVL2-8B (bf16, eager, ``trust_remote_code=True``).
2. Builds a small fake ``pixel_values`` + ``input_ids`` batch and runs
   ONE forward through the language model, capturing the output of the
   first decoder layer's attention. This is the reference.
3. Installs SPARC via ``add_custom_attention_layers`` with
   ``alpha=1.0``, ``beta=0.0``, ``tau=1e9`` (so no selection fires),
   and a dummy ``SelectedIndexBuffer`` pre-populated with a plausible
   ``image_positions``.
4. Runs the same forward again, captures the same layer's output.
5. Reports ``max |patched - reference|`` and ``|| patched - reference ||_2``
   per layer, PLUS a global PASS/FAIL under a tight tolerance
   (1e-3 abs in bf16 -- accumulated fp noise across the GQA reshape).

If the gate fails, do NOT run CHAIR. Fix the QKV split first.

CLI
---

    python scripts/26_internvl_exactness_gate.py
    python scripts/26_internvl_exactness_gate.py --image PATH.jpg
    python scripts/26_internvl_exactness_gate.py --n-layers 4    # smoke: only patch 4 layers
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
    parser.add_argument("--model-key", default="internvl2-8b")
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
        help="Maximum |patched - reference| that we still call a PASS. bf16 "
             "accumulates a bit across the fused-QKV reshape; 1e-3 is a "
             "reasonable ceiling. If the actual number is orders of "
             "magnitude larger (e.g. 1e-1), the QKV split is wrong.",
    )
    parser.add_argument(
        "--n-layers", type=int, default=None,
        help="Limit the gate to the first N layers (smoke). Default: all.",
    )
    args = parser.parse_args()

    if not args.image.exists():
        print(f"ERROR: image not found -- {args.image}", file=sys.stderr)
        return 1

    print("=" * 78)
    print(f"InternVL2 SPARC exactness gate  (alpha=1.0 == identity check)")
    print("=" * 78)

    # ---- load model ----
    device = select_device("cuda")
    print(f"loading {args.model_key} on {device}...")
    wrapper = build_model(args.model_key)
    wrapper.load(device)
    print(f"loaded. n_layers={wrapper.n_layers}  "
          f"image_token_id={getattr(wrapper, 'image_token_id', '?')}")

    family = detect_model_family(wrapper._model)  # noqa: SLF001
    if family != "internlm2":
        print(f"WARN: detected family={family!r}, not 'internlm2'. "
              f"Check _INTERNLM2_MARKERS in utils/attn.py.",
              file=sys.stderr)

    # ---- reference: unpatched forward ----
    # We piggy-back on generate_caption's ``_preprocess_image`` for the
    # image, and use the tokenizer for a short input_ids. Nothing about
    # the identity check depends on the actual content -- just that BOTH
    # runs see the same inputs.
    from PIL import Image
    img = Image.open(args.image).convert("RGB")
    pixel_values = wrapper._preprocess_image(img)  # noqa: SLF001
    tokenizer = wrapper._tokenizer  # noqa: SLF001

    # Simple input: prompt tokens only. InternVL's chat API is bypassed --
    # we build the smallest possible forward call that runs the LM
    # backbone through all decoder layers.
    prompt_ids = tokenizer(
        args.prompt, return_tensors="pt",
    )["input_ids"].to(device)

    decoder = decoder_of(wrapper._model)  # noqa: SLF001
    layers_to_check = list(range(wrapper.n_layers))
    if args.n_layers is not None:
        layers_to_check = layers_to_check[: args.n_layers]

    # We compare the ATTENTION OUTPUT of the SPARC-patched forward against
    # the ORIGINAL (unpatched) forward on the same layer, on the same
    # hidden_states. To do that surgically, we wrap each attention with a
    # capture hook rather than run a full forward twice: the SPARC patch
    # is applied *after* the reference capture, so both see the SAME input.

    # Capture the input to each layer's attention on an unpatched pass.
    captured_inputs: dict[int, tuple] = {}
    captured_reference_outputs: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int, attn_module):
        original_forward = attn_module.forward

        def _capture(*args, **kwargs):
            # Save inputs (args + kwargs) so we can replay them against
            # the patched forward with identical arguments.
            captured_inputs[layer_idx] = (args, kwargs)
            out = original_forward(*args, **kwargs)
            # attn.forward returns (attn_output, attn_weights) or a longer
            # tuple depending on the model. We only need [0].
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
        _ = wrapper._model.language_model(  # noqa: SLF001
            input_ids=prompt_ids,
            use_cache=False,
        )
    _restore_forwards(decoder, originals)
    print(f"  captured inputs / outputs for {len(captured_reference_outputs)} layer(s)")

    # ---- patched: SPARC with alpha=1.0, tau=1e9 (nothing selected) ----
    buffer = SelectedIndexBuffer()
    buffer.reset()
    buffer.update_input_len(int(prompt_ids.shape[-1]))
    # Empty image_positions -- image_attention path expects a tensor if
    # image_positions is not None; provide an empty long tensor so the
    # index_select yields a well-defined slice.
    buffer.image_positions = torch.zeros(0, dtype=torch.long)

    add_custom_attention_layers(
        wrapper._model,  # noqa: SLF001
        alpha=1.0, beta=0.0, tau=1e9,
        selected_layer=-1,
        se_layers=(0, wrapper.n_layers - 1),
        image_token_index=0,
        indices_buffer=buffer,
        family="internlm2",
    )

    print()
    print(f"patched pass (alpha=1.0, tau=1e9 -- nothing selected)...")
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
        print("The patched forward does not match the unpatched forward at "
              "alpha=1.0. The most likely culprit is the QKV split in "
              "forward_internlm2 (utils/attn.py) -- the fused wqkv reshape "
              "or the GQA head interleaving is inconsistent with the "
              "InternLM2 remote code. Do NOT run CHAIR until this passes.")
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
