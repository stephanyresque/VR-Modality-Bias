#!/usr/bin/env python
"""Step 0 -- pure inspection of InternVL3-8B (native HF checkpoint) before any
wrapper / patching.

We use the NATIVE HF checkpoint ``OpenGVLab/InternVL3-8B-hf`` (which loads
via ``AutoModelForImageTextToText`` on transformers v5 without
``trust_remote_code``) instead of the legacy remote-code checkpoint
``OpenGVLab/InternVL2-8B`` -- the latter breaks on v5 with
``'InternVLChatModel' object has no attribute 'all_tied_weights_keys'``.

InternVL3-8B-hf uses the Qwen2.5 language backbone, so we expect:

    * separate ``q_proj`` / ``k_proj`` / ``v_proj`` (NOT the fused ``wqkv``
      that InternLM2 uses),
    * ``o_proj`` (not ``wo``),
    * attention attribute ``layer.self_attn`` (not ``layer.attention``).

Which SPARC forward we plug in depends on ONE remaining question:

    * mRoPE (multimodal rotary; per-dim rope_section) => ``forward_qwen25vl``
    * plain 1D RoPE                                   => ``forward_llama``

This script prints the facts that answer that question -- do NOT skip it.

Prints
------

    1. top-level class name (feeds the family markers in utils/attn.py)
    2. child tree (~3 levels)
    3. decoder path (where the ``.layers`` list actually lives)
    4. attention submodule structure (q_proj/k_proj/v_proj separate?
       what attribute holds the attention module?)
    5. num_hidden_layers of the language backbone
    6. the id of the image token from config (native populates
       ``image_token_id`` / ``image_token_index``)
    7. lm_head candidates
    8. **RoPE flavour** -- prints whether ``config.text_config``
       (or the top-level config) carries ``rope_scaling`` /
       ``rope_parameters`` with an ``mrope_section`` field. Presence of
       ``mrope_section`` => forward_qwen25vl; absence => forward_llama.

Nothing is patched. Nothing is generated. Nothing is written to disk.
Runs in ~1-2 minutes on the DGX (weights load, one metadata scan, print).

CLI
---
    python scripts/25_internvl_inspect.py
    python scripts/25_internvl_inspect.py --model-id OpenGVLab/InternVL3-8B-hf
"""

from __future__ import annotations

import argparse
import sys
from typing import Any


def _tree_of(module, depth: int = 3, prefix: str = "") -> list[str]:
    """Return a short string tree of ``module`` for depths <= ``depth``."""
    lines = []
    for name, child in getattr(module, "named_children", lambda: [])():
        lines.append(f"{prefix}{name}: {type(child).__name__}")
        if depth > 1:
            lines.extend(_tree_of(child, depth=depth - 1, prefix=prefix + "  "))
    return lines


def _first(module, name: str) -> Any:
    return getattr(module, name, None)


def _walk(model, path: str):
    obj = model
    for part in path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-id", default="OpenGVLab/InternVL3-8B-hf",
        help="HF model id or local path (default: OpenGVLab/InternVL3-8B-hf).",
    )
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16",
        help="Load dtype. bf16 matches how InternVL was trained.",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="'cuda' or 'cpu'. Loading on cpu works for structure inspection "
             "but takes RAM (~16GB for InternVL3-8B).",
    )
    args = parser.parse_args()

    print("=" * 78)
    print(f"InternVL3 Step-0 inspection -- {args.model_id}")
    print("=" * 78)

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    print()
    print(f"Loading processor ({args.model_id})...")
    processor = AutoProcessor.from_pretrained(args.model_id)

    print(f"Loading model ({args.model_id}, dtype={args.dtype})...")
    # Native HF checkpoint -- NO trust_remote_code.
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(args.device)
    model.eval()

    # ---- 1. top-level class name ----
    print()
    print("-" * 78)
    print("1. TOP-LEVEL MODEL CLASS")
    print("-" * 78)
    top_cls = type(model).__name__
    print(f"  type(model).__name__ : {top_cls!r}")
    print(f"  -> add this string as a marker to utils/attn.py if not "
          f"already covered.")

    # ---- 2. child tree (~3 levels) ----
    print()
    print("-" * 78)
    print("2. MODEL TREE (depth <= 3)")
    print("-" * 78)
    for line in _tree_of(model, depth=3):
        print(f"  {line}")

    # ---- 3. decoder path (where .layers lives) ----
    print()
    print("-" * 78)
    print("3. DECODER PATH (needs to end at something with .layers)")
    print("-" * 78)
    candidate_paths = (
        "model.language_model",      # InternVLModel-style (v5 native)
        "language_model.model",
        "language_model",
        "model.language_model.model",
        "model.model",
        "model",
    )
    decoder = None
    for path in candidate_paths:
        node = _walk(model, path)
        if node is None:
            continue
        layers = _first(node, "layers")
        has_layers = layers is not None
        n_layers = len(layers) if has_layers else 0
        marker = ""
        if has_layers and decoder is None:
            decoder = node
            marker = "<-- USE THIS ONE"
        print(f"  model.{path}: {type(node).__name__}  "
              f"has_layers={has_layers}  n_layers={n_layers}  {marker}")

    if decoder is None:
        print("  ERROR: could not locate a decoder with .layers on the model.")
        return 1

    # ---- 4. attention submodule structure ----
    print()
    print("-" * 78)
    print("4. ATTENTION SUBMODULE (separate q/k/v_proj + o_proj expected on Qwen2.5)")
    print("-" * 78)
    layer0 = decoder.layers[0]
    print(f"  layer 0 : {type(layer0).__name__}")

    attn_attr = None
    for cand in ("self_attn", "attention", "attn"):
        if hasattr(layer0, cand):
            attn_attr = cand
            print(f"  attention attribute: layer.{cand!r}  <-- USE THIS ONE")
            break
    else:
        print("  WARN: no 'self_attn'/'attention'/'attn' on layer 0 -- "
              "inspect the tree above.")

    if attn_attr:
        attn = getattr(layer0, attn_attr)
        print(f"  attention class: {type(attn).__name__}")
        print("  attention children:")
        for name, child in attn.named_children():
            shape_note = ""
            if hasattr(child, "weight") and hasattr(child.weight, "shape"):
                shape_note = f"  weight.shape={tuple(child.weight.shape)}"
            print(f"    {name}: {type(child).__name__}{shape_note}")

        # Explicit projection presence check.
        has_wqkv = hasattr(attn, "wqkv")
        has_wo = hasattr(attn, "wo")
        has_q = hasattr(attn, "q_proj")
        has_k = hasattr(attn, "k_proj")
        has_v = hasattr(attn, "v_proj")
        has_o = hasattr(attn, "o_proj")
        print(f"  fused QKV?   wqkv={has_wqkv} wo={has_wo}  "
              f"(TRUE means InternLM2 backbone -- unexpected here)")
        print(f"  separate?    q_proj={has_q} k_proj={has_k} v_proj={has_v} "
              f"o_proj={has_o}  (TRUE means Qwen2.5 backbone -- expected)")

        for k in ("num_heads", "num_attention_heads", "num_key_value_heads",
                  "num_key_value_groups", "head_dim", "hidden_size",
                  "scaling", "attention_dropout"):
            v = getattr(attn, k, "<missing>")
            print(f"  attn.{k}: {v}")

    # ---- 5. num_hidden_layers ----
    print()
    print("-" * 78)
    print("5. N_LAYERS")
    print("-" * 78)
    for path in (
        "config.text_config.num_hidden_layers",
        "config.llm_config.num_hidden_layers",
        "config.num_hidden_layers",
    ):
        node = _walk(model, path)
        if node is not None:
            print(f"  {path} = {node}")

    # ---- 6. image token id ----
    print()
    print("-" * 78)
    print("6. IMAGE TOKEN ID  (native populates config.image_token_id)")
    print("-" * 78)
    cfg = model.config
    for cand in ("image_token_id", "image_token_index", "img_context_token_id"):
        v = getattr(cfg, cand, "<missing>")
        print(f"  config.{cand}: {v}")
    # Also via processor / tokenizer for cross-check.
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        for tok in ("<IMG_CONTEXT>", "<image>", "<img>"):
            try:
                tid = tokenizer.convert_tokens_to_ids(tok)
            except Exception as exc:
                tid = f"<error: {exc}>"
            print(f"  tokenizer.convert_tokens_to_ids({tok!r}) = {tid}")

    # ---- 7. LM head candidates ----
    print()
    print("-" * 78)
    print("7. LM HEAD CANDIDATES")
    print("-" * 78)
    lm_paths = (
        "lm_head",
        "language_model.lm_head",
        "model.lm_head",
        "model.language_model.lm_head",
        "language_model.model.lm_head",
    )
    for path in lm_paths:
        node = _walk(model, path)
        if node is not None:
            print(f"  {path}: {type(node).__name__}  "
                  f"out_features={getattr(node, 'out_features', '?')}")

    # ---- 8. RoPE flavour  (mRoPE => forward_qwen25vl, else forward_llama) ----
    print()
    print("-" * 78)
    print("8. ROPE FLAVOUR  <-- decides forward_qwen25vl vs forward_llama")
    print("-" * 78)
    # rope_parameters is the transformers v5 name; rope_scaling is legacy.
    for cfg_path in ("config.text_config", "config"):
        node = _walk(model, cfg_path)
        if node is None:
            continue
        print(f"  {cfg_path}:")
        for field in ("rope_parameters", "rope_scaling", "rope_theta",
                      "max_position_embeddings"):
            v = getattr(node, field, "<missing>")
            print(f"    .{field} : {v}")

    # Look for mrope_section explicitly -- presence is the deciding fact.
    has_mrope = False
    for cfg_path in ("config.text_config.rope_parameters",
                     "config.text_config.rope_scaling",
                     "config.rope_parameters",
                     "config.rope_scaling"):
        node = _walk(model, cfg_path)
        if isinstance(node, dict) and "mrope_section" in node:
            has_mrope = True
            print(f"  DETECTED: {cfg_path}.mrope_section = {node['mrope_section']}")
    print()
    if has_mrope:
        print("  => VERDICT: mRoPE present.  SPARC forward = forward_qwen25vl")
    else:
        print("  => VERDICT: no mrope_section found.  SPARC forward = forward_llama")
        print("     (Qwen2.5 TEXT backbone -- 1D RoPE like Llama.)")

    # ---- 9. tokenizer / processor sanity ----
    print()
    print("-" * 78)
    print("9. PROCESSOR / TOKENIZER SANITY (native chat template lives here)")
    print("-" * 78)
    print(f"  processor class : {type(processor).__name__}")
    print(f"  has apply_chat_template : {hasattr(processor, 'apply_chat_template')}")

    print()
    print("=" * 78)
    print("Step-0 inspection DONE. Paste this whole output back so the")
    print("wrapper + SPARC forward can be finalised without guessing.")
    print("=" * 78)
    return 0


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
