#!/usr/bin/env python
"""Step 0 -- pure inspection of InternVL2-8B before any wrapper / patching.

Loads the model with ``trust_remote_code=True`` and prints ONLY the facts
that decide the wrapper + the SPARC forward:

    1. top-level class name (feeds the family markers in utils/attn.py)
    2. decoder path (where the ``.layers`` list actually lives)
    3. attention submodule structure (fused ``wqkv`` + ``wo``? separate
       ``q_proj/k_proj/v_proj``? attribute name -- ``self_attn`` or
       ``attention``?)
    4. num_hidden_layers of the language backbone
    5. the id of the ``<IMG_CONTEXT>`` visual token (NOT config.image_token_id
       for InternVL)
    6. shape audit of ``wqkv`` and derived ``num_heads`` / ``num_kv_heads``
       (drives the QKV split in forward_internlm2)

Nothing is patched. Nothing is generated. Nothing is written to disk.
Runs in ~1 minute on the DGX (weights load, one metadata scan, print, exit).

The dump is what guides Passos 1-3 of the InternVL bloco -- do NOT skip.

CLI
---
    python scripts/25_internvl_inspect.py
    python scripts/25_internvl_inspect.py --model-id OpenGVLab/InternVL2-8B
"""

from __future__ import annotations

import argparse
import sys
from typing import Any


def _tree_of(module, depth: int = 3, prefix: str = "") -> list[str]:
    """Return a short string tree of ``module`` for depths <= ``depth``.

    Just names + class-name of each child. Kept short so the output fits
    a paste-back.
    """
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
        "--model-id", default="OpenGVLab/InternVL2-8B",
        help="HF model id or local path (default: OpenGVLab/InternVL2-8B).",
    )
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16",
        help="Load dtype. bf16 matches how InternVL was trained.",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="'cuda' or 'cpu'. Loading on cpu works for structure inspection "
             "but takes RAM (~16GB for InternVL2-8B).",
    )
    args = parser.parse_args()

    print("=" * 78)
    print(f"InternVL2 Step-0 inspection -- {args.model_id}")
    print("=" * 78)

    import torch
    from transformers import AutoModel, AutoTokenizer

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    print()
    print(f"Loading tokenizer ({args.model_id}) with trust_remote_code=True...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, trust_remote_code=True,
    )

    print(f"Loading model ({args.model_id}, dtype={args.dtype}) with trust_remote_code=True...")
    model = AutoModel.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
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
    print(f"  -> add this string as a marker to utils/attn.py.")

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
        "language_model.model",     # InternVL2 conventional (LM-with-model wrapper)
        "language_model",
        "model.language_model.model",
        "model.language_model",
        "model.model",
        "model",
    )
    for path in candidate_paths:
        node = _walk(model, path)
        if node is None:
            continue
        layers = _first(node, "layers")
        has_layers = layers is not None
        n_layers = len(layers) if has_layers else 0
        marker = "<-- USE THIS ONE" if has_layers else ""
        print(f"  model.{path}: {type(node).__name__}  "
              f"has_layers={has_layers}  n_layers={n_layers}  {marker}")

    # ---- 4. attention submodule structure ----
    print()
    print("-" * 78)
    print("4. ATTENTION SUBMODULE (fused vs separate QKV?)")
    print("-" * 78)
    decoder = None
    for path in candidate_paths:
        node = _walk(model, path)
        if node is not None and getattr(node, "layers", None) is not None:
            decoder = node
            break
    if decoder is None:
        print("  ERROR: could not locate a decoder with .layers on the model.")
        return 1
    layer0 = decoder.layers[0]
    print(f"  layer 0 : {type(layer0).__name__}")
    # Which attribute holds the attention module?
    attn_attr = None
    for cand in ("self_attn", "attention", "attn"):
        if hasattr(layer0, cand):
            attn_attr = cand
            print(f"  attention attribute: layer.{cand!r}  <-- USE THIS ONE")
            break
    else:
        print("  WARN: no 'self_attn'/'attention'/'attn' on layer 0 -- inspect the tree below.")
    if attn_attr:
        attn = getattr(layer0, attn_attr)
        print(f"  attention class: {type(attn).__name__}")
        print("  attention children:")
        for name, child in attn.named_children():
            shape_note = ""
            if hasattr(child, "weight") and hasattr(child.weight, "shape"):
                shape_note = f"  weight.shape={tuple(child.weight.shape)}"
            print(f"    {name}: {type(child).__name__}{shape_note}")

        # Explicit check for fused QKV.
        has_wqkv = hasattr(attn, "wqkv")
        has_wo = hasattr(attn, "wo")
        has_q = hasattr(attn, "q_proj")
        has_k = hasattr(attn, "k_proj")
        has_v = hasattr(attn, "v_proj")
        has_o = hasattr(attn, "o_proj")
        print(f"  fused QKV?  wqkv={has_wqkv} wo={has_wo}")
        print(f"  separate?   q_proj={has_q} k_proj={has_k} v_proj={has_v} o_proj={has_o}")

        # Head counts (feed the QKV reshape).
        for k in ("num_heads", "num_attention_heads", "num_key_value_heads",
                  "num_key_value_groups", "head_dim", "hidden_size"):
            v = getattr(attn, k, "<missing>")
            print(f"  attn.{k}: {v}")

    # ---- 5. IMG_CONTEXT token id ----
    print()
    print("-" * 78)
    print("5. VISUAL TOKEN ID  (InternVL uses <IMG_CONTEXT>, NOT config.image_token_id)")
    print("-" * 78)
    for tok in ("<IMG_CONTEXT>", "<img>", "<image>"):
        try:
            tid = tokenizer.convert_tokens_to_ids(tok)
        except Exception as exc:
            tid = f"<error: {exc}>"
        marker = ""
        if isinstance(tid, int) and tid > 0 and tok == "<IMG_CONTEXT>":
            marker = "  <-- USE THIS ONE"
        print(f"  tokenizer.convert_tokens_to_ids({tok!r}) = {tid}{marker}")

    # Also try to find it via config.
    cfg = model.config
    for cand in ("image_token_id", "img_context_token_id", "image_token_index"):
        v = getattr(cfg, cand, "<missing>")
        print(f"  config.{cand}: {v}")

    # ---- 6. LM head + num_hidden_layers ----
    print()
    print("-" * 78)
    print("6. LM HEAD AND N_LAYERS")
    print("-" * 78)
    lm_paths = (
        "language_model.lm_head",
        "language_model.output",
        "language_model.model.lm_head",
        "lm_head",
        "output",
    )
    for path in lm_paths:
        node = _walk(model, path)
        if node is not None:
            print(f"  {path}: {type(node).__name__}  "
                  f"out_features={getattr(node, 'out_features', '?')}")
    for path in (
        "config.llm_config.num_hidden_layers",
        "config.text_config.num_hidden_layers",
        "config.num_hidden_layers",
    ):
        node = _walk(model, path)
        if node is not None:
            print(f"  {path} = {node}")

    print()
    print("=" * 78)
    print("Step-0 inspection DONE. Paste this whole output back so the wrapper /")
    print("forward_internlm2 draft can be finalised without guessing.")
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
