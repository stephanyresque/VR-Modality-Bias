"""Forced teacher-forced decoding done step by step.

This module exists because the Etapa 2 instrument measures hidden states via
a **single forward pass** (``run_teacher_forcing``) with ``use_cache=False``,
while SPARC only takes effect **inside the autoregressive loop** (it compares
attention between consecutive generation steps and mutates the KV cache). The
two are therefore incompatible: plugging SPARC into ``run_teacher_forcing``
would be a silent no-op.

The function :func:`collect_forced_decoding` walks the reference caption
**token by token**, with ``use_cache=True``, feeding each token of
``caption_ref`` to the model in turn — exactly the conditions under which
SPARC fires — and collects the predictive hidden states. Output layout is
**deliberately identical** to ``ModelWrapper.run_teacher_forcing``:

    hidden_states : (n_layers, seq_len, hidden_dim)   with seq_len = caption_start + caption_len

So ``compute_kl_matrix`` reads ``hidden_states[layer, caption_start - 1 :
caption_start + caption_len - 1, :]`` exactly as before — **no index
arithmetic on the caller side**.

Index placement (the critical bit; see EXPERIMENT.md §4.2 / Phase 1 spec)
------------------------------------------------------------------------
* The **prefill** forward feeds ``input_ids[:caption_start]`` and produces
  hidden states at absolute positions ``0 .. caption_start - 1``.
* Each generation step ``j ∈ [0, caption_len)`` feeds the token at absolute
  position ``caption_start + j`` (i.e. ``caption_ref[j]``) and its output
  goes to the absolute index ``caption_start + j``.
* Therefore the **predictive state of caption token j** — the state used by
  ``compute_kl_matrix`` to predict ``caption_ref[j]`` — lives at the
  absolute index ``caption_start - 1 + j``:
    - ``j == 0`` → ``caption_start - 1`` (last position of the prefill).
    - ``j >= 1`` → ``caption_start - 1 + j`` (output of the step that fed
      ``caption_ref[j - 1]``).

We **intentionally feed all ``caption_len`` tokens** (including the last one,
which produces an output stored at ``caption_start + caption_len - 1`` and
is *not* read by ``compute_kl_matrix``). This is one extra forward but keeps
the output tensor layout byte-identical to the single-pass path, which is
what makes the §4.4 equivalence check a pure numerical comparison without
any index arithmetic on either side.

mRoPE / position bookkeeping
----------------------------
We do **not** assemble ``position_ids`` by hand. The model derives them
internally from ``cache_position`` and the ``rope_deltas`` it computed
during prefill (see ``Qwen2_5_VLForConditionalGeneration``, where
``rope_deltas`` lives in the output dataclass next to ``past_key_values``).
Our loop:

    1. PREFILL — feed the prefix with ``cache_position = arange(caption_start)``
       and capture ``outputs.rope_deltas``.
    2. STEP j — feed **only the new token** (one column) with
       ``cache_position = tensor([caption_start + j])`` plus the cached
       ``rope_deltas``. The model derives the mRoPE-correct
       ``position_ids`` from those two and runs through its KV cache.

This is what ``.generate``'s internal sampling loop does after
``_update_model_kwargs_for_generation`` rolls ``rope_deltas`` forward —
we just inline that one piece of state so we don't depend on
``prepare_inputs_for_generation`` (which in ``transformers`` 5.x slices
``input_ids`` only when ``cache_position`` is passed *to it*, a contract
that already bit us once).
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from typing import Any

import torch
from PIL import Image
from pyprojroot import here

try:
    from vr_modality_bias.models.base import HiddenStatesResult
except ModuleNotFoundError:
    sys.path.insert(0, str(here()))

    from src.vr_modality_bias.models.base import HiddenStatesResult

__all__ = ["collect_forced_decoding"]


def _iso_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def collect_forced_decoding(
    model_wrapper,
    image: Image.Image,
    prompt: str,
    caption_ref: str,
    *,
    sparc_buffer=None,
    output_dtype: torch.dtype = torch.float16,
) -> HiddenStatesResult:
    """Walk ``caption_ref`` token-by-token and capture the predictive hidden states.

    Args:
        model_wrapper: A loaded :class:`ModelWrapper` (currently validated
            for ``QwenVLWrapper``; the SmolVLM port lives in Phase 4).
        image: Source image — **already-noised** for the B condition (the
            caller passes ``noise_image_uniform(image, seed)``).
        prompt: Same prompt used by ``run_teacher_forcing``.
        caption_ref: The forced reference caption (same for A and B).
        sparc_buffer: Optional :class:`SelectedIndexBuffer` from the SPARC
            facade. When provided, it is reset and its ``input_len`` updated
            before the prefill (so the SPARC monkey-patched attention layers
            know which tokens are image patches for this particular image).
            When ``None``, SPARC is presumed inactive.
        output_dtype: Storage dtype for the returned hidden states. Defaults
            to ``float16`` to mirror ``run_teacher_forcing`` exactly.

    Returns:
        A :class:`HiddenStatesResult` with the same shape contract as
        ``ModelWrapper.run_teacher_forcing``:

            hidden_states : (n_layers, caption_start + caption_len, hidden_dim)
            input_ids     : (caption_start + caption_len,)
            caption_start : int
            caption_len   : int
            attention_mask: optional (caption_start + caption_len,)

    Raises:
        RuntimeError: If the prefix tokenisation drifts when the caption is
            appended (same sanity check as ``run_teacher_forcing``).
    """
    if model_wrapper._model is None or model_wrapper._processor is None or model_wrapper._device is None:
        raise RuntimeError("Model not loaded — call .load() first.")

    model = model_wrapper._model
    processor = model_wrapper._processor
    device = model_wrapper._device
    n_layers = int(model_wrapper.n_layers)

    image_rgb = image.convert("RGB")
    messages = model_wrapper._build_messages(prompt, image_rgb)
    prefix_text = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    # Tokenise prefix to discover caption_start (the absolute index where
    # forced tokens start landing).
    prefix_inputs = processor(text=[prefix_text], images=[image_rgb], return_tensors="pt")
    caption_start = int(prefix_inputs["input_ids"].shape[-1])

    # Tokenise the full sequence to lock in the forced caption tokens. This
    # matches what ``run_teacher_forcing`` does, so input_ids stay
    # byte-identical between the two paths.
    full_text = prefix_text + caption_ref.strip()
    full_inputs = processor(text=[full_text], images=[image_rgb], return_tensors="pt")
    full_ids = full_inputs["input_ids"][0]

    prefix_ids = prefix_inputs["input_ids"][0]
    if not torch.equal(full_ids[:caption_start], prefix_ids):
        raise RuntimeError(
            "Tokenisation of prefix changed when caption was appended. "
            "Cannot derive caption_start safely (forced-decoding path)."
        )
    caption_len = int(full_ids.shape[0] - caption_start)
    if caption_len <= 0:
        raise RuntimeError(
            f"caption_len <= 0 ({caption_len}); caption_ref empty. "
            f"caption_ref={caption_ref!r}"
        )

    seq_len = caption_start + caption_len  # final hidden_states length

    # SPARC bookkeeping for this image — must run BEFORE the prefill so the
    # custom forward sees the right input_len. The image_token_index was
    # already wired into the attention layers when SPARC was enabled
    # externally (see experiment.sparc.enable_sparc).
    if sparc_buffer is not None:
        sparc_buffer.reset()
        # ``input_len`` is the prompt length excluding the image patch
        # tokens — the same definition the existing
        # scripts/XX_inference_sparc.py uses via _probe_image_tokens.
        image_token_id = int(model.config.image_token_id)
        image_positions = (prefix_ids == image_token_id).nonzero(as_tuple=True)[0]
        num_image_patches = int(image_positions.numel())
        sparc_buffer.update_input_len(caption_start - num_image_patches)

    # Move tokenised inputs onto the model's device.
    prefix_input_ids = prefix_inputs["input_ids"].to(device)
    attention_mask_full = full_inputs["attention_mask"].to(device)
    pixel_values = (
        prefix_inputs["pixel_values"].to(device)
        if "pixel_values" in prefix_inputs
        else None
    )
    image_grid_thw = (
        prefix_inputs["image_grid_thw"].to(device)
        if "image_grid_thw" in prefix_inputs
        else None
    )

    # Allocate the absolute-layout output tensor on CPU; we copy each slice
    # over as it becomes available.
    hidden = torch.zeros(
        (n_layers, seq_len, int(model.config.hidden_size) if hasattr(model.config, "hidden_size")
         else _infer_hidden_dim(model)),
        dtype=output_dtype,
    )

    # ---- prefill -------------------------------------------------------
    # We pass ``past_key_values=None`` so the model instantiates its own
    # cache exactly the way ``.generate`` does internally (and so we avoid
    # version-specific surprises around ``DynamicCache()``'s initial state).
    # ``cache_position`` is also omitted on prefill — the model derives it
    # from ``input_ids.shape[1]`` when no cache exists yet.
    prefill_attention_mask = attention_mask_full[:, :caption_start]

    with torch.no_grad():
        prefill_outputs = model(
            input_ids=prefix_input_ids,
            attention_mask=prefill_attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            past_key_values=None,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )

    _scatter_prefill_into(hidden, prefill_outputs.hidden_states, n_layers, caption_start)

    # Pick up the cache and the mRoPE deltas that the prefill computed so
    # we can roll them forward across the per-token steps.
    past_key_values = prefill_outputs.past_key_values
    rope_deltas = getattr(prefill_outputs, "rope_deltas", None)

    # ---- generation steps ---------------------------------------------
    # Feed exactly one new token per call. The model uses cache_position +
    # rope_deltas (captured above) to derive the mRoPE-correct position_ids
    # internally; we do NOT assemble position_ids by hand.
    running_attention = prefill_attention_mask

    for j in range(caption_len):
        abs_pos = caption_start + j
        new_token = full_ids[abs_pos].view(1, 1).to(device)
        running_attention = torch.cat(
            [
                running_attention,
                torch.ones((1, 1), dtype=running_attention.dtype, device=device),
            ],
            dim=1,
        )
        cache_position_step = torch.tensor([abs_pos], device=device)

        step_kwargs: dict[str, Any] = {
            "input_ids": new_token,
            "attention_mask": running_attention,
            "past_key_values": past_key_values,
            "cache_position": cache_position_step,
            "use_cache": True,
            "output_hidden_states": True,
            "return_dict": True,
        }
        if rope_deltas is not None:
            step_kwargs["rope_deltas"] = rope_deltas

        with torch.no_grad():
            step_outputs = model(**step_kwargs)

        _scatter_step_into(hidden, step_outputs.hidden_states, n_layers, abs_pos)
        past_key_values = step_outputs.past_key_values
        # rope_deltas is invariant per (image, prompt) — no need to refresh.

    # ---- pack the result mirroring run_teacher_forcing ----------------
    input_ids_cpu = full_ids.to(device="cpu", dtype=torch.int64).contiguous()
    attention_mask_out: torch.Tensor | None = None
    full_attn_full = full_inputs.get("attention_mask")
    if full_attn_full is not None:
        attention_mask_out = (
            full_attn_full[0].to(device="cpu", dtype=torch.int8).contiguous()
        )

    return HiddenStatesResult(
        hidden_states=hidden.contiguous(),
        input_ids=input_ids_cpu,
        caption_start=caption_start,
        caption_len=caption_len,
        attention_mask=attention_mask_out,
        metadata={
            "model_id": getattr(model_wrapper, "model_id", "unknown"),
            "hidden_dim": int(hidden.shape[-1]),
            "n_layers": int(hidden.shape[0]),
            "collection_method": "forced_decoding",
            "sparc_active": sparc_buffer is not None,
            "timestamp_iso": _iso_now(),
        },
    )


# ---------------------------------------------------------------- helpers


def _infer_hidden_dim(model: torch.nn.Module) -> int:
    """Fallback hidden_dim probe — tries text_config then nested model.config."""
    for path in ("text_config.hidden_size", "hidden_size", "model.config.hidden_size"):
        obj: Any = model.config
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        if isinstance(obj, int) and obj > 0:
            return obj
    raise RuntimeError("Could not infer hidden_dim from model.config.")


def _scatter_prefill_into(
    hidden: torch.Tensor,
    hidden_states_tuple: tuple,
    n_layers: int,
    caption_start: int,
) -> None:
    """Copy the prefill ``hidden_states[1:]`` into ``hidden[:, 0:caption_start, :]``.

    Per the HuggingFace convention, ``outputs.hidden_states`` is a tuple of
    length ``n_layers + 1`` where index 0 is the embedding output. We drop it
    (the metrics pipeline indexes layers in ``[1, L]``).
    """
    if len(hidden_states_tuple) - 1 != n_layers:
        raise RuntimeError(
            f"Prefill returned {len(hidden_states_tuple) - 1} layer states, "
            f"expected n_layers={n_layers}."
        )
    for layer_idx in range(n_layers):
        # hidden_states_tuple[layer_idx + 1] shape: (1, caption_start, hidden_dim)
        layer_state = hidden_states_tuple[layer_idx + 1].squeeze(0)
        if layer_state.shape[0] != caption_start:
            raise RuntimeError(
                f"Prefill layer {layer_idx} has seq_len={layer_state.shape[0]} "
                f"but expected caption_start={caption_start}."
            )
        hidden[layer_idx, :caption_start, :] = layer_state.to(
            dtype=hidden.dtype, device=hidden.device, copy=True
        )


def _scatter_step_into(
    hidden: torch.Tensor,
    hidden_states_tuple: tuple,
    n_layers: int,
    abs_pos: int,
) -> None:
    """Copy a single-token step's ``hidden_states[1:]`` into ``hidden[:, abs_pos, :]``."""
    if len(hidden_states_tuple) - 1 != n_layers:
        raise RuntimeError(
            f"Step returned {len(hidden_states_tuple) - 1} layer states, "
            f"expected n_layers={n_layers}."
        )
    for layer_idx in range(n_layers):
        # Shape: (1, 1, hidden_dim)
        layer_state = hidden_states_tuple[layer_idx + 1]
        if layer_state.shape[1] != 1:
            raise RuntimeError(
                f"Step layer {layer_idx} produced {layer_state.shape[1]} positions, "
                "expected exactly 1 (single forced token)."
            )
        hidden[layer_idx, abs_pos, :] = layer_state[0, 0, :].to(
            dtype=hidden.dtype, device=hidden.device, copy=True
        )
