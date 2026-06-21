import torch
from types import MethodType
from functools import partial
from typing import Dict, Any, Optional, Tuple
import torch.nn.functional as F
import torch.nn as nn
import logging
from dataclasses import dataclass
import math
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    rotate_half,
    repeat_kv,
)
from transformers.cache_utils import Cache


# ------------------------------------------------------------------------
# Family detection — kept as a single-knob abstraction even though the
# study is now LLaVA-1.5-only. Returns "llama" for any LLaVA / Llama-
# backbone model. The Qwen-2.5-VL branch was removed together with the
# Qwen wrapper in Block 2 of the migration; ``forward_qwen25vl`` is
# gone, and the only registered family in ``_FORWARD_BY_FAMILY`` is
# ``"llama"``.
# ------------------------------------------------------------------------


_LLAMA_MARKERS = ("Llava", "Llama", "Idefics3", "SmolVLM", "Mllama")


def detect_model_family(model) -> str:
    """Return ``"llama"`` for any supported model.

    Post-Block-2, only the llama family (LLaVA-1.5 + the historical
    SmolVLM / Idefics3 markers, kept as a safety net for mocks) is
    supported. If a future block adds another backbone, extend this
    function AND add an entry to ``_FORWARD_BY_FAMILY``.
    """
    cls = type(model).__name__
    if any(cls.startswith(m) for m in _LLAMA_MARKERS):
        return "llama"
    # Attribute-based fallback for test mocks / unusual wrappers — any
    # decoder we can reach via decoder_of belongs to the llama family.
    try:
        decoder_of(model)
        return "llama"
    except AttributeError as exc:
        raise ValueError(
            f"Unknown model family for {cls}. Add a marker in _LLAMA_MARKERS "
            f"in utils/attn.py or pass family= explicitly to "
            f"add_custom_attention_layers."
        ) from exc


def decoder_of(model) -> object:
    """Return the decoder module (the one with ``.layers``) for SPARC patching.

    Layouts covered:

    * ``model.model.language_model``  — LLaVA-1.5 in transformers 5.x
      (LlavaModel wraps a LlamaModel directly).
    * ``model.language_model.model``  — older LLaVA layouts where
      ``language_model`` is the full ``LlamaForCausalLM`` and the
      decoder is one level deeper.
    * ``model.model.text_model``      — Idefics3 / SmolVLM convention
      (still listed for backward-compat test mocks; the wrapper itself
      was retired in Block 2).
    * ``model.model``                 — flattened checkpoints.
    """
    inner = getattr(model, "model", model)
    # Direct paths under inner.
    for attr in ("language_model", "text_model"):
        candidate = getattr(inner, attr, None)
        if candidate is None:
            continue
        if hasattr(candidate, "layers"):
            return candidate
        # One nesting level deeper — covers
        # ``LlavaForConditionalGeneration.language_model.model.layers``.
        nested = getattr(candidate, "model", None)
        if nested is not None and hasattr(nested, "layers"):
            return nested
    if hasattr(inner, "layers"):
        return inner
    raise AttributeError(
        f"Could not find decoder.layers on {type(model).__name__}. "
        "Looked at model.{language_model,text_model}[.model] and model.model."
    )

logger = logging.getLogger(__name__)


class SelectedIndexBuffer:
    def __init__(self):
        # Selected token indices from the current generation step (global
        # positions in the input_ids — already translated through
        # image_positions).
        self.indices1 = []
        # Selected indices from the PREVIOUS generation step (used by
        # ``calibrate`` to scale ``value_cache`` at those positions).
        self.indices2 = []
        self.input_len = 0  # prompt length excluding image patches
        self.num_image_patches = None
        # Explicit list of global positions in input_ids where
        # input_ids[p] == image_token_id. Replaces the implicit
        # [image_token_index, image_token_index + num_image_patches) block
        # the original SPARC code assumed — that assumption is wrong for
        # Idefics3 / SmolVLM, whose image patches are interleaved with
        # row/column separator tokens (<fake_token_around_image>,
        # <row_X_col_Y>, ...). For Qwen (contiguous block), this list is
        # exactly [idx, idx+1, ..., idx+N-1], so the per-id mask is a
        # drop-in replacement that yields identical results.
        # 1-D LongTensor (on any device — moved to attn device on use).
        self.image_positions = None

    def update_image_positions(self, positions):
        """Store the per-image list of global positions where input_ids == image_token_id.

        Must be called BEFORE the prefill of each image (the per-image SPARC
        bookkeeping is reset + update_input_len + update_image_positions).
        """
        self.image_positions = positions

    def update_indices1(self, indices, image_token_index=None):
        """Store the (translated, global) selected indices in ``indices1``.

        ``indices`` is the tensor of LOCAL positions inside the image-token
        block returned by ``(ratio >= tau).nonzero()`` — values in
        ``[0, num_image_patches)``. We translate to GLOBAL input_ids
        positions:

        * If ``self.image_positions`` is set (the unified, layout-correct
          path) → ``self.indices1 = image_positions[indices_squeezed]``.
          Works for both contiguous (Qwen) and interleaved (SmolVLM) layouts.
        * Otherwise (back-compat) → ``self.indices1 = indices + image_token_index``,
          which only works when image tokens form a contiguous block.
        """
        local = indices.squeeze(dim=-1)
        if self.image_positions is not None:
            self.indices1 = self.image_positions.to(local.device)[local]
        else:
            if image_token_index is None:
                raise ValueError(
                    "update_indices1 needs either image_positions set on the "
                    "buffer (the unified path) or an explicit image_token_index "
                    "(the legacy contiguous-block path)."
                )
            self.indices1 = local + image_token_index

    def update_indices2(self):
        """Copy indices1 → indices2 at the start of each new generation step.

        Calibration in the current step uses indices selected at the
        previous step — that's how the value cache gets boosted at the
        SAME positions across consecutive steps.
        """
        self.indices2 = self.indices1

    def update_input_len(self, len):
        self.input_len = len

    def reset(self):
        self.indices1 = []
        self.indices2 = []
        self.input_len = 0
        self.num_image_patches = None
        self.image_positions = None

    def calibrate(self, value, alpha):
        """Scale value-cache rows at ``indices2`` by ``alpha`` (in place)."""
        if len(self.indices2) > 0:
            value[:, :, self.indices2] *= alpha

    def update_patch_num(self, num_image_patches):
        self.num_image_patches = num_image_patches


# Llama text-decoder attention (transformers 5.x).
# Used for Idefics3 / SmolVLM, which embed a standard Llama text model.
# Structurally identical to ``forward_qwen25vl`` below; the only difference
# is the rotary-embedding call — Llama uses 1D RoPE, Qwen-2.5-VL uses mRoPE.
def forward_llama(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    image_token_index: Optional[int] = None,
    alpha: Optional[float] = 1.0,
    beta: Optional[float] = 0.0,
    tau: Optional[float] = 2,
    selected: Optional[bool] = False,
    se_layers: Optional[Tuple[int, int]] = None,
    indices_buffer: Optional[SelectedIndexBuffer] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if self.layer_idx == 0:
        indices_buffer.update_indices2()

    gen_new_token = (
        past_key_values is not None and past_key_values.get_seq_length(self.layer_idx) > 0
    )

    if self.layer_idx >= se_layers[0] and self.layer_idx <= se_layers[1]:
        if len(indices_buffer.indices2) > 0:
            indices_buffer.calibrate(past_key_values.layers[self.layer_idx].values, alpha)

    if past_key_values is not None:
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx
        )

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )
    attn_weights = nn.functional.dropout(
        attn_weights, p=self.attention_dropout, training=self.training
    )

    if gen_new_token == False and self.layer_idx == 0:
        indices_buffer.update_patch_num(
            attn_weights.shape[-1] - indices_buffer.input_len
        )

    # Image-attention: either the per-id mask (correct for any layout —
    # both Qwen contiguous and Idefics3 interleaved) or the legacy
    # contiguous slice. The per-id mask is the only path SmolVLM should
    # take; it's also drop-in identical for Qwen since their image tokens
    # ARE contiguous, so image_positions = arange(idx, idx+N).
    if indices_buffer.image_positions is not None:
        ip = indices_buffer.image_positions.to(attn_weights.device)
        image_attention = attn_weights[:, :, -1, :].index_select(-1, ip).mean(dim=1)
    else:
        image_attention = attn_weights[
            :,
            :,
            -1,
            image_token_index : image_token_index + indices_buffer.num_image_patches,
        ].mean(dim=1)

    if gen_new_token:
        if selected:
            # mean by head dim
            ratio = (image_attention - self.image_attention) / self.image_attention
            ratio = ratio.squeeze(dim=0)
            indices = (ratio >= tau).nonzero()
            indices_buffer.update_indices1(indices, image_token_index=image_token_index)

    if not gen_new_token:
        self.image_attention = image_attention
    else:
        self.image_attention = (
            1 - beta
        ) * image_attention + beta * self.image_attention

    attn_output = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, -1)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights


_FORWARD_BY_FAMILY = {
    "llama": forward_llama,
}


def add_custom_attention_layers(
    model,
    alpha=1,
    beta=0,
    tau=2,
    selected_layer=15,
    se_layers=(0, 31),
    image_token_index=None,
    indices_buffer=None,
    family: Optional[str] = None,
):
    """Monkey-patch every decoder layer's ``self_attn.forward`` with SPARC.

    Args:
        family: ``"llama"`` or ``None`` to auto-detect from the model class.
            Post-Block-2 the study is LLaVA-1.5-only, so the only registered
            family is ``"llama"`` — kept as a knob so a future block can add
            another backbone without surgery here.
    """
    if family is None:
        family = detect_model_family(model)
    if family not in _FORWARD_BY_FAMILY:
        raise ValueError(f"Unknown SPARC family {family!r}. "
                         f"Known: {sorted(_FORWARD_BY_FAMILY)}.")

    forward_fn = _FORWARD_BY_FAMILY[family]
    decoder = decoder_of(model)
    for i, layer in enumerate(decoder.layers):
        selected = True if selected_layer == i else False
        forward_ = partial(
            forward_fn,
            alpha=alpha,
            beta=beta,
            tau=tau,
            selected=selected,
            se_layers=se_layers,
            image_token_index=image_token_index,
            indices_buffer=indices_buffer,
        )
        layer.self_attn.forward = MethodType(forward_, layer.self_attn)
