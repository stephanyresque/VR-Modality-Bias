"""SPARC's custom attention layers: per-family forward variants (Qwen mRoPE,
Llama 1D RoPE, legacy InternLM2 fused-QKV), the shared ``SelectedIndexBuffer``
state, and the ``add_custom_attention_layers`` monkey-patch installer.
"""

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
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    apply_multimodal_rotary_pos_emb,
)
from transformers.cache_utils import Cache


# ------------------------------------------------------------------------
# Family detection — looks at the top-level model class name to choose the
# right SPARC forward + the right path to the decoder layers.
#
# qwen   : Qwen2.5-VL (mRoPE, model.model.language_model.layers)
# llama  : Idefics3 / SmolVLM (standard 1D RoPE, model.model.text_model.layers)
# ------------------------------------------------------------------------


_QWEN_MARKERS = ("Qwen2_5_VL", "Qwen2VL", "QwenVL")
_LLAMA_MARKERS = ("Idefics3", "SmolVLM", "Llava", "Mllama")
# ``forward_internlm2`` (fused wqkv + wo) is only needed for the LEGACY
# remote-code path (``OpenGVLab/InternVL2-8B`` via trust_remote_code, class
# ``InternVLChatModel`` wrapping ``InternLM2ForCausalLM``). We are now on the
# NATIVE HF checkpoint ``OpenGVLab/InternVL3-8B-hf``
# (``InternVLForConditionalGeneration``), whose backbone is Qwen2.5 with
# SEPARATE ``q_proj``/``k_proj``/``v_proj`` -- so ``forward_llama`` /
# ``forward_qwen25vl`` are the right choices there, NOT
# ``forward_internlm2``. The InternLM2 forward is kept in-file for the case
# where someone re-visits the remote-code path in a compatible env.
_INTERNLM2_MARKERS = ("InternLM2",)  # narrow: exclude InternVL native class


def detect_model_family(model) -> str:
    """Return the SPARC family string for a model.

    Currently: ``"qwen"`` (mRoPE), ``"llama"`` (1D RoPE, separate q/k/v),
    or ``"internlm2"`` (1D RoPE, fused wqkv -- legacy InternVL2 remote path).

    For the NATIVE InternVL HF checkpoint
    (``InternVLForConditionalGeneration``), the backbone determines the
    forward: Qwen2.5-VL text-video backbone (``qwen2_5_vl`` model_type)
    => ``"qwen"``; plain Qwen2/Qwen2.5 text backbone (1D RoPE)
    => ``"llama"``; legacy InternLM2 backbone (fused wqkv, if ever)
    => ``"internlm2"``.

    Primary signal is the top-level model class name. As a fallback
    (covers test mocks and any future wrapper that doesn't match the
    markers), we look at which inner attribute holds the decoder.
    """
    cls = type(model).__name__
    if any(cls.startswith(m) for m in _INTERNLM2_MARKERS):
        return "internlm2"
    if cls.startswith("InternVL"):
        # Native InternVL HF: inspect the text backbone's model_type.
        text_cfg = getattr(getattr(model, "config", None), "text_config", None)
        inner_type = getattr(text_cfg, "model_type", None)
        if inner_type == "qwen2_5_vl":
            return "qwen"
        if inner_type == "internlm2":
            return "internlm2"
        # Default: plain Qwen2/Qwen2.5 text or Llama-style -- all 1D RoPE.
        return "llama"
    if any(cls.startswith(m) for m in _QWEN_MARKERS):
        return "qwen"
    if any(cls.startswith(m) for m in _LLAMA_MARKERS):
        return "llama"
    # Attribute-based fallback.
    inner = getattr(model, "model", model)
    if getattr(inner, "language_model", None) is not None:
        return "qwen"
    if getattr(inner, "text_model", None) is not None:
        return "llama"
    raise ValueError(
        f"Unknown model family for {cls}. Add a marker in one of "
        f"_INTERNLM2_MARKERS / _QWEN_MARKERS / _LLAMA_MARKERS in "
        f"utils/attn.py, or pass family= explicitly to "
        f"add_custom_attention_layers."
    )


def decoder_of(model) -> object:
    """Return the decoder module (the one with ``.layers``) for SPARC patching.

    Layouts covered:

    * Qwen 2.5-VL : ``model.model.language_model.layers``
    * Idefics3 / SmolVLM (llama family): ``model.model.text_model.layers``
      -- via the ``text_model`` branch below.
    * InternVL3-8B-hf (native, ``InternVLForConditionalGeneration``):
      ``model.model.language_model.layers`` -- the Qwen2/Qwen2.5 backbone
      IS the module with ``.layers``, so no extra nesting is needed.
    * InternVL2 (legacy remote code, kept as fallback):
      ``model.language_model.model.layers`` -- the extra nesting
      (``.language_model.model``) comes from ``InternLM2ForCausalLM``
      wrapping an inner ``.model``.
    * Flattened checkpoints: ``model.model.layers`` (fallback).
    """
    inner = getattr(model, "model", model)
    for attr in ("language_model", "text_model"):
        candidate = getattr(inner, attr, None)
        if candidate is None:
            continue
        if hasattr(candidate, "layers"):
            return candidate
        # One extra level of nesting -- InternLM2ForCausalLM has .model,
        # LlavaForCausalLM (legacy) also has .model.
        nested = getattr(candidate, "model", None)
        if nested is not None and hasattr(nested, "layers"):
            return nested
    if hasattr(inner, "layers"):
        return inner
    raise AttributeError(
        f"Could not find decoder.layers on {type(model).__name__}. "
        "Looked at model.{language_model,text_model}[.model] and model.model."
    )


def _attention_module_of(layer) -> object:
    """Return the layer's attention submodule.

    Different families put it under different attribute names:

        Qwen, LLaVA, SmolVLM, Idefics3 : ``layer.self_attn``
        InternLM2 (InternVL backbone)  : ``layer.attention``

    The dispatch here checks the well-known names in order; the SPARC
    ``add_custom_attention_layers`` uses this to patch the right thing
    without every wrapper hard-coding the attribute.
    """
    for cand in ("self_attn", "attention"):
        if hasattr(layer, cand):
            return getattr(layer, cand)
    raise AttributeError(
        f"decoder layer {type(layer).__name__} has neither `.self_attn` "
        f"nor `.attention` -- Step-0 inspection needed."
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


# InternLM2 attention (LEGACY InternVL2 remote-code backbone; the native
# InternVL3-8B-hf checkpoint uses a Qwen2.5 backbone with SEPARATE q/k/v_proj
# and therefore routes through forward_llama / forward_qwen25vl instead).
# Kept in-file for the case where someone re-visits the remote-code path
# in a compatible transformers version. Structurally identical to
# forward_llama EXCEPT:
#
#   * Fused Q/K/V:  InternLM2 has a single `self.wqkv` linear that emits
#                   (num_heads + 2 * num_kv_heads) * head_dim features per
#                   token. We split them by reshaping into
#                   (b, q, num_kv_heads, num_kv_groups + 2, head_dim) and
#                   slicing: the first `num_kv_groups` head-slots per group
#                   are Q heads, the second-to-last is K, the last is V.
#                   (Replicated from the official OpenGVLab InternLM2
#                   ``modeling_internlm2.py``.)
#
#   * Output projection: `self.wo` instead of `self.o_proj`.
#
#   * No `self.scaling` attribute assumed; we compute `1/sqrt(head_dim)`
#     inline. If the actual module DOES carry `self.scaling` (checked via
#     Step 0), we prefer it -- avoids fp-drift vs the unpatched forward.
#
#   * No `self.attention_dropout`; SPARC always runs in eval mode so we
#     just skip the dropout call.
#
# CRITICAL: the QKV split MUST reproduce the unpatched forward bit-for-bit
# when `alpha=1.0`. That's the whole point of ``scripts/internvl_
# exactness_gate.py`` -- run it before trusting this forward on real data.
# If the reshape is off (e.g. num_kv_groups swapped with num_kv_heads), the
# gate fails and this forward must NOT be used.
def forward_internlm2(
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

    # ---- fused QKV projection + split -------------------------------------
    # Layout convention (from OpenGVLab InternLM2 modeling code):
    #   wqkv(x): (B, T, num_kv_heads * (num_kv_groups + 2) * head_dim)
    # Reshape to (B, T, num_kv_heads, num_kv_groups + 2, head_dim), then:
    #   Q = qkv[..., :num_kv_groups, :]  reshape -> (B, T, num_heads, D)
    #   K = qkv[..., -2, :]              -> (B, T, num_kv_heads, D)
    #   V = qkv[..., -1, :]              -> (B, T, num_kv_heads, D)
    num_kv_heads = getattr(self, "num_key_value_heads")
    num_kv_groups = getattr(self, "num_key_value_groups")
    head_dim = getattr(self, "head_dim")

    qkv_states = self.wqkv(hidden_states)
    qkv_states = qkv_states.view(
        bsz, q_len, num_kv_heads, num_kv_groups + 2, head_dim,
    )
    query_states = qkv_states[..., :num_kv_groups, :]  # (B, T, num_kv_heads, num_kv_groups, D)
    query_states = query_states.reshape(bsz, q_len, num_kv_heads * num_kv_groups, head_dim)
    key_states = qkv_states[..., -2, :]   # (B, T, num_kv_heads, D)
    value_states = qkv_states[..., -1, :]

    # Transpose to (B, H, T, D) for the attention math.
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    # ---- rotary position embedding ---------------------------------------
    # Assumes the decoder layer forwards `position_embeddings=(cos, sin)`,
    # matching the transformers 5.x convention that Idefics3/Llama/Qwen2.5
    # already use. If Step 0 shows InternLM2 remote code passes something
    # different (e.g. bare position_ids), this call needs a variant.
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # ---- SPARC bookkeeping (identical to forward_llama) ------------------
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

    key_states = repeat_kv(key_states, num_kv_groups)
    value_states = repeat_kv(value_states, num_kv_groups)

    # ---- scaled dot-product attention ------------------------------------
    scaling = getattr(self, "scaling", 1.0 / math.sqrt(head_dim))
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )
    # Note: no dropout -- SPARC always runs eval-mode.

    if gen_new_token == False and self.layer_idx == 0:
        indices_buffer.update_patch_num(
            attn_weights.shape[-1] - indices_buffer.input_len
        )

    # ---- image_attention + selection + smoothing (identical to llama) ----
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
            ratio = (image_attention - self.image_attention) / self.image_attention
            ratio = ratio.squeeze(dim=0)
            indices = (ratio >= tau).nonzero()
            indices_buffer.update_indices1(indices, image_token_index=image_token_index)

    if not gen_new_token:
        self.image_attention = image_attention
    else:
        self.image_attention = (1 - beta) * image_attention + beta * self.image_attention

    # ---- output ----------------------------------------------------------
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, -1)
    attn_output = self.wo(attn_output)   # <-- wo, not o_proj

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights


# qwen2.5-vl attention (current transformers Cache/rotary API)
def forward_qwen25vl(
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
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.config.rope_parameters["mrope_section"]
    )

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

    # Image-attention via per-id mask (correct for any layout). See
    # forward_llama above for the rationale — Qwen is contiguous so this
    # is identical to the legacy slice; Idefics3 has interleaved
    # separators, so the mask is the only correct option.
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
    "qwen":     forward_qwen25vl,
    "llama":    forward_llama,
    "internlm2": forward_internlm2,
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
        family: ``"qwen"``, ``"llama"``, or ``None`` to auto-detect from the
            model class. The Qwen variant rotates Q/K via mRoPE; the Llama
            variant uses standard 1D RoPE.
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
        # Family-agnostic attribute lookup: `.self_attn` for qwen / llava /
        # smolvlm / idefics3; `.attention` for InternLM2 (InternVL). See
        # `_attention_module_of` above.
        attn_module = _attention_module_of(layer)
        attn_module.forward = MethodType(forward_, attn_module)
