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


def detect_model_family(model) -> str:
    """Return ``"qwen"`` or ``"llama"`` for the SPARC forward dispatch.

    Primary signal is the top-level model class name. As a fallback
    (covers test mocks and any future wrapper that doesn't match the
    markers), we look at which inner attribute holds the decoder:

        model.model.language_model → qwen (Qwen2.5-VL convention)
        model.model.text_model     → llama (Idefics3 / SmolVLM convention)
    """
    cls = type(model).__name__
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
        f"Unknown model family for {cls}. Add a marker in _QWEN_MARKERS / "
        f"_LLAMA_MARKERS in utils/attn.py or pass family= explicitly to "
        f"add_custom_attention_layers."
    )


def decoder_of(model) -> object:
    """Return the decoder module (the one with ``.layers``) for SPARC patching.

    Qwen : model.model.language_model
    Llama: model.model.text_model
    Fallback: model.model (some checkpoints flatten the hierarchy)
    """
    inner = getattr(model, "model", model)
    for attr in ("language_model", "text_model"):
        candidate = getattr(inner, attr, None)
        if candidate is not None and hasattr(candidate, "layers"):
            return candidate
    if hasattr(inner, "layers"):
        return inner
    raise AttributeError(
        f"Could not find decoder.layers on {type(model).__name__}. "
        "Looked at model.model.language_model and model.model.text_model."
    )

logger = logging.getLogger(__name__)


class SelectedIndexBuffer:
    def __init__(self):
        self.indices1 = (
            []
        )  # Buffer to store selected token indices from the current token generation step
        self.indices2 = (
            []
        )  # Buffer to store selected token indices from the previous token generation step
        self.input_len = 0  # Length of the input sequence
        self.num_image_patches = None

    def update_indices1(self, indices, image_token_index):
        """
        Updates indices1 with the selected token indices from a specific layer (e.g., the 20th layer).
        This is called during the current token generation to store new selections.
        The image_token_index offset is applied to adjust for the position of image tokens.
        """
        indices = indices + image_token_index
        indices = indices.squeeze(dim=-1)
        self.indices1 = indices

    def update_indices2(self):
        """
        Copies indices1 to indices2.
        Called at the beginning of a new token generation step so that calibration can use
        the indices selected from the previous step.
        """
        self.indices2 = self.indices1

    def update_input_len(self, len):
        self.input_len = len

    def reset(self):
        self.indices1 = []
        self.indices2 = []
        self.input_len = 0
        self.num_image_patches = None

    def calibrate(self, value, alpha):
        """
        Applies calibration to token representations using indices from the previous step (indices2).
        The calibration scales selected token positions by a factor alpha.
        """
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
            indices_buffer.update_indices1(indices, image_token_index)

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
            indices_buffer.update_indices1(indices, image_token_index)

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
    "qwen":  forward_qwen25vl,
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
        layer.self_attn.forward = MethodType(forward_, layer.self_attn)
