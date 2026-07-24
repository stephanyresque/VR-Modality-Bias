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
        # ---- adaptive-intensity state (unused when adaptive=False) ----
        # LOCAL counterparts of indices1 / indices2, i.e. offsets into
        # image_positions rather than into input_ids. The adaptive registry is
        # indexed locally; the cache write still uses the global positions.
        self.indices1_local = []
        self.indices2_local = []
        # Target amplification factor decided at the current step (target1) and
        # the one actually applied to the cache (target2). Same one-step delay
        # as indices1 -> indices2.
        self.target1 = 1.0
        self.target2 = 1.0
        # Factor each image position currently carries in the value cache.
        # float32, length == len(image_positions). Allocated lazily by
        # update_image_positions (the patch count is unknown before it).
        self.accum_factors = None
        # Per-position ratio target/accumulated resolved once per step; None
        # when the cache already carries the right factors.
        self.correction = None
        # ---- question-conditioned selection state (unused when qcond=False) ----
        # Global prompt positions strictly after the last image position: the
        # question text plus the trailing chat-template tokens. Their attention
        # rows over the image columns are the relevance signal.
        # 1-D LongTensor (on any device), None by default.
        self.question_positions = None
        # Instrumentation only: the prefill top-k, as LOCAL offsets on CPU, so a
        # script can log which visual tokens were picked per question (the
        # sink-contamination check) without re-reading the attention.
        self.prefill_selected_local = None
        # ---- conserved-reinforcement state (Ponto 3; unused when conserve=False) ----
        # Visual sinks frozen at the prefill, exactly like the qcond selection:
        # LOCAL offsets into image_positions and their GLOBAL input_ids positions.
        self.sink_local = []
        self.sink_positions = None
        # Instrumentation only: sink count, mean raw question attention on the
        # sinks at the prefill, and the mass moved at the last decode step.
        self.n_sinks = 0
        self.prefill_sink_mass = 0.0
        self.reallocated_mass = 0.0

    def update_question_positions(self, positions):
        """Store the global prompt positions that carry the question.

        Must be called BEFORE the prefill when ``qcond`` is on, alongside
        ``update_image_positions``.
        """
        self.question_positions = positions

    def update_image_positions(self, positions):
        """Store the per-image list of global positions where input_ids == image_token_id.

        Must be called BEFORE the prefill of each image (the per-image SPARC
        bookkeeping is reset + update_input_len + update_image_positions).

        Also (re)allocates the adaptive registry at 1.0. This is the earliest
        point where the number of image patches is known, so ``reset`` can only
        drop the registry to None and the allocation has to happen here.
        """
        self.image_positions = positions
        self.accum_factors = torch.ones(
            positions.numel(), dtype=torch.float32, device=positions.device
        )
        self.correction = None

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
        self.indices1_local = local
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

    def update_target1(self, target):
        """Store the adaptive target factor decided at the current step."""
        self.target1 = float(target)

    def update_indices2(self):
        """Copy indices1 → indices2 at the start of each new generation step.

        Calibration in the current step uses indices selected at the
        previous step — that's how the value cache gets boosted at the
        SAME positions across consecutive steps.
        """
        self.indices2 = self.indices1
        self.indices2_local = self.indices1_local
        self.target2 = self.target1

    def update_input_len(self, len):
        self.input_len = len

    def reset(self):
        self.indices1 = []
        self.indices2 = []
        self.input_len = 0
        self.num_image_patches = None
        self.image_positions = None
        self.indices1_local = []
        self.indices2_local = []
        self.target1 = 1.0
        self.target2 = 1.0
        self.accum_factors = None
        self.correction = None
        self.question_positions = None
        self.prefill_selected_local = None
        self.sink_local = []
        self.sink_positions = None
        self.n_sinks = 0
        self.prefill_sink_mass = 0.0
        self.reallocated_mass = 0.0

    def update_sinks(self, sink_local):
        """Freeze the prefill visual sinks (Ponto 3).

        Stores the LOCAL offsets and translates them to GLOBAL input_ids
        positions through ``image_positions`` — the same local->global path
        ``update_indices1`` uses for the selection, so the reallocation and the
        value-cache calibration index the same columns.
        """
        self.sink_local = sink_local
        if self.image_positions is not None:
            self.sink_positions = self.image_positions.to(sink_local.device)[sink_local]
        else:
            self.sink_positions = sink_local
        self.n_sinks = int(sink_local.numel())

    def calibrate(self, value, alpha):
        """Scale value-cache rows at ``indices2`` by ``alpha`` (in place)."""
        if len(self.indices2) > 0:
            value[:, :, self.indices2] *= alpha

    def prepare_adaptive_correction(self):
        """Resolve this step's correction ratio and advance the registry.

        Must run once per step (layer 0), NOT once per layer: every layer in
        the ``se_layers`` window carries the same accumulated factor, so a
        per-layer recompute would leave ``accum_factors == target`` after the
        first layer and silently skip all the others.

        Positions in ``indices2`` (the previous step's selection, same one-step
        delay the alpha^c path uses) target ``target2``; every other image
        position targets 1.0, which is what makes the intensity relax when a
        token leaves the selection or the anchoring recovers.
        """
        if self.image_positions is None or self.accum_factors is None:
            raise RuntimeError(
                "Adaptive SPARC needs update_image_positions() before the "
                "prefill: the accumulated-factor registry is indexed by the "
                "image-token positions and cannot be sized without them."
            )
        target = torch.ones_like(self.accum_factors)
        if len(self.indices2_local) > 0:
            target[self.indices2_local.to(target.device)] = float(self.target2)
        if torch.equal(target, self.accum_factors):
            self.correction = None
            return
        self.correction = target / self.accum_factors
        self.accum_factors = target

    def commit_prefill_factor(self):
        """Record that the prefill already wrote ``target1`` at the selected rows.

        The post-reference layers of the prefill multiply their value states by
        ``target1`` before the cache write, so the registry has to agree, or the
        decode-step correction would apply the factor a second time.

        A consequence worth spelling out: at decode step 1, ``update_indices2``
        copies the prefill selection and ``target2 = target1``, so
        ``prepare_adaptive_correction`` finds ``target == accum_factors`` and
        resolves a null correction. Nothing is written at step 1, which is
        correct: the prefill already applied that factor. From step 2 on, the
        Point-1 adaptive target takes over and the correction is the ratio
        against the prefill factor.
        """
        if self.accum_factors is None:
            raise RuntimeError(
                "commit_prefill_factor needs the registry: call "
                "update_image_positions() before the prefill."
            )
        if len(self.indices1_local) == 0:
            raise RuntimeError(
                "commit_prefill_factor called with an empty prefill selection; "
                "question_conditioned_selection never returns one."
            )
        local = self.indices1_local.to(self.accum_factors.device)
        self.accum_factors[local] = float(self.target1)

    def calibrate_adaptive(self, value):
        """Rescale value-cache rows to this step's target factor (in place).

        Multiplies by ``target / accumulated`` rather than by ``alpha``, so the
        factor a position carries is always exactly the step's target, bounded
        by the ceiling, instead of the product over every step it was selected.
        """
        if self.correction is None:
            return
        positions = self.image_positions.to(value.device)
        correction = self.correction.to(device=value.device, dtype=value.dtype)
        value[:, :, positions] *= correction.view(-1, 1)

    def update_patch_num(self, num_image_patches):
        self.num_image_patches = num_image_patches


def adaptive_target_factor(image_attention, reference, lam, ceiling) -> float:
    """Return ``min(1 + lam * deficit, ceiling)`` for the current step.

    ``deficit`` is the relative drop of the current visual attention against
    the moving-average reference (``self.image_attention`` BEFORE its update),
    truncated at zero: attention at or above the reference means no deficit.

    Computed in float32. ``attn_weights`` is downcast to the model dtype right
    after the softmax, and the deficit is a small difference between two small
    numbers; the upcast is local to this signal and never touches the attention
    path.

    A non-finite target (a zero or degenerate reference makes the division NaN)
    returns 1.0, the neutral factor. Unlike the selection ``ratio``, where a NaN
    only spoils one step, a NaN target would be written into ``accum_factors``
    and would poison the value cache for the rest of the generation.
    """
    current = image_attention.detach().float().mean()
    ref = reference.detach().float().mean()
    deficit = torch.clamp((ref - current) / ref, min=0.0)
    target = torch.clamp(1.0 + float(lam) * deficit, max=float(ceiling))
    if not bool(torch.isfinite(target)):
        return 1.0
    return float(target)


def _rows_to_image_attention(attn_weights, image_positions, row_positions):
    """Mean attention the given rows pay each image column, averaged over heads.

    Returns a ``[n_image]`` float32 vector. Batch is squeezed: the rest of the
    SPARC code already assumes bsz == 1 (see the ``ratio.squeeze(dim=0)`` in the
    forwards). An empty row set yields zeros, which is the neutral background.
    """
    n_image = image_positions.numel()
    if row_positions.numel() == 0:
        return torch.zeros(n_image, dtype=torch.float32, device=attn_weights.device)
    rows = attn_weights.index_select(2, row_positions.to(attn_weights.device))
    block = rows.index_select(3, image_positions.to(attn_weights.device))
    return block[0].detach().float().mean(dim=(0, 1))


def _background_positions(attn_weights, question_positions) -> torch.Tensor:
    """Every prompt row that is NOT part of the question."""
    n_rows = attn_weights.shape[2]
    mask = torch.ones(n_rows, dtype=torch.bool, device=attn_weights.device)
    mask[question_positions.to(attn_weights.device)] = False
    return mask.nonzero(as_tuple=True)[0]


def question_conditioned_selection(
    attn_weights, image_positions, question_positions, qtop_frac
) -> torch.Tensor:
    """Top-k visual tokens by CONTRASTIVE question relevance, at prefill.

    Relevance of visual token j is the attention the question rows pay it minus
    the attention every other prompt row pays it, truncated at zero. The
    subtraction is what removes the visual sinks: a sink is attended by any text
    row out of activation habit, so it cancels, while a region the question
    specifically looks at survives. Raw question attention (the v1 signal) is
    dominated by sinks: Jaccard 0.81 between selections for different questions
    on the same image, one fixed position in 92% of them.

    Returns the LOCAL offsets into ``image_positions`` of the top
    ``k = max(1, floor(qtop_frac * N))``, in decreasing relevance. Top-k rather
    than a threshold: it can never come back empty.

    Fallback: if the contrast is zero everywhere (degenerate case, e.g. every
    row attends the image identically), rank by the raw question attention
    instead. Never an empty selection.

    Non-finite relevances are zeroed before the top-k. A NaN would otherwise
    win or lose the ranking arbitrarily depending on the sort, and the picked
    positions then feed the value cache for the whole generation.
    """
    question_relevance = _rows_to_image_attention(
        attn_weights, image_positions, question_positions
    )
    background_relevance = _rows_to_image_attention(
        attn_weights, image_positions, _background_positions(attn_weights, question_positions)
    )
    question_relevance = torch.nan_to_num(question_relevance, nan=0.0, posinf=0.0, neginf=0.0)
    background_relevance = torch.nan_to_num(
        background_relevance, nan=0.0, posinf=0.0, neginf=0.0
    )

    relevance = torch.clamp(question_relevance - background_relevance, min=0.0)
    if not bool(torch.any(relevance > 0)):
        logger.debug(
            "Contrastive relevance is zero everywhere; falling back to the raw "
            "question attention for the top-k."
        )
        relevance = question_relevance

    k = max(1, int(math.floor(float(qtop_frac) * image_positions.numel())))
    return torch.topk(relevance, k).indices


def question_conditioned_sinks(
    attn_weights, image_positions, question_positions, sink_frac, selected_local
) -> torch.Tensor:
    """Local offsets of the visual SINKS at the prefill (Ponto 3).

    A sink is a visual position with zero contrastive relevance (attended out of
    habit by any text row, not preferred by the question) AND raw question
    attention in the top ``sink_frac`` of the visual positions. That is the
    operational signature the Point-2 finding measured.

    The qcond selection is subtracted so sinks and the selection are disjoint by
    construction: ``question_conditioned_selection`` returns exactly k indices
    via ``topk``, so it can pad the selection with zero-contrast columns that
    would otherwise also match the sink rule. This reuses the exact building
    blocks of that function and never calls it, so the selection itself stays
    byte-identical.

    Returns the LOCAL offsets into ``image_positions``; ``[]`` (an empty
    LongTensor) is a valid result and makes the reallocation a no-op.
    """
    question_relevance = _rows_to_image_attention(
        attn_weights, image_positions, question_positions
    )
    background_relevance = _rows_to_image_attention(
        attn_weights, image_positions, _background_positions(attn_weights, question_positions)
    )
    question_relevance = torch.nan_to_num(question_relevance, nan=0.0, posinf=0.0, neginf=0.0)
    background_relevance = torch.nan_to_num(
        background_relevance, nan=0.0, posinf=0.0, neginf=0.0
    )
    contrast = torch.clamp(question_relevance - background_relevance, min=0.0)

    n_image = image_positions.numel()
    k = max(1, int(math.floor(float(sink_frac) * n_image)))
    top_raw = torch.topk(question_relevance, k).indices
    is_top_raw = torch.zeros(n_image, dtype=torch.bool, device=attn_weights.device)
    is_top_raw[top_raw] = True

    is_selected = torch.zeros(n_image, dtype=torch.bool, device=attn_weights.device)
    if selected_local is not None and selected_local.numel() > 0:
        is_selected[selected_local.to(is_top_raw.device)] = True

    sink_mask = (contrast == 0) & is_top_raw & ~is_selected
    return sink_mask.nonzero(as_tuple=True)[0]


def conserve_reallocation(attn_row, sink_cols, target_cols, rho, eps: float = 1e-12):
    """Reallocate visual sink attention to the selected visual columns (Ponto 3).

    Operates per head on the last post-softmax attention row. Each sink loses the
    same fraction ``rho`` of its own mass; the pooled budget is added to the
    selected columns in proportion to the head's current attention on them, or
    uniformly when that attention is ~0. Text columns and unselected visual
    columns are never touched, so the row sum is preserved by construction.

    float32 local math, returned in the input dtype (the Point-1 discipline).
    Exact short-circuit with no arithmetic when ``rho == 0``, the sink set is
    empty, or the selection is empty: the input tensor is returned unchanged.
    Callers guarantee ``sink_cols`` and ``target_cols`` are disjoint.
    """
    if (
        rho == 0
        or sink_cols is None
        or len(sink_cols) == 0
        or target_cols is None
        or len(target_cols) == 0
    ):
        return attn_row

    orig_dtype = attn_row.dtype
    row = attn_row.to(torch.float32)
    last = row.dim() - 1
    sink_cols = sink_cols.to(row.device)
    target_cols = target_cols.to(row.device)

    sink_vals = row.index_select(last, sink_cols)
    removed = float(rho) * sink_vals
    budget = removed.sum(dim=last, keepdim=True)

    target_vals = row.index_select(last, target_cols)
    target_sum = target_vals.sum(dim=last, keepdim=True)
    n_target = target_cols.numel()
    add_proportional = budget * (target_vals / target_sum.clamp(min=eps))
    add_uniform = (budget / n_target).expand_as(target_vals)
    add = torch.where(target_sum <= eps, add_uniform, add_proportional)

    # Out-of-place so the input row is never mutated (identity holds on
    # short-circuit, and the caller decides whether to write back).
    row = row.index_copy(last, sink_cols, sink_vals - removed)
    row = row.index_copy(last, target_cols, target_vals + add)
    return row.to(orig_dtype)


def prefill_target(lam, ceiling) -> float:
    """The prefill reinforcement factor: ``min(1 + lam, ceiling)``, unconditional.

    Point 1's deficit gate has no analogue inside the prefill: there is no
    "before" for a moving reference to compare against, and the measured prefill
    deficit was ~0 (targets 1.0000 to 1.0008). The empirical justification that
    the deficit exists at answer time is already on record from Point 1, so the
    prefill applies the full intensity rather than inventing a reference. The
    adaptive gate stays intact for the decode steps, where history exists.

    ``lam=0`` still resolves to exactly 1.0, so the neutrality gate survives.
    """
    return min(1.0 + float(lam), float(ceiling))


def _applies_prefill_boost(
    layer_idx, gen_new_token, qcond, adaptive, selected_layer, se_layers, indices_buffer
) -> bool:
    """Whether this layer must scale its value states during the prefill.

    Shared by both forwards rather than inlined twice: the predicate has six
    terms and a silent divergence between the Llama and Qwen copies would be
    invisible until an evaluation run.

    Only the layers strictly above the reference layer qualify. The reference
    layer decides the selection from its own attention, which is computed after
    its value states are already committed, so it cannot boost itself.
    """
    return (
        not gen_new_token
        and qcond
        and adaptive
        and selected_layer is not None
        and layer_idx > selected_layer
        and layer_idx <= se_layers[1]
        and indices_buffer.prefill_selected_local is not None
    )


def _boost_prefill_values(value_states, indices_buffer) -> None:
    """Scale the selected visual rows of ``value_states`` in place, at prefill.

    Runs BEFORE ``past_key_values.update`` and before the attention matmul, so
    the factor lands both in the cache and in the attention output of the last
    prompt position, which is where the first answer token is decided. That
    position is never reached by the decode-step calibration, which only fires
    when ``gen_new_token`` is true.
    """
    positions = indices_buffer.indices1.to(value_states.device)
    factor = torch.as_tensor(
        indices_buffer.target1, dtype=value_states.dtype, device=value_states.device
    )
    value_states[:, :, positions] *= factor


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
    adaptive: Optional[bool] = False,
    lam: Optional[float] = 0.0,
    ceiling: Optional[float] = 2.0,
    qcond: Optional[bool] = False,
    qtop_frac: Optional[float] = 0.05,
    selected_layer: Optional[int] = None,
    conserve: Optional[bool] = False,
    rho: Optional[float] = 0.5,
    sink_frac: Optional[float] = 0.05,
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
        if adaptive:
            indices_buffer.prepare_adaptive_correction()

    gen_new_token = (
        past_key_values is not None and past_key_values.get_seq_length(self.layer_idx) > 0
    )

    if self.layer_idx >= se_layers[0] and self.layer_idx <= se_layers[1]:
        if adaptive:
            # Relaxation has to run on steps where nothing is selected, so the
            # guard is the pending correction, not the emptiness of indices2.
            if indices_buffer.correction is not None:
                indices_buffer.calibrate_adaptive(
                    past_key_values.layers[self.layer_idx].values
                )
        elif len(indices_buffer.indices2) > 0:
            indices_buffer.calibrate(past_key_values.layers[self.layer_idx].values, alpha)

    if _applies_prefill_boost(
        self.layer_idx, gen_new_token, qcond, adaptive, selected_layer,
        se_layers, indices_buffer,
    ):
        _boost_prefill_values(value_states, indices_buffer)

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
            # With qcond the prefill top-k is frozen for the whole generation:
            # the question does not change, so neither does what is relevant.
            if not qcond:
                ratio = (image_attention - self.image_attention) / self.image_attention
                ratio = ratio.squeeze(dim=0)
                indices = (ratio >= tau).nonzero()
                indices_buffer.update_indices1(indices, image_token_index=image_token_index)
            if adaptive:
                indices_buffer.update_target1(
                    adaptive_target_factor(
                        image_attention, self.image_attention, lam, ceiling
                    )
                )
    elif selected and qcond and adaptive:
        # Prefill, reference layer. The selection and the target decided here are
        # consumed by the layers ABOVE, still inside this same prefill forward:
        # the answer token of a VQA prompt comes out of the prefill logits, a
        # forward the decode-step calibration never touches.
        if indices_buffer.question_positions is None:
            raise RuntimeError(
                "Question-conditioned SPARC needs update_question_positions() "
                "before the prefill: the relevance signal is read from the "
                "attention rows of the question and cannot be located without "
                "them."
            )
        # image_positions is guaranteed here: adaptive is on, so layer 0 already
        # ran prepare_adaptive_correction, which raises without it.
        image_pos = indices_buffer.image_positions
        qp = indices_buffer.question_positions
        local = question_conditioned_selection(attn_weights, image_pos, qp, qtop_frac)
        indices_buffer.update_indices1(
            local.unsqueeze(-1), image_token_index=image_token_index
        )
        indices_buffer.prefill_selected_local = local.detach().cpu()
        indices_buffer.update_target1(prefill_target(lam, ceiling))
        indices_buffer.commit_prefill_factor()
        if conserve:
            sink_local = question_conditioned_sinks(
                attn_weights, image_pos, qp, sink_frac, indices_buffer.indices1_local
            )
            indices_buffer.update_sinks(sink_local)
            if sink_local.numel() > 0:
                q_rel = _rows_to_image_attention(attn_weights, image_pos, qp)
                indices_buffer.prefill_sink_mass = float(q_rel[sink_local].mean())
            assert not (
                set(indices_buffer.indices1_local.tolist()) & set(sink_local.tolist())
            ), "Ponto 3: sinks and the qcond selection must be disjoint."

    if not gen_new_token:
        self.image_attention = image_attention
    else:
        self.image_attention = (
            1 - beta
        ) * image_attention + beta * self.image_attention

    # Ponto 3: conserved reinforcement. Runs only at decode, on the layers above
    # the reference (same window as the qcond boost), AFTER image_attention and
    # its EMA are read so the Point-1 deficit still measures natural behaviour,
    # and BEFORE the value matmul so the reallocation reaches the output.
    if (
        conserve
        and gen_new_token
        and selected_layer is not None
        and self.layer_idx > selected_layer
        and self.layer_idx <= se_layers[1]
    ):
        attn_row = attn_weights[:, :, -1, :]
        new_row = conserve_reallocation(
            attn_row, indices_buffer.sink_positions, indices_buffer.indices1, rho
        )
        if new_row is not attn_row:
            indices_buffer.reallocated_mass = float(
                (new_row.float() - attn_row.float()).clamp(min=0).sum()
            )
            attn_weights[:, :, -1, :] = new_row

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
    adaptive: Optional[bool] = False,
    lam: Optional[float] = 0.0,
    ceiling: Optional[float] = 2.0,
    qcond: Optional[bool] = False,
    qtop_frac: Optional[float] = 0.05,
    selected_layer: Optional[int] = None,
    conserve: Optional[bool] = False,
    rho: Optional[float] = 0.5,
    sink_frac: Optional[float] = 0.05,
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
        if adaptive:
            indices_buffer.prepare_adaptive_correction()

    gen_new_token = (
        past_key_values is not None and past_key_values.get_seq_length(self.layer_idx) > 0
    )

    if self.layer_idx >= se_layers[0] and self.layer_idx <= se_layers[1]:
        if adaptive:
            # Relaxation has to run on steps where nothing is selected, so the
            # guard is the pending correction, not the emptiness of indices2.
            if indices_buffer.correction is not None:
                indices_buffer.calibrate_adaptive(
                    past_key_values.layers[self.layer_idx].values
                )
        elif len(indices_buffer.indices2) > 0:
            indices_buffer.calibrate(past_key_values.layers[self.layer_idx].values, alpha)

    if _applies_prefill_boost(
        self.layer_idx, gen_new_token, qcond, adaptive, selected_layer,
        se_layers, indices_buffer,
    ):
        _boost_prefill_values(value_states, indices_buffer)

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
            # With qcond the prefill top-k is frozen for the whole generation:
            # the question does not change, so neither does what is relevant.
            if not qcond:
                ratio = (image_attention - self.image_attention) / self.image_attention
                ratio = ratio.squeeze(dim=0)
                indices = (ratio >= tau).nonzero()
                indices_buffer.update_indices1(indices, image_token_index=image_token_index)
            if adaptive:
                indices_buffer.update_target1(
                    adaptive_target_factor(
                        image_attention, self.image_attention, lam, ceiling
                    )
                )
    elif selected and qcond and adaptive:
        # Prefill, reference layer. The selection and the target decided here are
        # consumed by the layers ABOVE, still inside this same prefill forward:
        # the answer token of a VQA prompt comes out of the prefill logits, a
        # forward the decode-step calibration never touches.
        if indices_buffer.question_positions is None:
            raise RuntimeError(
                "Question-conditioned SPARC needs update_question_positions() "
                "before the prefill: the relevance signal is read from the "
                "attention rows of the question and cannot be located without "
                "them."
            )
        # image_positions is guaranteed here: adaptive is on, so layer 0 already
        # ran prepare_adaptive_correction, which raises without it.
        image_pos = indices_buffer.image_positions
        qp = indices_buffer.question_positions
        local = question_conditioned_selection(attn_weights, image_pos, qp, qtop_frac)
        indices_buffer.update_indices1(
            local.unsqueeze(-1), image_token_index=image_token_index
        )
        indices_buffer.prefill_selected_local = local.detach().cpu()
        indices_buffer.update_target1(prefill_target(lam, ceiling))
        indices_buffer.commit_prefill_factor()
        if conserve:
            sink_local = question_conditioned_sinks(
                attn_weights, image_pos, qp, sink_frac, indices_buffer.indices1_local
            )
            indices_buffer.update_sinks(sink_local)
            if sink_local.numel() > 0:
                q_rel = _rows_to_image_attention(attn_weights, image_pos, qp)
                indices_buffer.prefill_sink_mass = float(q_rel[sink_local].mean())
            assert not (
                set(indices_buffer.indices1_local.tolist()) & set(sink_local.tolist())
            ), "Ponto 3: sinks and the qcond selection must be disjoint."

    if not gen_new_token:
        self.image_attention = image_attention
    else:
        self.image_attention = (
            1 - beta
        ) * image_attention + beta * self.image_attention

    # Ponto 3: conserved reinforcement. Runs only at decode, on the layers above
    # the reference (same window as the qcond boost), AFTER image_attention and
    # its EMA are read so the Point-1 deficit still measures natural behaviour,
    # and BEFORE the value matmul so the reallocation reaches the output.
    if (
        conserve
        and gen_new_token
        and selected_layer is not None
        and self.layer_idx > selected_layer
        and self.layer_idx <= se_layers[1]
    ):
        attn_row = attn_weights[:, :, -1, :]
        new_row = conserve_reallocation(
            attn_row, indices_buffer.sink_positions, indices_buffer.indices1, rho
        )
        if new_row is not attn_row:
            indices_buffer.reallocated_mass = float(
                (new_row.float() - attn_row.float()).clamp(min=0).sum()
            )
            attn_weights[:, :, -1, :] = new_row

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
    adaptive=False,
    lam=0.0,
    ceiling=2.0,
    qcond=False,
    qtop_frac=0.05,
    conserve=False,
    rho=0.5,
    sink_frac=0.05,
):
    """Monkey-patch every decoder layer's ``self_attn.forward`` with SPARC.

    Args:
        family: ``"qwen"``, ``"llama"``, or ``None`` to auto-detect from the
            model class. The Qwen variant rotates Q/K via mRoPE; the Llama
            variant uses standard 1D RoPE.
        adaptive: Switch the value-cache reinforcement from the original
            accumulating ``alpha^c`` to the deficit-driven target factor with a
            ceiling. ``alpha`` is unused when this is on. Only ``forward_llama``
            and ``forward_qwen25vl`` honour it; the legacy ``forward_internlm2``
            swallows it via ``**kwargs`` and stays on ``alpha^c``.
        lam: Deficit sensitivity. ``lam=0`` makes the adaptive path neutral.
        ceiling: Saturation cap on the target factor.
        qcond: Select the visual tokens at the prefill by the contrastive
            attention the question pays them, freeze that selection, and apply
            the reinforcement inside the prefill itself. Requires ``adaptive``;
            only ``forward_llama`` and ``forward_qwen25vl`` honour it. The caller
            must have set ``question_positions`` on the buffer.
        qtop_frac: Fraction of the visual tokens the prefill selection keeps.
        conserve: Reallocate attention mass from the visual sinks to the qcond
            selection at each decode step, within the same layer window as the
            qcond boost. Requires ``qcond``; only ``forward_llama`` and
            ``forward_qwen25vl`` honour it. ``rho=0`` or an empty sink set makes
            it an exact no-op.
        rho: Fraction of each sink's attention mass reallocated per step.
        sink_frac: Top fraction by raw question attention that is a sink candidate.
    """
    if family is None:
        family = detect_model_family(model)
    if family not in _FORWARD_BY_FAMILY:
        raise ValueError(f"Unknown SPARC family {family!r}. "
                         f"Known: {sorted(_FORWARD_BY_FAMILY)}.")

    forward_fn = _FORWARD_BY_FAMILY[family]

    if qcond:
        # Only the layers above the reference get the prefill factor written into
        # their cache. ``accum_factors`` is one number per position, shared by
        # every layer, so letting the decode calibration touch a layer that never
        # received the prefill factor would apply target/prefill_target to a raw
        # value. Narrow the window instead.
        if selected_layer + 1 > se_layers[1]:
            raise ValueError(
                f"qcond=True with selected_layer={selected_layer} and "
                f"se_layers={tuple(se_layers)} leaves no layer above the "
                "reference to apply the reinforcement."
            )
        se_layers = (max(se_layers[0], selected_layer + 1), se_layers[1])

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
            adaptive=adaptive,
            lam=lam,
            ceiling=ceiling,
            qcond=qcond,
            qtop_frac=qtop_frac,
            selected_layer=selected_layer,
            conserve=conserve,
            rho=rho,
            sink_frac=sink_frac,
        )
        # Family-agnostic attribute lookup: `.self_attn` for qwen / llava /
        # smolvlm / idefics3; `.attention` for InternLM2 (InternVL). See
        # `_attention_module_of` above.
        attn_module = _attention_module_of(layer)
        attn_module.forward = MethodType(forward_, attn_module)
