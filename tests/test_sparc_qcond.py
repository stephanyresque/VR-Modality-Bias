"""Tests for the question-conditioned selection of SPARC (``qcond=True``).

Seven invariants:

1. ``question_conditioned_selection`` returns the right local top-k, and the
   buffer translates it through non-contiguous ``image_positions``.
2. ``prefill_target_factor`` reproduces the Point-1 formula, at the prefill.
3. The value cache changes at STEP 1, and only at the selected positions. This
   is the test Point 1 alone could not pass: its selection is empty at step 1.
4. The prefill selection stays frozen while generating.
5. Neutrality gate: ``lam=0`` with ``qcond=True`` leaves the cache untouched.
6. Non-regression: ``qcond=False`` keeps the Point-1 behaviour (blind at step 1)
   and the ``alpha^c`` path.
7. ``qcond=True`` without ``update_question_positions`` raises at the prefill.
"""

from __future__ import annotations

import importlib.util
import math
from functools import partial
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache

from vr_modality_bias.experiment.sparc import SparcHyperparams
from vr_modality_bias.utils.attn import (
    SelectedIndexBuffer,
    forward_llama,
    prefill_target_factor,
    question_conditioned_selection,
)

# ---------------------------------------------------------------- fixtures

HIDDEN = 8
N_HEADS = 2
HEAD_DIM = 4
PREFILL_LEN = 10
# Non-contiguous on purpose (the SmolVLM layout): a local/global mix-up would
# survive a contiguous block.
IMAGE_POSITIONS = (1, 2, 4, 5, 6)
QUESTION_POSITIONS = (7, 8, 9)
# floor(0.4 * 5) = 2, so two of the five visual tokens get selected.
QTOP_FRAC = 0.4
EXPECTED_K = 2
CEILING = 2.0

ALWAYS_SELECT = -1e9

# A (weight seed, input seed) pair whose prefill actually shows a visual-attention
# deficit. Without one, test 3 would be vacuous: the target would resolve to 1.0
# and nothing would be written to the cache.
DEFICIT_WEIGHT_SEED = 1
DEFICIT_INPUT_SEED = 3


class _TinyAttn(nn.Module):
    """Smallest module satisfying what ``forward_llama`` reads off ``self``."""

    def __init__(self, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = HEAD_DIM
        self.num_key_value_groups = 1
        self.scaling = HEAD_DIM**-0.5
        self.attention_dropout = 0.0
        self.q_proj = nn.Linear(HIDDEN, N_HEADS * HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(HIDDEN, N_HEADS * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(HIDDEN, N_HEADS * HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(N_HEADS * HEAD_DIM, HIDDEN, bias=False)


def _pos(q_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """cos=1, sin=0 keeps RoPE inert; it is not what is under test."""
    return torch.ones(1, q_len, HEAD_DIM), torch.zeros(1, q_len, HEAD_DIM)


def _fresh_layer(seed: int = DEFICIT_WEIGHT_SEED) -> _TinyAttn:
    torch.manual_seed(seed)
    layer = _TinyAttn()
    layer.eval()
    return layer


def _fresh_buffer(*, with_question_positions: bool = True) -> SelectedIndexBuffer:
    buffer = SelectedIndexBuffer()
    buffer.reset()
    buffer.update_input_len(PREFILL_LEN - len(IMAGE_POSITIONS))
    buffer.update_image_positions(torch.tensor(IMAGE_POSITIONS, dtype=torch.long))
    if with_question_positions:
        buffer.update_question_positions(
            torch.tensor(QUESTION_POSITIONS, dtype=torch.long)
        )
    return buffer


def _patch(
    layer: _TinyAttn,
    buffer: SelectedIndexBuffer,
    *,
    alpha: float = 1.0,
    beta: float = 0.1,
    tau: float = 2.0,
    selected: bool = True,
    adaptive: bool = True,
    lam: float = 0.0,
    ceiling: float = CEILING,
    qcond: bool = True,
    qtop_frac: float = QTOP_FRAC,
) -> None:
    forward_ = partial(
        forward_llama,
        alpha=alpha,
        beta=beta,
        tau=tau,
        selected=selected,
        se_layers=(0, 31),
        image_token_index=IMAGE_POSITIONS[0],
        indices_buffer=buffer,
        adaptive=adaptive,
        lam=lam,
        ceiling=ceiling,
        qcond=qcond,
        qtop_frac=qtop_frac,
    )
    layer.forward = MethodType(forward_, layer)


def _inputs(n_steps: int, seed: int = DEFICIT_INPUT_SEED):
    gen = torch.Generator().manual_seed(seed)
    prefill = torch.randn(1, PREFILL_LEN, HIDDEN, generator=gen)
    steps = [torch.randn(1, 1, HIDDEN, generator=gen) for _ in range(n_steps)]
    return prefill, steps


@torch.no_grad()
def _drive(layer, prefill, steps):
    """Prefill, then one forward per step. Snapshots the cache after each."""
    cache = DynamicCache()
    outputs = [
        layer(
            hidden_states=prefill,
            past_key_values=cache,
            position_embeddings=_pos(PREFILL_LEN),
        )[0]
    ]
    after_prefill = cache.layers[0].values.clone()
    per_step = []
    for hidden in steps:
        outputs.append(
            layer(hidden_states=hidden, past_key_values=cache, position_embeddings=_pos(1))[0]
        )
        per_step.append(cache.layers[0].values.clone())
    return cache, outputs, after_prefill, per_step


def _image_rows(values: torch.Tensor) -> torch.Tensor:
    return values[:, :, torch.tensor(IMAGE_POSITIONS)]


def _synthetic_attention(image_row_values, question_row_values=None) -> torch.Tensor:
    """Prefill attention whose question-to-image submatrix is exactly specified.

    ``image_row_values`` maps a question row index to the attention it gives to
    each of the five image columns.
    """
    attn = torch.zeros(1, N_HEADS, PREFILL_LEN, PREFILL_LEN)
    for head in range(N_HEADS):
        for row, values in image_row_values.items():
            for local, column in enumerate(IMAGE_POSITIONS):
                attn[0, head, row, column] = values[local]
    if question_row_values is not None:
        for head, rows in question_row_values.items():
            for row, values in rows.items():
                for local, column in enumerate(IMAGE_POSITIONS):
                    attn[0, head, row, column] = values[local]
    return attn


IMAGE_POS_T = torch.tensor(IMAGE_POSITIONS, dtype=torch.long)
QUESTION_POS_T = torch.tensor(QUESTION_POSITIONS, dtype=torch.long)


# ---------------------------------------------------------------- 1. selection


def test_selection_returns_the_top_k_by_question_relevance():
    relevance = [0.1, 0.5, 0.2, 0.9, 0.3]
    attn = _synthetic_attention({row: relevance for row in QUESTION_POSITIONS})
    local = question_conditioned_selection(attn, IMAGE_POS_T, QUESTION_POS_T, QTOP_FRAC)
    # Descending relevance: 0.9 at local 3, then 0.5 at local 1.
    assert local.tolist() == [3, 1]
    assert local.dtype is torch.int64


def test_selection_k_is_floor_of_the_fraction_with_a_floor_of_one():
    attn = _synthetic_attention({row: [0.1, 0.5, 0.2, 0.9, 0.3] for row in QUESTION_POSITIONS})
    # floor(0.01 * 5) == 0, but the selection must never come back empty.
    assert question_conditioned_selection(attn, IMAGE_POS_T, QUESTION_POS_T, 0.01).tolist() == [3]
    assert len(question_conditioned_selection(attn, IMAGE_POS_T, QUESTION_POS_T, 0.4)) == 2
    assert question_conditioned_selection(attn, IMAGE_POS_T, QUESTION_POS_T, 1.0).tolist() == [
        3, 1, 4, 2, 0
    ]


def test_selection_averages_over_heads():
    """Head 0 loves local 0, head 1 ignores it; the mean must decide."""
    attn = _synthetic_attention({row: [0.0, 0.4, 0.0, 0.0, 0.0] for row in QUESTION_POSITIONS})
    for row in QUESTION_POSITIONS:
        attn[0, 0, row, IMAGE_POSITIONS[0]] = 1.0  # head 0 only -> mean 0.5
    local = question_conditioned_selection(attn, IMAGE_POS_T, QUESTION_POS_T, 0.2)
    assert local.tolist() == [0]


def test_selection_ignores_rows_outside_the_question():
    """A non-question row screaming at local 4 must not move the ranking."""
    attn = _synthetic_attention({row: [0.1, 0.5, 0.2, 0.9, 0.3] for row in QUESTION_POSITIONS})
    for head in range(N_HEADS):
        attn[0, head, 0, IMAGE_POSITIONS[4]] = 100.0  # row 0 is not a question row
    assert question_conditioned_selection(attn, IMAGE_POS_T, QUESTION_POS_T, 0.4).tolist() == [3, 1]


def test_selection_zeroes_non_finite_relevance_before_the_top_k():
    attn = _synthetic_attention({row: [0.1, 0.5, 0.2, 0.9, 0.3] for row in QUESTION_POSITIONS})
    for head in range(N_HEADS):
        attn[0, head, QUESTION_POSITIONS[0], IMAGE_POSITIONS[3]] = float("nan")
    local = question_conditioned_selection(attn, IMAGE_POS_T, QUESTION_POS_T, 0.4)
    assert 3 not in local.tolist(), "a NaN relevance must never win the ranking"
    assert local.tolist() == [1, 4]


def test_buffer_translates_the_local_top_k_through_noncontiguous_positions():
    attn = _synthetic_attention({row: [0.1, 0.5, 0.2, 0.9, 0.3] for row in QUESTION_POSITIONS})
    local = question_conditioned_selection(attn, IMAGE_POS_T, QUESTION_POS_T, QTOP_FRAC)

    buffer = _fresh_buffer()
    buffer.update_indices1(local.unsqueeze(-1), image_token_index=IMAGE_POSITIONS[0])
    # local 3 -> global 5, local 1 -> global 2.
    assert buffer.indices1.tolist() == [5, 2]
    assert buffer.indices1_local.tolist() == [3, 1]


def test_single_selected_token_stays_one_dimensional():
    """k=1 through unsqueeze(-1): the squeeze in update_indices1 must not make
    it a 0-d scalar, which would silently break the registry indexing."""
    attn = _synthetic_attention({row: [0.1, 0.5, 0.2, 0.9, 0.3] for row in QUESTION_POSITIONS})
    local = question_conditioned_selection(attn, IMAGE_POS_T, QUESTION_POS_T, 0.01)
    buffer = _fresh_buffer()
    buffer.update_indices1(local.unsqueeze(-1), image_token_index=IMAGE_POSITIONS[0])
    assert buffer.indices1_local.dim() == 1
    assert buffer.indices1.tolist() == [5]


# ---------------------------------------------------------------- 2. prefill target


def test_prefill_target_reproduces_the_deficit_formula():
    # Question rows 7 and 8 give 0.4 to every image column, row 9 (the last
    # prompt position) gives 0.1. ref = mean(0.4, 0.4, 0.1) = 0.3, current = 0.1.
    # deficit = (0.3 - 0.1) / 0.3 = 2/3, so lam=0.75 -> target 1.5.
    attn = _synthetic_attention({7: [0.4] * 5, 8: [0.4] * 5, 9: [0.1] * 5})
    target = prefill_target_factor(attn, IMAGE_POS_T, QUESTION_POS_T, 0.75, 9.0)
    assert target == pytest.approx(1.5)


def test_prefill_target_saturates_at_the_ceiling():
    attn = _synthetic_attention({7: [0.4] * 5, 8: [0.4] * 5, 9: [0.1] * 5})
    assert prefill_target_factor(attn, IMAGE_POS_T, QUESTION_POS_T, 1e6, CEILING) == CEILING


def test_prefill_target_is_one_without_a_deficit():
    """The last row attends the image MORE than the question rows did."""
    attn = _synthetic_attention({7: [0.1] * 5, 8: [0.1] * 5, 9: [0.9] * 5})
    assert prefill_target_factor(attn, IMAGE_POS_T, QUESTION_POS_T, 5.0, CEILING) == 1.0


def test_prefill_target_is_one_when_the_reference_is_degenerate():
    attn = _synthetic_attention({row: [0.0] * 5 for row in QUESTION_POSITIONS})
    target = prefill_target_factor(attn, IMAGE_POS_T, QUESTION_POS_T, 5.0, CEILING)
    assert math.isfinite(target)
    assert target == 1.0


def test_prefill_target_is_never_non_finite():
    attn = _synthetic_attention({7: [float("nan")] * 5, 8: [0.0] * 5, 9: [0.0] * 5})
    target = prefill_target_factor(attn, IMAGE_POS_T, QUESTION_POS_T, 5.0, CEILING)
    assert math.isfinite(target)
    assert target == 1.0


# ---------------------------------------------------------------- 3. step-1 effect


def test_cache_is_amplified_at_step_one_and_only_where_selected():
    """The whole point of Point 2. Point 1 alone leaves step 1 untouched."""
    prefill, steps = _inputs(1)
    layer = _fresh_layer()
    buffer = _fresh_buffer()
    _patch(layer, buffer, lam=1e6, tau=ALWAYS_SELECT)
    _, _, after_prefill, per_step = _drive(layer, prefill, steps)

    selected_local = buffer.prefill_selected_local.tolist()
    assert len(selected_local) == EXPECTED_K
    assert buffer.target2 == CEILING, "the prefill must have found a real deficit"

    ratios = _image_rows(per_step[0]) / _image_rows(after_prefill)
    per_position = ratios[0, 0, :, 0]
    for local in range(len(IMAGE_POSITIONS)):
        want = CEILING if local in selected_local else 1.0
        assert per_position[local] == pytest.approx(want), f"local {local}"


def test_step_one_target_comes_from_the_prefill_not_from_the_moving_average():
    prefill, steps = _inputs(1)
    layer = _fresh_layer()
    buffer = _fresh_buffer()
    _patch(layer, buffer, lam=0.5, tau=ALWAYS_SELECT)
    _drive(layer, prefill, steps)
    # target2 is what step 1 applied; it was decided at the prefill.
    assert buffer.target2 > 1.0
    assert buffer.accum_factors.max() == pytest.approx(buffer.target2)


# ---------------------------------------------------------------- 4. frozen selection


def test_prefill_selection_is_frozen_during_generation():
    prefill, steps = _inputs(4)
    layer = _fresh_layer()
    buffer = _fresh_buffer()
    # tau=ALWAYS_SELECT: the relative-rise criterion would take every token.
    _patch(layer, buffer, lam=0.5, tau=ALWAYS_SELECT)
    _drive(layer, prefill, steps)

    frozen = buffer.prefill_selected_local.tolist()
    assert len(frozen) == EXPECTED_K
    assert buffer.indices1_local.tolist() == frozen
    assert buffer.indices2_local.tolist() == frozen
    assert sorted(buffer.indices1.tolist()) == sorted(IMAGE_POSITIONS[i] for i in frozen)


def test_intensity_still_adapts_per_step_while_the_selection_is_frozen():
    prefill, steps = _inputs(4)
    layer = _fresh_layer()
    buffer = _fresh_buffer()
    _patch(layer, buffer, lam=0.5, tau=ALWAYS_SELECT)
    _drive(layer, prefill, steps)
    # Point 1 untouched: the factor a selected position carries is the current
    # step's target, not a product over steps.
    assert float(buffer.accum_factors.max()) <= CEILING
    assert float(buffer.accum_factors.max()) == pytest.approx(buffer.target2)


# ---------------------------------------------------------------- 5. neutrality gate


def test_neutrality_gate_lambda_zero_with_qcond_matches_a_run_without_sparc():
    prefill, steps = _inputs(4)

    reference_layer = _fresh_layer()
    reference_buffer = _fresh_buffer()
    _patch(reference_layer, reference_buffer, selected=False, adaptive=False, qcond=False)
    reference_cache, reference_outputs, _, _ = _drive(reference_layer, prefill, steps)

    qcond_layer = _fresh_layer()
    qcond_buffer = _fresh_buffer()
    _patch(qcond_layer, qcond_buffer, lam=0.0, tau=ALWAYS_SELECT)
    qcond_cache, qcond_outputs, _, _ = _drive(qcond_layer, prefill, steps)

    # Non-vacuous: the prefill DID select, it just resolved a target of 1.0.
    assert len(qcond_buffer.prefill_selected_local) == EXPECTED_K
    assert qcond_buffer.target2 == 1.0
    assert torch.equal(qcond_cache.layers[0].values, reference_cache.layers[0].values)
    for got, want in zip(qcond_outputs, reference_outputs, strict=True):
        assert torch.equal(got, want)
    assert torch.equal(qcond_buffer.accum_factors, torch.ones(len(IMAGE_POSITIONS)))


# ---------------------------------------------------------------- 6. non-regression


def test_point_one_alone_is_still_blind_at_step_one():
    """qcond=False: the inherited selection is empty at step 1, so the cache is
    untouched. This is the defect Point 2 exists to fix; pinning it here means
    test 3 above is measuring a real change."""
    prefill, steps = _inputs(1)
    layer = _fresh_layer()
    buffer = _fresh_buffer()
    _patch(layer, buffer, lam=1e6, tau=ALWAYS_SELECT, qcond=False)
    _, _, after_prefill, per_step = _drive(layer, prefill, steps)

    assert buffer.prefill_selected_local is None
    assert torch.equal(_image_rows(per_step[0]), _image_rows(after_prefill))


def test_qcond_false_leaves_the_prefill_state_alone():
    prefill, steps = _inputs(3)
    layer = _fresh_layer()
    buffer = _fresh_buffer()
    _patch(layer, buffer, lam=0.5, tau=ALWAYS_SELECT, qcond=False)
    _drive(layer, prefill, steps)
    # The relative-rise criterion is back in charge, so everything is selected.
    assert buffer.prefill_selected_local is None
    assert len(buffer.indices1_local) == len(IMAGE_POSITIONS)


def test_alpha_c_path_is_untouched_by_the_new_kwargs():
    prefill, steps = _inputs(4)
    layer = _fresh_layer()
    buffer = _fresh_buffer()
    _patch(layer, buffer, alpha=1.5, tau=ALWAYS_SELECT, adaptive=False, qcond=False)
    cache, _, after_prefill, _ = _drive(layer, prefill, steps)

    ratios = _image_rows(cache.layers[0].values) / _image_rows(after_prefill)
    # calibrate fires on steps 2, 3 and 4.
    assert torch.allclose(ratios, torch.full_like(ratios, 1.5**3), rtol=1e-5)
    assert torch.equal(buffer.accum_factors, torch.ones(len(IMAGE_POSITIONS)))
    assert buffer.prefill_selected_local is None


def test_hparams_reject_qcond_without_adaptive():
    with pytest.raises(ValueError, match="qcond"):
        SparcHyperparams(alpha=1.3, adaptive=False, qcond=True)


@pytest.mark.parametrize("qtop_frac", [0.0, -0.1, 1.01, 2.0])
def test_hparams_reject_a_qtop_frac_outside_the_unit_interval(qtop_frac):
    with pytest.raises(ValueError, match="qtop_frac"):
        SparcHyperparams(alpha=1.0, adaptive=True, qcond=True, qtop_frac=qtop_frac)


def test_hparams_accept_qtop_frac_at_one():
    assert SparcHyperparams(alpha=1.0, adaptive=True, qcond=True, qtop_frac=1.0).qtop_frac == 1.0


def test_hparams_ignore_qtop_frac_when_qcond_is_off():
    assert SparcHyperparams(alpha=1.3, qtop_frac=5.0).qtop_frac == 5.0


def test_hparams_expose_qcond_in_as_dict():
    hparams = SparcHyperparams(alpha=1.0, adaptive=True, qcond=True, qtop_frac=0.25)
    assert hparams.as_dict()["qcond"] is True
    assert hparams.as_dict()["qtop_frac"] == 0.25


# ---------------------------------------------------------------- 7. guard


def test_qcond_without_question_positions_raises_at_the_prefill():
    prefill, _ = _inputs(0)
    layer = _fresh_layer()
    buffer = _fresh_buffer(with_question_positions=False)
    _patch(layer, buffer, lam=0.5)
    with pytest.raises(RuntimeError, match="update_question_positions"):
        _drive(layer, prefill, [])


# ---------------------------------------------------------------- question positions probe

_SCRIPTS = Path(__file__).parent.parent / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"_qcond_script_{name}", _SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _MockProcessor:
    def __init__(self, ids):
        self._ids = ids

    def apply_chat_template(self, *_args, **_kwargs):
        return "<prefix>"

    def __call__(self, text, images=None, return_tensors="pt"):
        return {"input_ids": torch.tensor([self._ids])}


class _MockWrapper:
    def __init__(self, ids, image_token_id=42):
        self._processor = _MockProcessor(ids)
        self._model = SimpleNamespace(config=SimpleNamespace(image_token_id=image_token_id))

    @staticmethod
    def _build_messages(prompt, image):
        return [{"role": "user", "content": prompt}]


@pytest.mark.parametrize("script_name", ["phase3_generate", "pope_generate"])
def test_question_positions_follow_the_last_image_token_contiguous(script_name):
    """Qwen layout: image tokens form a block."""
    script = _load_script(script_name)
    ids = [1, 42, 42, 42, 7, 8, 9]  # images at 1..3, question at 4..6
    input_len, image_positions, question_positions = script._probe_sparc_layout(
        _MockWrapper(ids), None, "q"
    )
    assert image_positions.tolist() == [1, 2, 3]
    assert question_positions.tolist() == [4, 5, 6]
    assert input_len == len(ids) - 3


@pytest.mark.parametrize("script_name", ["phase3_generate", "pope_generate"])
def test_question_positions_follow_the_last_image_token_interleaved(script_name):
    """Idefics3 / SmolVLM layout: separators sit between the image tokens.

    The question set starts after the LAST image token, so the separators that
    live inside the image block are excluded, and only the trailing text remains.
    """
    script = _load_script(script_name)
    ids = [1, 42, 99, 42, 99, 42, 7, 8]  # images at 1, 3, 5; question at 6..7
    input_len, image_positions, question_positions = script._probe_sparc_layout(
        _MockWrapper(ids), None, "q"
    )
    assert image_positions.tolist() == [1, 3, 5]
    assert question_positions.tolist() == [6, 7]
    assert input_len == len(ids) - 3
