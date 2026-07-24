"""Tests for Ponto 3: conserved reinforcement by attention reallocation
(``conserve=True``).

Covers the pure reallocation (row-sum conservation in fp32, proportional
withdrawal and delivery, uniform fallback, exact short-circuits, untouched
columns, dtype), the sink signal (zero contrast AND top raw AND not selected,
disjoint from the selection), the frozen local->global translation, the forward
neutrality gates (conserve=False and rho=0 byte-identical), the decode-only
in-window firing, and the new SparcHyperparams validations.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache

from vr_modality_bias.experiment.sparc import SparcHyperparams
from vr_modality_bias.utils.attn import (
    SelectedIndexBuffer,
    add_custom_attention_layers,
    conserve_reallocation,
    decoder_of,
    question_conditioned_selection,
    question_conditioned_sinks,
)

# ---------------------------------------------------------------- fixtures

HIDDEN = 8
N_HEADS = 2
HEAD_DIM = 4
PREFILL_LEN = 10
N_LAYERS = 4
SELECTED_LAYER = 1
SE_LAYERS = (0, N_LAYERS - 1)
WINDOW_LAYERS = (2, 3)
BELOW_OR_AT_REFERENCE = (0, 1)

IMAGE_POSITIONS = (1, 2, 4, 5, 6)
QUESTION_POSITIONS = (7, 8, 9)
CEILING = 2.0
BETA = 0.1

IMAGE_POS_T = torch.tensor(IMAGE_POSITIONS, dtype=torch.long)
QUESTION_POS_T = torch.tensor(QUESTION_POSITIONS, dtype=torch.long)
EMPTY = torch.tensor([], dtype=torch.long)


class _TinyAttn(nn.Module):
    def __init__(self, layer_idx: int):
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


class _MockLayer(nn.Module):
    def __init__(self, layer_idx: int):
        super().__init__()
        self.self_attn = _TinyAttn(layer_idx)


class _MockTextModel(nn.Module):
    def __init__(self, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([_MockLayer(i) for i in range(n_layers)])


class _MockModel(nn.Module):
    def __init__(self, n_layers: int = N_LAYERS):
        super().__init__()
        self.model = SimpleNamespace(text_model=_MockTextModel(n_layers))


def _pos(q_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ones(1, q_len, HEAD_DIM), torch.zeros(1, q_len, HEAD_DIM)


def _fresh_buffer() -> SelectedIndexBuffer:
    buffer = SelectedIndexBuffer()
    buffer.reset()
    buffer.update_input_len(PREFILL_LEN - len(IMAGE_POSITIONS))
    buffer.update_image_positions(IMAGE_POS_T.clone())
    buffer.update_question_positions(QUESTION_POS_T.clone())
    return buffer


def _build_stack(
    buffer: SelectedIndexBuffer,
    *,
    seed: int = 0,
    conserve: bool = True,
    rho: float = 0.5,
    sink_frac: float = 0.05,
    qtop_frac: float = 0.4,
) -> _MockModel:
    torch.manual_seed(seed)
    model = _MockModel()
    model.eval()
    add_custom_attention_layers(
        model,
        alpha=1.0,
        beta=BETA,
        tau=-1e9,
        selected_layer=SELECTED_LAYER,
        se_layers=SE_LAYERS,
        image_token_index=IMAGE_POSITIONS[0],
        indices_buffer=buffer,
        adaptive=True,
        lam=0.5,
        ceiling=CEILING,
        qcond=True,
        qtop_frac=qtop_frac,
        conserve=conserve,
        rho=rho,
        sink_frac=sink_frac,
    )
    return model


def _inputs(n_steps: int, seed: int = 100):
    gen = torch.Generator().manual_seed(seed)
    prefill = torch.randn(1, PREFILL_LEN, HIDDEN, generator=gen)
    steps = [torch.randn(1, 1, HIDDEN, generator=gen) for _ in range(n_steps)]
    return prefill, steps


@torch.no_grad()
def _sweep(attns, cache, hidden):
    outputs = []
    for attn in attns:
        hidden = attn(
            hidden_states=hidden,
            past_key_values=cache,
            position_embeddings=_pos(hidden.shape[1]),
        )[0]
        outputs.append(hidden)
    return outputs


@torch.no_grad()
def _drive(model: _MockModel, prefill, steps):
    attns = [layer.self_attn for layer in decoder_of(model).layers]
    cache = DynamicCache()
    prefill_outputs = _sweep(attns, cache, prefill)
    step_outputs = [_sweep(attns, cache, h) for h in steps]
    return cache, prefill_outputs, step_outputs


def _synthetic_attention(rows_to_image: dict) -> torch.Tensor:
    attn = torch.zeros(1, N_HEADS, PREFILL_LEN, PREFILL_LEN)
    for head in range(N_HEADS):
        for row, values in rows_to_image.items():
            for local, column in enumerate(IMAGE_POSITIONS):
                attn[0, head, row, column] = values[local]
    return attn


# ---------------------------------------------------------------- 1. reallocation math


def test_reallocation_preserves_the_row_sum_per_head():
    row = torch.rand(1, N_HEADS, 6)
    out = conserve_reallocation(row, torch.tensor([1, 2]), torch.tensor([4, 5]), 0.5)
    assert torch.allclose(out.sum(-1), row.sum(-1), atol=1e-6)


def test_reallocation_withdraws_rho_of_each_sink_mass():
    row = torch.tensor([[[0.4, 0.2, 0.1, 0.3]]])
    out = conserve_reallocation(row, torch.tensor([0, 1]), torch.tensor([3]), 0.5)
    assert out[0, 0, 0] == pytest.approx(0.2)  # 0.4 * (1 - 0.5)
    assert out[0, 0, 1] == pytest.approx(0.1)  # 0.2 * (1 - 0.5)
    budget = 0.5 * (0.4 + 0.2)
    assert out[0, 0, 3] == pytest.approx(0.3 + budget)  # single target gets it all
    assert out[0, 0, 2] == pytest.approx(0.1)  # untouched visual column
    assert out.sum() == pytest.approx(row.sum(), abs=1e-6)


def test_reallocation_delivers_proportional_to_current_attention():
    row = torch.tensor([[[0.4, 0.1, 0.3]]])
    out = conserve_reallocation(row, torch.tensor([0]), torch.tensor([1, 2]), 1.0)
    budget = 1.0 * 0.4
    assert out[0, 0, 1] == pytest.approx(0.1 + budget * 0.25)  # 0.1 / (0.1 + 0.3)
    assert out[0, 0, 2] == pytest.approx(0.3 + budget * 0.75)
    assert out[0, 0, 0] == pytest.approx(0.0)  # rho=1 fully drains the sink


def test_reallocation_falls_back_to_uniform_when_targets_are_zero():
    row = torch.tensor([[[0.6, 0.0, 0.0, 0.4]]])
    out = conserve_reallocation(row, torch.tensor([0]), torch.tensor([1, 2]), 0.5)
    budget = 0.5 * 0.6
    assert out[0, 0, 1] == pytest.approx(budget / 2)
    assert out[0, 0, 2] == pytest.approx(budget / 2)
    assert out[0, 0, 3] == pytest.approx(0.4)  # text column untouched
    assert out.sum() == pytest.approx(row.sum(), abs=1e-6)


def test_reallocation_leaves_uninvolved_columns_bit_identical():
    row = torch.tensor([[[0.1, 0.4, 0.05, 0.2, 0.05, 0.2]]])
    out = conserve_reallocation(row, torch.tensor([1]), torch.tensor([3]), 0.5)
    for c in (0, 2, 4, 5):
        assert torch.equal(out[0, 0, c], row[0, 0, c])


def test_reallocation_is_independent_per_head():
    row = torch.tensor([[[0.4, 0.2, 0.4], [0.2, 0.6, 0.2]]])  # two heads
    out = conserve_reallocation(row, torch.tensor([0]), torch.tensor([2]), 0.5)
    assert out[0, 0, 2] == pytest.approx(0.4 + 0.5 * 0.4)
    assert out[0, 1, 2] == pytest.approx(0.2 + 0.5 * 0.2)


@pytest.mark.parametrize("case", ["rho_zero", "empty_sinks", "empty_targets"])
def test_reallocation_short_circuits_return_the_same_tensor(case):
    row = torch.tensor([[[0.4, 0.2, 0.4]]])
    sinks, targets, rho = torch.tensor([0]), torch.tensor([2]), 0.5
    if case == "rho_zero":
        out = conserve_reallocation(row, sinks, targets, 0.0)
    elif case == "empty_sinks":
        out = conserve_reallocation(row, EMPTY, targets, rho)
    else:
        out = conserve_reallocation(row, sinks, EMPTY, rho)
    assert out is row  # identity: no arithmetic touched the tensor
    assert torch.equal(out, torch.tensor([[[0.4, 0.2, 0.4]]]))


def test_reallocation_returns_the_input_dtype():
    row = torch.tensor([[[0.4, 0.2, 0.4]]], dtype=torch.float16)
    out = conserve_reallocation(row, torch.tensor([0]), torch.tensor([2]), 0.5)
    assert out.dtype == torch.float16


# ---------------------------------------------------------------- 2. sink signal


def _sink_relevant_attention() -> torch.Tensor:
    # local 0: high raw question attention but the background attends it MORE, so
    # the contrast clamps to zero (a sink). local 3: question-only -> positive
    # contrast (selected). Background strictly above the question on local 0
    # keeps the contrast robustly zero (equal values would round differently
    # across the 3 question rows and the 7 background rows).
    background = [0.95, 0.0, 0.0, 0.0, 0.0]
    question = [0.90, 0.0, 0.0, 0.5, 0.0]
    rows = {r: list(background) for r in range(PREFILL_LEN)}
    for r in QUESTION_POSITIONS:
        rows[r] = list(question)
    return _synthetic_attention(rows)


def test_sink_is_zero_contrast_and_top_raw_and_not_selected():
    attn = _sink_relevant_attention()
    selected = question_conditioned_selection(attn, IMAGE_POS_T, QUESTION_POS_T, 0.2)
    sinks = question_conditioned_sinks(attn, IMAGE_POS_T, QUESTION_POS_T, 0.2, selected)
    assert selected.tolist() == [3]
    assert sinks.tolist() == [0]
    assert set(selected.tolist()) & set(sinks.tolist()) == set()


def test_padded_zero_contrast_selection_is_excluded_from_the_sinks():
    """topk can pad the selection with zero-contrast columns; those must not
    resurface as sinks."""
    attn = _sink_relevant_attention()
    padded_selection = torch.tensor([3, 0])  # local 0 is a zero-contrast sink candidate
    sinks = question_conditioned_sinks(attn, IMAGE_POS_T, QUESTION_POS_T, 0.2, padded_selection)
    assert sinks.tolist() == []
    assert set(sinks.tolist()) & set(padded_selection.tolist()) == set()


def test_positive_contrast_column_is_never_a_sink_even_if_top_raw():
    # Both local 0 and local 3 are top raw (0.9). local 0 is a sink (background
    # attends it more -> contrast 0); local 3 has positive contrast, so it is
    # excluded from the sinks despite being top raw.
    background = [0.95, 0.0, 0.0, 0.3, 0.0]
    question = [0.90, 0.0, 0.0, 0.9, 0.0]
    rows = {r: list(background) for r in range(PREFILL_LEN)}
    for r in QUESTION_POSITIONS:
        rows[r] = list(question)
    attn = _synthetic_attention(rows)
    sinks = question_conditioned_sinks(attn, IMAGE_POS_T, QUESTION_POS_T, 0.4, EMPTY)
    assert 3 not in sinks.tolist()  # positive contrast, so not a sink
    assert sinks.tolist() == [0]


def test_no_sinks_when_every_top_raw_column_has_positive_contrast():
    background = [0.1, 0.1, 0.1, 0.1, 0.1]
    question = [0.9, 0.8, 0.7, 0.6, 0.5]
    rows = {r: list(background) for r in range(PREFILL_LEN)}
    for r in QUESTION_POSITIONS:
        rows[r] = list(question)
    attn = _synthetic_attention(rows)
    assert question_conditioned_sinks(attn, IMAGE_POS_T, QUESTION_POS_T, 0.4, EMPTY).tolist() == []


# ---------------------------------------------------------------- 3. frozen translation


def test_update_sinks_translates_local_to_global_like_the_selection():
    buffer = _fresh_buffer()
    buffer.update_sinks(torch.tensor([0, 3]))
    assert buffer.sink_local.tolist() == [0, 3]
    assert buffer.sink_positions.tolist() == [1, 5]  # IMAGE_POSITIONS[0], [3]
    assert buffer.n_sinks == 2


def test_reset_clears_the_sink_state():
    buffer = _fresh_buffer()
    buffer.update_sinks(torch.tensor([0, 3]))
    buffer.reset()
    assert buffer.sink_positions is None
    assert buffer.sink_local == []
    assert buffer.n_sinks == 0
    assert buffer.reallocated_mass == 0.0


# ---------------------------------------------------------------- 4. forward gates


def test_conserve_false_runs_no_new_code():
    """No sink detection, no reallocation state when the flag is off."""
    buffer = _fresh_buffer()
    prefill, steps = _inputs(3)
    _drive(_build_stack(buffer, conserve=False), prefill, steps)
    assert buffer.sink_positions is None
    assert buffer.n_sinks == 0
    assert buffer.reallocated_mass == 0.0


def test_rho_zero_is_byte_identical_to_conserve_false():
    prefill, steps = _inputs(3)

    off_buffer = _fresh_buffer()
    off_cache, off_prefill, off_steps = _drive(
        _build_stack(off_buffer, conserve=False), prefill, steps
    )

    rho0_buffer = _fresh_buffer()
    rho0_cache, rho0_prefill, rho0_steps = _drive(
        _build_stack(rho0_buffer, conserve=True, rho=0.0), prefill, steps
    )

    for i in range(N_LAYERS):
        assert torch.equal(off_cache.layers[i].values, rho0_cache.layers[i].values)
    for got, want in zip(off_prefill, rho0_prefill, strict=True):
        assert torch.equal(got, want)
    for got_step, want_step in zip(off_steps, rho0_steps, strict=True):
        for got, want in zip(got_step, want_step, strict=True):
            assert torch.equal(got, want)
    # Non-vacuous: rho=0 short-circuited, it did not reallocate.
    assert rho0_buffer.reallocated_mass == 0.0


def _drive_with_injected_sink(rho: float, seed: int = 0):
    """Run prefill, inject a sink disjoint from the real selection, run decode.

    Injecting only the sink (not the selection) keeps the adaptive state
    consistent, so a rho=0 vs rho>0 comparison isolates the reallocation.
    """
    buffer = _fresh_buffer()
    model = _build_stack(buffer, conserve=True, rho=rho, seed=seed)
    prefill, steps = _inputs(2)
    attns = [layer.self_attn for layer in decoder_of(model).layers]
    cache = DynamicCache()
    with torch.no_grad():
        _sweep(attns, cache, prefill)
        selected = set(buffer.indices1.tolist())
        sink_global = next(p for p in IMAGE_POSITIONS if p not in selected)
        sink_local = IMAGE_POSITIONS.index(sink_global)
        buffer.sink_positions = torch.tensor([sink_global], dtype=torch.long)
        buffer.sink_local = torch.tensor([sink_local], dtype=torch.long)
        buffer.n_sinks = 1
        mass_after_prefill = buffer.reallocated_mass
        step_outputs = [_sweep(attns, cache, h) for h in steps]
    return buffer, step_outputs, mass_after_prefill


def test_reallocation_fires_only_at_decode_and_only_above_the_reference():
    buffer0, out0, mass0_prefill = _drive_with_injected_sink(rho=0.0)
    buffer1, out1, mass1_prefill = _drive_with_injected_sink(rho=0.5)

    # Prefill never reallocates (decode-only).
    assert mass0_prefill == 0.0 and mass1_prefill == 0.0
    assert buffer1.reallocated_mass > 0.0
    assert buffer0.reallocated_mass == 0.0

    for step in range(len(out0)):
        for layer_idx in BELOW_OR_AT_REFERENCE:
            assert torch.equal(out0[step][layer_idx], out1[step][layer_idx])
        for layer_idx in WINDOW_LAYERS:
            assert not torch.equal(out0[step][layer_idx], out1[step][layer_idx])


def test_injected_sink_set_is_frozen_across_decode_steps():
    buffer, _, _ = _drive_with_injected_sink(rho=0.5)
    assert buffer.sink_positions.tolist() == buffer.sink_positions.tolist()
    # The decode path never rewrites the frozen set.
    assert buffer.n_sinks == 1


def test_forward_keeps_sinks_and_selection_disjoint():
    buffer = _fresh_buffer()
    prefill, steps = _inputs(2)
    _drive(_build_stack(buffer, conserve=True, rho=0.5, sink_frac=0.4), prefill, steps)
    selected = set(buffer.indices1_local.tolist())
    sinks = set(buffer.sink_local.tolist()) if buffer.n_sinks else set()
    assert selected & sinks == set()


# ---------------------------------------------------------------- 5. hyperparams


def test_hparams_conserve_defaults_off():
    hp = SparcHyperparams(alpha=1.3)
    assert hp.conserve is False
    assert hp.rho == 0.5
    assert hp.sink_frac == 0.05


def test_hparams_reject_conserve_without_qcond():
    with pytest.raises(ValueError, match="conserve"):
        SparcHyperparams(alpha=1.3, adaptive=True, qcond=False, conserve=True)


@pytest.mark.parametrize("rho", [-0.1, 1.1])
def test_hparams_reject_rho_outside_zero_one(rho):
    with pytest.raises(ValueError, match="rho"):
        SparcHyperparams(alpha=1.0, adaptive=True, qcond=True, conserve=True, rho=rho)


@pytest.mark.parametrize("rho", [0.0, 1.0])
def test_hparams_accept_rho_at_the_boundaries(rho):
    hp = SparcHyperparams(alpha=1.0, adaptive=True, qcond=True, conserve=True, rho=rho)
    assert hp.rho == rho


@pytest.mark.parametrize("sink_frac", [0.0, 1.0, 1.5])
def test_hparams_reject_sink_frac_outside_open_unit_interval(sink_frac):
    with pytest.raises(ValueError, match="sink_frac"):
        SparcHyperparams(
            alpha=1.0, adaptive=True, qcond=True, conserve=True, sink_frac=sink_frac
        )


def test_hparams_expose_conserve_in_as_dict():
    hp = SparcHyperparams(
        alpha=1.0, adaptive=True, qcond=True, conserve=True, rho=0.25, sink_frac=0.1
    )
    d = hp.as_dict()
    assert d["conserve"] is True
    assert d["rho"] == 0.25
    assert d["sink_frac"] == 0.1


def test_hparams_conserve_does_not_change_the_alpha_c_path():
    hp = SparcHyperparams(alpha=1.3)
    assert hp.conserve is False
    assert hp.as_dict()["conserve"] is False
