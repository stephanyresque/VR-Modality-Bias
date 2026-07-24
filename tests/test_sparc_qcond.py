"""Tests for the question-conditioned selection of SPARC (``qcond=True``), v1.1.

Eight invariants:

1. Contrastive relevance picks the question-specific column and drops the sink.
2. A degenerate contrast falls back to the raw question attention, never empty.
3. ``prefill_target`` is the unconditional ``min(1 + lam, ceiling)``.
4. The reinforcement lands INSIDE the prefill, on the post-reference layers only.
5. ``accum_factors`` agrees with what the prefill wrote, so decode step 1 is a
   null correction and step 2 corrects against the prefill factor.
6. With ``qcond`` no decode calibration touches a layer at or below the reference.
7. Neutrality gate: ``lam=0`` leaves everything bit-identical.
8. Non-regression: ``qcond=False`` and the ``alpha^c`` path are untouched, and an
   impossible calibration window is rejected.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache

from vr_modality_bias.experiment.sparc import SparcHyperparams
from vr_modality_bias.utils.attn import (
    SelectedIndexBuffer,
    _rows_to_image_attention,
    add_custom_attention_layers,
    decoder_of,
    prefill_target,
    question_conditioned_selection,
)

# ---------------------------------------------------------------- fixtures

HIDDEN = 8
N_HEADS = 2
HEAD_DIM = 4
PREFILL_LEN = 10
N_LAYERS = 4
# Reference at 1 leaves layers 2 and 3 above it (the boost window) and layers
# 0 and 1 below it (which must never be written).
SELECTED_LAYER = 1
SE_LAYERS = (0, N_LAYERS - 1)
BOOST_LAYERS = (2, 3)
UNTOUCHED_LAYERS = (0, 1)

# Non-contiguous on purpose (the SmolVLM layout).
IMAGE_POSITIONS = (1, 2, 4, 5, 6)
QUESTION_POSITIONS = (7, 8, 9)
# floor(0.4 * 5) = 2.
QTOP_FRAC = 0.4
EXPECTED_K = 2
CEILING = 2.0
BETA = 0.1

ALWAYS_SELECT = -1e9
NEVER_SELECT = 1e9

IMAGE_POS_T = torch.tensor(IMAGE_POSITIONS, dtype=torch.long)
QUESTION_POS_T = torch.tensor(QUESTION_POSITIONS, dtype=torch.long)


class _TinyAttn(nn.Module):
    """Smallest module satisfying what ``forward_llama`` reads off ``self``."""

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
    """``model.model.text_model.layers`` is the Idefics3 path ``decoder_of`` walks."""

    def __init__(self, n_layers: int = N_LAYERS):
        super().__init__()
        self.model = SimpleNamespace(text_model=_MockTextModel(n_layers))


def _pos(q_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """cos=1, sin=0 keeps RoPE inert; it is not what is under test."""
    return torch.ones(1, q_len, HEAD_DIM), torch.zeros(1, q_len, HEAD_DIM)


def _fresh_buffer(*, with_question_positions: bool = True) -> SelectedIndexBuffer:
    buffer = SelectedIndexBuffer()
    buffer.reset()
    buffer.update_input_len(PREFILL_LEN - len(IMAGE_POSITIONS))
    buffer.update_image_positions(IMAGE_POS_T.clone())
    if with_question_positions:
        buffer.update_question_positions(QUESTION_POS_T.clone())
    return buffer


def _build_stack(
    buffer: SelectedIndexBuffer,
    *,
    seed: int = 0,
    qcond: bool = True,
    adaptive: bool = True,
    lam: float = 0.5,
    ceiling: float = CEILING,
    alpha: float = 1.0,
    tau: float = ALWAYS_SELECT,
    qtop_frac: float = QTOP_FRAC,
    selected_layer: int = SELECTED_LAYER,
    se_layers: tuple[int, int] = SE_LAYERS,
) -> _MockModel:
    """Patch a real 4-layer stack through ``add_custom_attention_layers``.

    Going through the installer rather than a hand-made partial is what puts the
    se_layers narrowing of change 1e under test.
    """
    torch.manual_seed(seed)
    model = _MockModel()
    model.eval()
    add_custom_attention_layers(
        model,
        alpha=alpha,
        beta=BETA,
        tau=tau,
        selected_layer=selected_layer,
        se_layers=se_layers,
        image_token_index=IMAGE_POSITIONS[0],
        indices_buffer=buffer,
        adaptive=adaptive,
        lam=lam,
        ceiling=ceiling,
        qcond=qcond,
        qtop_frac=qtop_frac,
    )
    return model


def _no_sparc_stack(buffer: SelectedIndexBuffer, *, seed: int = 0) -> _MockModel:
    """A stack that never writes to the cache: nothing is ever selected."""
    return _build_stack(
        buffer, seed=seed, qcond=False, adaptive=False, alpha=1.0, tau=NEVER_SELECT
    )


def _inputs(n_steps: int, seed: int = 3):
    gen = torch.Generator().manual_seed(seed)
    prefill = torch.randn(1, PREFILL_LEN, HIDDEN, generator=gen)
    steps = [torch.randn(1, 1, HIDDEN, generator=gen) for _ in range(n_steps)]
    return prefill, steps


@torch.no_grad()
def _drive(model: _MockModel, prefill, steps):
    """Run the stack. Returns per-layer attention outputs and cache snapshots.

    ``snapshots[0]`` is the value cache right after the prefill, ``snapshots[t]``
    after decode step ``t``.
    """
    attns = [layer.self_attn for layer in decoder_of(model).layers]
    cache = DynamicCache()

    def sweep(hidden):
        outputs = []
        for attn in attns:
            hidden = attn(
                hidden_states=hidden,
                past_key_values=cache,
                position_embeddings=_pos(hidden.shape[1]),
            )[0]
            outputs.append(hidden)
        return outputs

    def snapshot():
        return [cache.layers[i].values.clone() for i in range(N_LAYERS)]

    prefill_outputs = sweep(prefill)
    snapshots = [snapshot()]
    step_outputs = []
    for hidden in steps:
        step_outputs.append(sweep(hidden))
        snapshots.append(snapshot())
    return cache, prefill_outputs, step_outputs, snapshots


def _image_rows(values: torch.Tensor) -> torch.Tensor:
    return values[:, :, IMAGE_POS_T]


def _synthetic_attention(rows_to_image: dict) -> torch.Tensor:
    """Prefill attention whose row-to-image-column entries are exactly given.

    ``rows_to_image`` maps a row index to the five per-image-column values. Rows
    absent from the mapping give zero to every image column.
    """
    attn = torch.zeros(1, N_HEADS, PREFILL_LEN, PREFILL_LEN)
    for head in range(N_HEADS):
        for row, values in rows_to_image.items():
            for local, column in enumerate(IMAGE_POSITIONS):
                attn[0, head, row, column] = values[local]
    return attn


def _raw_selection(attn, k: int) -> list[int]:
    """The v1 signal: top-k by raw question attention, no background subtracted."""
    relevance = _rows_to_image_attention(attn, IMAGE_POS_T, QUESTION_POS_T)
    return torch.topk(relevance, k).indices.tolist()


def _jaccard(a, b) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb)


# ---------------------------------------------------------------- 1. contrastive


# Local 0 is a sink: every row of the prompt pays it 0.9. Local 3 is what this
# question actually looks at: only the question rows pay it, and less than the
# sink gets. Raw attention therefore ranks the sink first; the contrast cancels it.
_SINK_LOCAL = 0
_RELEVANT_LOCAL = 3
_SINK_VALUE = 0.9
_RELEVANT_VALUE = 0.5


def _sink_and_relevant_attention(relevant_local: int, second_local: int) -> torch.Tensor:
    background = [0.0] * 5
    background[_SINK_LOCAL] = _SINK_VALUE
    question = list(background)
    question[relevant_local] = _RELEVANT_VALUE
    question[second_local] = _RELEVANT_VALUE - 0.1
    rows = {row: list(background) for row in range(PREFILL_LEN)}
    for row in QUESTION_POSITIONS:
        rows[row] = question
    return _synthetic_attention(rows)


def test_contrastive_relevance_drops_the_sink_and_keeps_the_relevant_column():
    attn = _sink_and_relevant_attention(_RELEVANT_LOCAL, 1)
    contrastive = question_conditioned_selection(attn, IMAGE_POS_T, QUESTION_POS_T, 0.2)
    assert contrastive.tolist() == [_RELEVANT_LOCAL]
    # The v1 signal would have picked the sink instead.
    assert _raw_selection(attn, 1) == [_SINK_LOCAL]


def test_contrastive_selection_varies_with_the_question_where_raw_does_not():
    """Jaccard between two questions on the same image: the whole v1 failure."""
    attn_a = _sink_and_relevant_attention(3, 1)
    attn_b = _sink_and_relevant_attention(4, 2)

    contrastive_a = question_conditioned_selection(attn_a, IMAGE_POS_T, QUESTION_POS_T, QTOP_FRAC)
    contrastive_b = question_conditioned_selection(attn_b, IMAGE_POS_T, QUESTION_POS_T, QTOP_FRAC)
    raw_a = _raw_selection(attn_a, EXPECTED_K)
    raw_b = _raw_selection(attn_b, EXPECTED_K)

    contrastive_overlap = _jaccard(contrastive_a.tolist(), contrastive_b.tolist())
    raw_overlap = _jaccard(raw_a, raw_b)
    assert contrastive_overlap < raw_overlap
    assert contrastive_overlap == 0.0  # {3,1} vs {4,2}
    assert _SINK_LOCAL in raw_a and _SINK_LOCAL in raw_b


def test_contrastive_relevance_is_clamped_at_zero():
    """A column the background likes MORE than the question must not go negative
    and outrank a column with a genuine, small positive contrast."""
    rows = {row: [0.0] * 5 for row in range(PREFILL_LEN)}
    for row in range(PREFILL_LEN):
        rows[row][2] = 0.9  # background-only column
    for row in QUESTION_POSITIONS:
        rows[row] = [0.0, 0.0, 0.1, 0.05, 0.0]  # question pays column 2 LESS
    attn = _synthetic_attention(rows)
    local = question_conditioned_selection(attn, IMAGE_POS_T, QUESTION_POS_T, 0.2)
    assert local.tolist() == [3]


# ---------------------------------------------------------------- 2. fallback


def test_fallback_to_raw_relevance_when_the_background_dominates_everywhere():
    """Every column is attended more by the background than by the question, so
    the clamped contrast is zero everywhere and carries no ranking."""
    question = [0.1, 0.5, 0.2, 0.9, 0.3]
    rows = {row: [v + 0.1 for v in question] for row in range(PREFILL_LEN)}
    for row in QUESTION_POSITIONS:
        rows[row] = question
    attn = _synthetic_attention(rows)
    local = question_conditioned_selection(attn, IMAGE_POS_T, QUESTION_POS_T, QTOP_FRAC)
    assert local.tolist() == [3, 1]  # the raw ranking
    assert local.tolist() == _raw_selection(attn, EXPECTED_K)


def test_fallback_when_every_row_attends_the_image_identically():
    """Exact-zero contrast. Dyadic values so the row means are bit-equal across
    the 3 question rows and the 7 background rows; otherwise the residue of
    ``sum/n`` survives the clamp and ranks the top-k on floating-point noise."""
    values = [0.125, 0.5, 0.25, 0.75, 0.375]
    attn = _synthetic_attention({row: values for row in range(PREFILL_LEN)})
    local = question_conditioned_selection(attn, IMAGE_POS_T, QUESTION_POS_T, QTOP_FRAC)
    assert local.tolist() == [3, 1]
    assert local.tolist() == _raw_selection(attn, EXPECTED_K)


def test_selection_is_never_empty():
    attn = _synthetic_attention({row: [0.0] * 5 for row in range(PREFILL_LEN)})
    assert len(question_conditioned_selection(attn, IMAGE_POS_T, QUESTION_POS_T, 0.01)) == 1


def test_selection_zeroes_non_finite_relevance_before_the_top_k():
    attn = _sink_and_relevant_attention(_RELEVANT_LOCAL, 1)
    for head in range(N_HEADS):
        attn[0, head, QUESTION_POSITIONS[0], IMAGE_POSITIONS[_RELEVANT_LOCAL]] = float("nan")
    local = question_conditioned_selection(attn, IMAGE_POS_T, QUESTION_POS_T, 0.2)
    assert _RELEVANT_LOCAL not in local.tolist()


def test_k_is_floor_of_the_fraction_with_a_floor_of_one():
    attn = _synthetic_attention({row: [0.1, 0.5, 0.2, 0.9, 0.3] for row in range(PREFILL_LEN)})
    assert len(question_conditioned_selection(attn, IMAGE_POS_T, QUESTION_POS_T, 0.01)) == 1
    assert len(question_conditioned_selection(attn, IMAGE_POS_T, QUESTION_POS_T, 0.4)) == 2
    assert len(question_conditioned_selection(attn, IMAGE_POS_T, QUESTION_POS_T, 1.0)) == 5


def test_buffer_translates_the_local_top_k_through_noncontiguous_positions():
    attn = _sink_and_relevant_attention(3, 1)
    local = question_conditioned_selection(attn, IMAGE_POS_T, QUESTION_POS_T, QTOP_FRAC)
    buffer = _fresh_buffer()
    buffer.update_indices1(local.unsqueeze(-1), image_token_index=IMAGE_POSITIONS[0])
    assert buffer.indices1_local.tolist() == [3, 1]
    assert buffer.indices1.tolist() == [5, 2]  # locals 3 and 1 -> globals 5 and 2


def test_single_selected_token_stays_one_dimensional():
    attn = _sink_and_relevant_attention(3, 1)
    local = question_conditioned_selection(attn, IMAGE_POS_T, QUESTION_POS_T, 0.01)
    buffer = _fresh_buffer()
    buffer.update_indices1(local.unsqueeze(-1), image_token_index=IMAGE_POSITIONS[0])
    assert buffer.indices1_local.dim() == 1
    assert buffer.indices1.tolist() == [5]


# ---------------------------------------------------------------- 3. prefill target


def test_prefill_target_is_unconditional():
    assert prefill_target(0.5, 2.0) == 1.5
    assert prefill_target(0.0, 2.0) == 1.0
    assert prefill_target(5.0, 2.0) == 2.0


def test_prefill_target_is_a_plain_float():
    assert isinstance(prefill_target(0.5, 2.0), float)


# ---------------------------------------------------------------- 4. prefill boost

LAM = 0.5
PREFILL_FACTOR = 1.5  # min(1 + 0.5, 2.0)


@torch.no_grad()
def _unboosted_values(attn_module, hidden_in):
    """What this layer's ``v_proj`` produced before the boost touched it."""
    projected = attn_module.v_proj(hidden_in)
    return projected.view(1, hidden_in.shape[1], -1, HEAD_DIM).transpose(1, 2)


def test_prefill_boost_scales_only_the_selected_rows_of_post_reference_layers():
    prefill, _ = _inputs(0)
    buffer = _fresh_buffer()
    model = _build_stack(buffer, lam=LAM)
    _, outputs, _, snapshots = _drive(model, prefill, [])

    selected_globals = set(buffer.indices1.tolist())
    assert len(selected_globals) == EXPECTED_K

    attns = [layer.self_attn for layer in decoder_of(model).layers]
    hidden_ins = [prefill, *outputs[:-1]]

    for layer_idx in range(N_LAYERS):
        # Each layer is compared against ITS OWN unboosted projection, not
        # against a separate no-SPARC run: above layer selected_layer+1 the
        # hidden input has already changed, so every row of v_proj differs and a
        # cross-run ratio would not isolate the boost.
        expected = _unboosted_values(attns[layer_idx], hidden_ins[layer_idx])
        got = snapshots[0][layer_idx]
        for position in range(PREFILL_LEN):
            boosted = layer_idx in BOOST_LAYERS and position in selected_globals
            if boosted:
                assert torch.allclose(
                    got[:, :, position], expected[:, :, position] * PREFILL_FACTOR
                ), f"layer {layer_idx}, position {position}"
            else:
                assert torch.equal(got[:, :, position], expected[:, :, position]), (
                    f"layer {layer_idx}, position {position} moved but should not have"
                )


def test_layers_up_to_the_reference_are_byte_identical_to_a_run_without_sparc():
    prefill, _ = _inputs(0)
    reference_buffer = _fresh_buffer()
    _, _, _, reference_snapshots = _drive(_no_sparc_stack(reference_buffer), prefill, [])
    buffer = _fresh_buffer()
    _, _, _, snapshots = _drive(_build_stack(buffer, lam=LAM), prefill, [])
    for layer_idx in UNTOUCHED_LAYERS:
        assert torch.equal(snapshots[0][layer_idx], reference_snapshots[0][layer_idx])


def test_prefill_boost_changes_the_last_prompt_position_output():
    """The answer token of a VQA prompt is decided from this position."""
    prefill, _ = _inputs(0)

    reference_buffer = _fresh_buffer()
    _, reference_outputs, _, _ = _drive(_no_sparc_stack(reference_buffer), prefill, [])

    buffer = _fresh_buffer()
    _, outputs, _, _ = _drive(_build_stack(buffer, lam=LAM), prefill, [])

    for layer_idx in UNTOUCHED_LAYERS:
        assert torch.equal(outputs[layer_idx], reference_outputs[layer_idx])
    for layer_idx in BOOST_LAYERS:
        assert not torch.equal(
            outputs[layer_idx][0, -1, :], reference_outputs[layer_idx][0, -1, :]
        ), f"layer {layer_idx} last-position output did not move"


def test_reference_layer_cannot_boost_itself():
    """Its own value states are committed before its attention picks the top-k."""
    prefill, _ = _inputs(0)
    reference_buffer = _fresh_buffer()
    _, _, _, reference_snapshots = _drive(_no_sparc_stack(reference_buffer), prefill, [])
    buffer = _fresh_buffer()
    _, _, _, snapshots = _drive(_build_stack(buffer, lam=LAM), prefill, [])
    assert torch.equal(snapshots[0][SELECTED_LAYER], reference_snapshots[0][SELECTED_LAYER])


# ---------------------------------------------------------------- 5. registry


def test_registry_matches_what_the_prefill_wrote():
    prefill, _ = _inputs(0)
    buffer = _fresh_buffer()
    _drive(_build_stack(buffer, lam=LAM), prefill, [])

    selected = buffer.prefill_selected_local.tolist()
    for local in range(len(IMAGE_POSITIONS)):
        want = PREFILL_FACTOR if local in selected else 1.0
        assert float(buffer.accum_factors[local]) == pytest.approx(want)


def test_decode_step_one_is_a_null_correction():
    """The prefill already applied the factor; step 1 must not apply it again."""
    prefill, steps = _inputs(1)
    buffer = _fresh_buffer()
    _, _, _, snapshots = _drive(_build_stack(buffer, lam=LAM), prefill, steps)

    assert buffer.target2 == PREFILL_FACTOR
    for layer_idx in range(N_LAYERS):
        assert torch.equal(
            _image_rows(snapshots[1][layer_idx]), _image_rows(snapshots[0][layer_idx])
        ), f"layer {layer_idx} was written at step 1"


def test_decode_step_two_corrects_against_the_prefill_factor():
    prefill, steps = _inputs(2)
    buffer = _fresh_buffer()
    _, _, _, snapshots = _drive(_build_stack(buffer, lam=LAM), prefill, steps)

    applied = buffer.target2  # the adaptive target decided at step 1
    expected_ratio = applied / PREFILL_FACTOR
    selected = buffer.prefill_selected_local.tolist()

    for layer_idx in BOOST_LAYERS:
        ratios = _image_rows(snapshots[2][layer_idx]) / _image_rows(snapshots[1][layer_idx])
        for local in range(len(IMAGE_POSITIONS)):
            want = expected_ratio if local in selected else 1.0
            assert ratios[0, 0, local, 0] == pytest.approx(want, rel=1e-5)
    # The registry now carries the step's target, not a product.
    assert float(buffer.accum_factors.max()) == pytest.approx(max(applied, 1.0))


# ---------------------------------------------------------------- 6. decode window


def test_decode_never_calibrates_at_or_below_the_reference_layer():
    prefill, steps = _inputs(3)
    buffer = _fresh_buffer()
    _, _, _, snapshots = _drive(_build_stack(buffer, lam=LAM), prefill, steps)

    for layer_idx in UNTOUCHED_LAYERS:
        for step in range(1, len(snapshots)):
            assert torch.equal(
                _image_rows(snapshots[step][layer_idx]),
                _image_rows(snapshots[0][layer_idx]),
            ), f"layer {layer_idx} written at step {step}"


def test_installer_narrows_the_calibration_window_under_qcond():
    buffer = _fresh_buffer()
    model = _build_stack(buffer, lam=LAM)
    for layer in decoder_of(model).layers:
        keywords = layer.self_attn.forward.__func__.keywords
        assert keywords["se_layers"] == (SELECTED_LAYER + 1, SE_LAYERS[1])
        assert keywords["selected_layer"] == SELECTED_LAYER


def test_installer_leaves_the_window_alone_without_qcond():
    buffer = _fresh_buffer()
    model = _build_stack(buffer, qcond=False, adaptive=False, alpha=1.5)
    for layer in decoder_of(model).layers:
        assert layer.self_attn.forward.__func__.keywords["se_layers"] == SE_LAYERS


def test_installer_rejects_a_window_with_no_layer_above_the_reference():
    buffer = _fresh_buffer()
    with pytest.raises(ValueError, match="no layer above the reference"):
        _build_stack(buffer, selected_layer=SE_LAYERS[1])


# ---------------------------------------------------------------- 7. neutrality


def test_neutrality_gate_lambda_zero_with_qcond_matches_a_run_without_sparc():
    prefill, steps = _inputs(3)

    reference_buffer = _fresh_buffer()
    reference_cache, reference_outputs, reference_steps, _ = _drive(
        _no_sparc_stack(reference_buffer), prefill, steps
    )

    buffer = _fresh_buffer()
    cache, outputs, step_outputs, _ = _drive(_build_stack(buffer, lam=0.0), prefill, steps)

    # Non-vacuous: the prefill DID select, it just resolved a factor of 1.0.
    assert len(buffer.prefill_selected_local) == EXPECTED_K
    assert buffer.target2 == 1.0
    assert torch.equal(buffer.accum_factors, torch.ones(len(IMAGE_POSITIONS)))

    for layer_idx in range(N_LAYERS):
        assert torch.equal(cache.layers[layer_idx].values, reference_cache.layers[layer_idx].values)
    for got, want in zip(outputs, reference_outputs, strict=True):
        assert torch.equal(got, want)
    for got_step, want_step in zip(step_outputs, reference_steps, strict=True):
        for got, want in zip(got_step, want_step, strict=True):
            assert torch.equal(got, want)


# ---------------------------------------------------------------- 8. non-regression


def test_qcond_false_keeps_the_prefill_untouched():
    """Point 1 alone never writes during the prefill, nor at step 1."""
    prefill, steps = _inputs(1)

    reference_buffer = _fresh_buffer()
    _, _, _, reference_snapshots = _drive(_no_sparc_stack(reference_buffer), prefill, [])

    buffer = _fresh_buffer()
    _, _, _, snapshots = _drive(
        _build_stack(buffer, qcond=False, lam=1e6, tau=ALWAYS_SELECT), prefill, steps
    )

    assert buffer.prefill_selected_local is None
    for layer_idx in range(N_LAYERS):
        assert torch.equal(snapshots[0][layer_idx], reference_snapshots[0][layer_idx])
        assert torch.equal(
            _image_rows(snapshots[1][layer_idx]), _image_rows(snapshots[0][layer_idx])
        )


def test_alpha_c_path_is_untouched():
    prefill, steps = _inputs(4)
    buffer = _fresh_buffer()
    _, _, _, snapshots = _drive(
        _build_stack(buffer, qcond=False, adaptive=False, alpha=1.5, tau=ALWAYS_SELECT),
        prefill,
        steps,
    )
    # calibrate fires on steps 2, 3 and 4, on every layer of the untouched window.
    for layer_idx in range(N_LAYERS):
        ratios = _image_rows(snapshots[-1][layer_idx]) / _image_rows(snapshots[0][layer_idx])
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


def test_hparams_reject_a_window_with_no_layer_above_the_reference():
    with pytest.raises(ValueError, match="no layer above the reference"):
        SparcHyperparams(
            alpha=1.0, adaptive=True, qcond=True, selected_layer=31, se_layers=(0, 31)
        )


def test_hparams_accept_a_window_with_exactly_one_layer_above():
    hparams = SparcHyperparams(
        alpha=1.0, adaptive=True, qcond=True, selected_layer=30, se_layers=(0, 31)
    )
    assert hparams.qcond is True


def test_hparams_default_qtop_frac_is_five_percent():
    assert SparcHyperparams(alpha=1.3).qtop_frac == 0.05


def test_hparams_ignore_the_window_check_when_qcond_is_off():
    assert SparcHyperparams(alpha=1.3, selected_layer=31, se_layers=(0, 31)).qcond is False


def test_hparams_expose_qcond_in_as_dict():
    hparams = SparcHyperparams(
        alpha=1.0, adaptive=True, qcond=True, qtop_frac=0.25, selected_layer=20
    )
    assert hparams.as_dict()["qcond"] is True
    assert hparams.as_dict()["qtop_frac"] == 0.25


# ---------------------------------------------------------------- guard


def test_qcond_without_question_positions_raises_at_the_prefill():
    prefill, _ = _inputs(0)
    buffer = _fresh_buffer(with_question_positions=False)
    model = _build_stack(buffer, lam=LAM)
    with pytest.raises(RuntimeError, match="update_question_positions"):
        _drive(model, prefill, [])


def test_commit_prefill_factor_needs_a_registry():
    buffer = SelectedIndexBuffer()
    buffer.reset()
    with pytest.raises(RuntimeError, match="update_image_positions"):
        buffer.commit_prefill_factor()


def test_commit_prefill_factor_needs_a_selection():
    buffer = _fresh_buffer()
    with pytest.raises(RuntimeError, match="empty prefill selection"):
        buffer.commit_prefill_factor()


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


@pytest.mark.parametrize("script_name", ["phase3_generate"])
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


@pytest.mark.parametrize("script_name", ["phase3_generate"])
def test_question_positions_follow_the_last_image_token_interleaved(script_name):
    """Idefics3 / SmolVLM layout: separators sit between the image tokens."""
    script = _load_script(script_name)
    ids = [1, 42, 99, 42, 99, 42, 7, 8]  # images at 1, 3, 5; question at 6..7
    input_len, image_positions, question_positions = script._probe_sparc_layout(
        _MockWrapper(ids), None, "q"
    )
    assert image_positions.tolist() == [1, 3, 5]
    assert question_positions.tolist() == [6, 7]
    assert input_len == len(ids) - 3


@pytest.mark.parametrize("script_name", ["phase3_generate"])
def test_qtop_frac_cli_default_is_five_percent(script_name):
    script = _load_script(script_name)
    args = script.build_parser().parse_args([])
    assert args.qtop_frac == 0.05
