"""Tests for the adaptive-intensity path of SPARC (``adaptive=True``).

Four invariants, in the order the spec lists them:

1. Neutrality gate: ``lam=0`` must leave the value cache and the attention
   output bit-identical to a run where SPARC never touches the cache.
2. The ceiling is never exceeded, and a saturated target does not compound
   across steps the way ``alpha^c`` does.
3. A position that leaves the selection relaxes back to factor 1.0.
4. Non-regression: with ``adaptive=False`` the ``alpha^c`` path is untouched
   and the adaptive registry stays inert.
"""

from __future__ import annotations

import math
from functools import partial
from types import MethodType

import pytest
import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache

from vr_modality_bias.experiment.sparc import SparcHyperparams
from vr_modality_bias.utils.attn import (
    SelectedIndexBuffer,
    adaptive_target_factor,
    forward_llama,
)

# ---------------------------------------------------------------- fixtures

HIDDEN = 8
N_HEADS = 2
HEAD_DIM = 4
PREFILL_LEN = 6
# Non-contiguous on purpose: the registry is indexed locally, the cache write
# globally, and a contiguous block would hide a mix-up between the two.
IMAGE_POSITIONS = (1, 2, 4)
CEILING = 2.0

# ``(ratio >= tau).nonzero()`` with these bounds selects everything / nothing.
ALWAYS_SELECT = -1e9
NEVER_SELECT = 1e9


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
    """cos=1, sin=0 makes ``apply_rotary_pos_emb`` the identity.

    RoPE is not under test here; keeping it inert isolates the SPARC bookkeeping.
    """
    return torch.ones(1, q_len, HEAD_DIM), torch.zeros(1, q_len, HEAD_DIM)


def _fresh_layer(seed: int = 0) -> _TinyAttn:
    torch.manual_seed(seed)
    layer = _TinyAttn()
    layer.eval()
    return layer


def _fresh_buffer() -> SelectedIndexBuffer:
    buffer = SelectedIndexBuffer()
    buffer.reset()
    buffer.update_input_len(PREFILL_LEN - len(IMAGE_POSITIONS))
    buffer.update_image_positions(torch.tensor(IMAGE_POSITIONS, dtype=torch.long))
    return buffer


def _patch(
    layer: _TinyAttn,
    buffer: SelectedIndexBuffer,
    *,
    alpha: float = 1.0,
    beta: float = 0.1,
    tau: float = 2.0,
    selected: bool = True,
    adaptive: bool = False,
    lam: float = 0.0,
    ceiling: float = CEILING,
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
    )
    layer.forward = MethodType(forward_, layer)


def _inputs(n_steps: int, seed: int = 1):
    gen = torch.Generator().manual_seed(seed)
    prefill = torch.randn(1, PREFILL_LEN, HIDDEN, generator=gen)
    steps = [torch.randn(1, 1, HIDDEN, generator=gen) for _ in range(n_steps)]
    return prefill, steps


@torch.no_grad()
def _drive(layer, prefill, steps, buffer=None):
    """Prefill + one forward per step. Returns cache, outputs, post-prefill
    snapshot of the value cache, and the registry after each step."""
    cache = DynamicCache()
    outputs = [
        layer(
            hidden_states=prefill,
            past_key_values=cache,
            position_embeddings=_pos(PREFILL_LEN),
        )[0]
    ]
    snapshot = cache.layers[0].values.clone()
    registry = []
    for hidden in steps:
        outputs.append(
            layer(hidden_states=hidden, past_key_values=cache, position_embeddings=_pos(1))[0]
        )
        if buffer is not None and buffer.accum_factors is not None:
            registry.append(buffer.accum_factors.clone())
    return cache, outputs, snapshot, registry


def _image_rows_of(values: torch.Tensor) -> torch.Tensor:
    return values[:, :, torch.tensor(IMAGE_POSITIONS)]


def _image_rows(cache: DynamicCache) -> torch.Tensor:
    return _image_rows_of(cache.layers[0].values)


# ---------------------------------------------------------------- deficit signal


def test_target_is_one_when_attention_is_at_or_above_the_reference():
    reference = torch.tensor([[0.2, 0.4]])
    current = torch.tensor([[0.4, 0.6]])
    assert adaptive_target_factor(current, reference, lam=10.0, ceiling=CEILING) == 1.0


def test_target_scales_linearly_with_the_deficit():
    reference = torch.tensor([[0.5, 0.5]])
    current = torch.tensor([[0.25, 0.25]])  # deficit = 0.5
    target = adaptive_target_factor(current, reference, lam=0.5, ceiling=CEILING)
    assert target == pytest.approx(1.25)


def test_target_saturates_at_the_ceiling():
    reference = torch.tensor([[0.5, 0.5]])
    current = torch.tensor([[0.25, 0.25]])
    assert adaptive_target_factor(current, reference, lam=1e6, ceiling=CEILING) == CEILING


def test_target_aggregates_over_the_image_tokens():
    """Heads are already averaged upstream; the signal averages over tokens."""
    reference = torch.tensor([[0.4, 0.6]])  # mean 0.5
    current = torch.tensor([[0.1, 0.4]])  # mean 0.25 -> deficit 0.5
    assert adaptive_target_factor(current, reference, lam=1.0, ceiling=9.0) == pytest.approx(1.5)


def test_target_is_neutral_when_the_reference_is_zero():
    """0/0 -> NaN. A NaN target would land in the registry and stay in the cache."""
    reference = torch.zeros(1, 2)
    current = torch.zeros(1, 2)
    assert adaptive_target_factor(current, reference, lam=5.0, ceiling=CEILING) == 1.0


def test_target_is_neutral_when_the_reference_is_zero_and_attention_is_not():
    reference = torch.zeros(1, 2)
    current = torch.tensor([[0.3, 0.3]])
    assert adaptive_target_factor(current, reference, lam=5.0, ceiling=CEILING) == 1.0


def test_target_never_returns_a_non_finite_value():
    reference = torch.tensor([[0.0, 0.0]])
    current = torch.tensor([[float("nan"), 0.1]])
    target = adaptive_target_factor(current, reference, lam=5.0, ceiling=CEILING)
    assert math.isfinite(target)
    assert target == 1.0


# ---------------------------------------------------------------- registry


def test_update_image_positions_allocates_the_registry_at_one():
    buffer = _fresh_buffer()
    assert torch.equal(buffer.accum_factors, torch.ones(len(IMAGE_POSITIONS)))
    assert buffer.accum_factors.dtype is torch.float32
    assert buffer.correction is None


def test_reset_drops_the_registry():
    buffer = _fresh_buffer()
    buffer.reset()
    assert buffer.accum_factors is None
    assert buffer.correction is None
    assert buffer.target1 == 1.0
    assert buffer.target2 == 1.0
    assert buffer.indices1_local == []
    assert buffer.indices2_local == []


def test_adaptive_correction_requires_image_positions():
    buffer = SelectedIndexBuffer()
    buffer.reset()
    with pytest.raises(RuntimeError, match="update_image_positions"):
        buffer.prepare_adaptive_correction()


def test_saturated_target_does_not_compound_across_steps():
    """The whole point of the correction: repeating the target keeps the factor
    at the target, where the alpha^c path would have reached CEILING ** n."""
    buffer = _fresh_buffer()
    value = torch.ones(1, 1, PREFILL_LEN, 2)

    for _ in range(5):
        buffer.update_indices1(torch.tensor([[0], [2]], dtype=torch.long))
        buffer.update_target1(CEILING)
        buffer.update_indices2()
        buffer.prepare_adaptive_correction()
        buffer.calibrate_adaptive(value)
        assert float(buffer.accum_factors.max()) <= CEILING

    assert buffer.accum_factors.tolist() == [CEILING, 1.0, CEILING]
    scaled = value[0, 0, torch.tensor(IMAGE_POSITIONS), 0]
    assert scaled.tolist() == [CEILING, 1.0, CEILING]
    assert scaled.max() < CEILING**5


def test_position_relaxes_to_one_when_it_leaves_the_selection():
    buffer = _fresh_buffer()
    value = torch.ones(1, 1, PREFILL_LEN, 2)
    original = value.clone()

    # Step t: local 0 and 2 selected (global 1 and 4), target 1.5.
    buffer.update_indices1(torch.tensor([[0], [2]], dtype=torch.long))
    buffer.update_target1(1.5)
    buffer.update_indices2()
    buffer.prepare_adaptive_correction()
    buffer.calibrate_adaptive(value)
    assert buffer.accum_factors.tolist() == [1.5, 1.0, 1.5]
    assert value[0, 0, 1, 0] == 1.5
    assert value[0, 0, 4, 0] == 1.5
    assert value[0, 0, 2, 0] == 1.0  # image token, never selected

    # Step t+1: nothing selected, so every position relaxes back to 1.0.
    buffer.update_indices1(torch.empty((0, 1), dtype=torch.long))
    buffer.update_target1(1.0)
    buffer.update_indices2()
    buffer.prepare_adaptive_correction()
    buffer.calibrate_adaptive(value)
    assert torch.allclose(buffer.accum_factors, torch.ones(len(IMAGE_POSITIONS)))
    assert torch.allclose(value, original, atol=1e-6)


def test_correction_is_skipped_when_the_cache_already_carries_the_target():
    buffer = _fresh_buffer()
    buffer.update_indices2()
    buffer.prepare_adaptive_correction()
    assert buffer.correction is None


# ---------------------------------------------------------------- forward path


def test_neutrality_gate_lambda_zero_matches_a_run_without_sparc():
    prefill, steps = _inputs(5)

    reference_layer = _fresh_layer()
    reference_buffer = _fresh_buffer()
    # selected=False -> indices1 never fills -> calibrate never fires.
    _patch(reference_layer, reference_buffer, selected=False)
    reference_cache, reference_outputs, _, _ = _drive(reference_layer, prefill, steps)

    adaptive_layer = _fresh_layer()
    adaptive_buffer = _fresh_buffer()
    _patch(adaptive_layer, adaptive_buffer, adaptive=True, lam=0.0, tau=ALWAYS_SELECT)
    adaptive_cache, adaptive_outputs, _, _ = _drive(adaptive_layer, prefill, steps, adaptive_buffer)

    # Non-vacuous: the adaptive run DID select every image token each step, it
    # just resolved a target of 1.0 and therefore never wrote to the cache.
    assert len(adaptive_buffer.indices2_local) == len(IMAGE_POSITIONS)
    assert torch.equal(adaptive_cache.layers[0].values, reference_cache.layers[0].values)
    for got, want in zip(adaptive_outputs, reference_outputs, strict=True):
        assert torch.equal(got, want)
    assert torch.equal(adaptive_buffer.accum_factors, torch.ones(len(IMAGE_POSITIONS)))


def test_forward_never_exceeds_the_ceiling():
    prefill, steps = _inputs(8)
    layer = _fresh_layer()
    buffer = _fresh_buffer()
    _patch(layer, buffer, adaptive=True, lam=1e6, tau=ALWAYS_SELECT)
    cache, _, snapshot, registry = _drive(layer, prefill, steps, buffer)

    assert registry, "registry was never recorded"
    for factors in registry:
        assert float(factors.max()) <= CEILING + 1e-6
    # Non-vacuous: with lam=1e6 the ceiling is actually reached at some step.
    assert max(float(f.max()) for f in registry) == pytest.approx(CEILING)

    ratios = _image_rows(cache) / _image_rows_of(snapshot)
    assert float(ratios.max()) <= CEILING + 1e-6


def test_forward_relaxes_the_cache_once_selection_stops():
    prefill, steps = _inputs(3)
    _, tail_steps = _inputs(4, seed=2)

    layer = _fresh_layer()
    buffer = _fresh_buffer()
    _patch(layer, buffer, adaptive=True, lam=1e6, tau=ALWAYS_SELECT)
    cache, _, snapshot, _ = _drive(layer, prefill, steps, buffer)

    amplified = _image_rows(cache) / _image_rows_of(snapshot)
    assert float(amplified.max()) > 1.0, "nothing was amplified; test is vacuous"

    # Same buffer and cache, but nothing is selected from now on. The relaxation
    # has to fire even though indices2 goes empty.
    _patch(layer, buffer, adaptive=True, lam=1e6, tau=NEVER_SELECT)
    with torch.no_grad():
        for hidden in tail_steps:
            layer(hidden_states=hidden, past_key_values=cache, position_embeddings=_pos(1))

    restored = _image_rows(cache) / _image_rows_of(snapshot)
    assert torch.allclose(restored, torch.ones_like(restored), atol=1e-6)
    assert torch.allclose(buffer.accum_factors, torch.ones(len(IMAGE_POSITIONS)))


def test_alpha_c_path_still_compounds_when_adaptive_is_off():
    prefill, steps = _inputs(4)
    layer = _fresh_layer()
    buffer = _fresh_buffer()
    _patch(layer, buffer, alpha=1.5, tau=ALWAYS_SELECT, adaptive=False)
    cache, _, snapshot, _ = _drive(layer, prefill, steps, buffer)

    # calibrate fires on steps 2, 3 and 4 (step 1 has an empty indices2).
    ratios = _image_rows(cache) / _image_rows_of(snapshot)
    assert torch.allclose(ratios, torch.full_like(ratios, 1.5**3), rtol=1e-5)
    # The registry exists but the original path never advances it.
    assert torch.equal(buffer.accum_factors, torch.ones(len(IMAGE_POSITIONS)))
    assert buffer.correction is None


# ---------------------------------------------------------------- hyperparams


def test_hparams_defaults_keep_the_original_path():
    hparams = SparcHyperparams(alpha=1.3)
    assert hparams.adaptive is False
    assert hparams.lam == 0.0
    assert hparams.ceiling == 2.0


def test_hparams_adaptive_does_not_require_alpha_above_one():
    hparams = SparcHyperparams(alpha=1.0, adaptive=True, lam=0.5)
    assert hparams.adaptive is True
    assert hparams.lam == 0.5
    assert hparams.ceiling == 2.0


def test_hparams_adaptive_rejects_a_ceiling_at_or_below_one():
    with pytest.raises(ValueError, match="ceiling"):
        SparcHyperparams(alpha=1.0, adaptive=True, ceiling=1.0)


def test_hparams_adaptive_rejects_a_negative_lam():
    with pytest.raises(ValueError, match="lam"):
        SparcHyperparams(alpha=1.0, adaptive=True, lam=-0.1)


def test_hparams_non_adaptive_ignores_lam_and_ceiling():
    hparams = SparcHyperparams(alpha=1.3, lam=-5.0, ceiling=0.5)
    assert hparams.adaptive is False
    assert hparams.lam == -5.0
    assert hparams.ceiling == 0.5


def test_hparams_non_adaptive_still_rejects_alpha_at_one():
    with pytest.raises(ValueError, match="alpha"):
        SparcHyperparams(alpha=1.0, adaptive=False)
