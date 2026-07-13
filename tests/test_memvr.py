"""Tests for the MemVR core (``utils/memvr.py`` + ``experiment/memvr.py``), v1.

Coverage:

1. Entropy: known values (uniform top-10 -> 1.0, one-hot -> ~0), top-k only, clamp.
2. Window: depth-fraction default, the L-2 cap, override, empty-after-cap raise.
3. Adapter/NaN guard: alpha=0, Z=None, zero Z, zero adapter all short-circuit to
   the pure FFN bit-for-bit; a normal mix moves the output and stays finite.
4. Trigger: fires inside the window, not outside, at most once per forward,
   rearms next forward, injects on the armed layer only, respects the L-2 edge.
5. Z capture: shape/dtype/values at the prefill, not recaptured on decode,
   requires image_positions.
6. Neutrality: gamma=1.0 never fires and is bit-identical; no hooks persist.
7. Restoration by the context manager on normal exit and on exception.
8. Coexistence with ``enable_sparc`` (both patches active + full restoration).
9. ``get_final_norm`` discovery on the wrappers.
"""

from __future__ import annotations

import functools
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from vr_modality_bias.experiment.memvr import (
    MemVRHyperparams,
    enable_memvr,
    resolve_effective_window,
)
from vr_modality_bias.utils.attn import decoder_of
from vr_modality_bias.utils.memvr import (
    MemVRBuffer,
    _mlp_module_of,
    memvr_adapter_mix,
    normalized_topk_entropy,
)

# ---------------------------------------------------------------- fixtures

HIDDEN = 8
INTER = 16
VOCAB = 20
N_LAYERS = 6
# Non-contiguous on purpose (the SmolVLM interleaved layout).
IMAGE_POSITIONS = (1, 2, 4)
IMG_ID = 99

# For the SPARC coexistence mock: N_HEADS * HEAD_DIM == HIDDEN.
N_HEADS = 2
HEAD_DIM = 4

IMAGE_POS_T = torch.tensor(IMAGE_POSITIONS, dtype=torch.long)


class _TinyMLP(nn.Module):
    """Llama-style MLP; ``up_proj``/``down_proj`` are the injection anchors."""

    def __init__(self, hidden: int = HIDDEN, inter: int = INTER):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _MLPBlock(nn.Module):
    """Smallest decoder block for the MemVR-only tests (no attention needed)."""

    def __init__(self, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.mlp = _TinyMLP()

    def forward(self, hidden_states, **kwargs):
        # Returns a bare tensor (transformers 5.x style) to exercise the
        # entropy hook's non-tuple output path.
        return hidden_states + self.mlp(hidden_states)


class _TinyAttn(nn.Module):
    """SPARC-compatible attention (matches test_sparc_qcond's mock)."""

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


class _FullBlock(nn.Module):
    """Decoder block with both self_attn (SPARC) and mlp (MemVR) for coexistence."""

    def __init__(self, layer_idx: int):
        super().__init__()
        self.self_attn = _TinyAttn(layer_idx)
        self.mlp = _TinyMLP()

    def forward(self, hidden_states, past_key_values=None, position_embeddings=None, **kw):
        attn_out = self.self_attn(
            hidden_states=hidden_states,
            past_key_values=past_key_values,
            position_embeddings=position_embeddings,
        )[0]
        h = hidden_states + attn_out
        h = h + self.mlp(h)
        return (h,)


class _TextModel(nn.Module):
    def __init__(self, n_layers, block_cls):
        super().__init__()
        self.layers = nn.ModuleList([block_cls(i) for i in range(n_layers)])
        self.norm = nn.LayerNorm(HIDDEN)  # the final norm the logit lens uses


class _Inner(nn.Module):
    def __init__(self, n_layers, block_cls):
        super().__init__()
        self.text_model = _TextModel(n_layers, block_cls)


class _Model(nn.Module):
    """``model.model.text_model.layers`` (+ ``.norm``) is the Idefics3 path."""

    def __init__(self, n_layers: int = N_LAYERS, block_cls=_MLPBlock):
        super().__init__()
        self.model = _Inner(n_layers, block_cls)
        self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)
        self.config = SimpleNamespace(image_token_id=IMG_ID)


class _MockWrapper:
    def __init__(self, model: _Model):
        self._model = model

    @property
    def n_layers(self) -> int:
        return len(decoder_of(self._model).layers)

    def get_final_norm(self):
        return decoder_of(self._model).norm

    def get_lm_head(self):
        return self._model.lm_head


def _build_model(n_layers: int = N_LAYERS, seed: int = 0, block_cls=_MLPBlock) -> _Model:
    torch.manual_seed(seed)
    model = _Model(n_layers, block_cls)
    model.eval()
    return model


def _prefill(seq: int = 6, seed: int = 3) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(1, seq, HIDDEN, generator=gen)


def _decode(seed: int = 4) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(1, 1, HIDDEN, generator=gen)


def _pos(q_len: int):
    """cos=1, sin=0 keeps RoPE inert for the SPARC forward."""
    return torch.ones(1, q_len, HEAD_DIM), torch.zeros(1, q_len, HEAD_DIM)


@torch.no_grad()
def _sweep(model: _Model, hidden: torch.Tensor):
    """Run each decoder layer in order; return the per-layer output tensors."""
    outs = []
    h = hidden
    for layer in decoder_of(model).layers:
        out = layer(h)
        h = out[0] if isinstance(out, (tuple, list)) else out
        outs.append(h)
    return outs


def _snapshot(buffer: MemVRBuffer) -> SimpleNamespace:
    """Copy the instrumentation before enable_memvr's exit resets the buffer."""
    return SimpleNamespace(
        n_fires_total=buffer.n_fires_total,
        fired_in_prefill=buffer.fired_in_prefill,
        fire_layer=buffer.fire_layer,
        fire_entropy=buffer.fire_entropy,
        armed_layer=buffer.armed_layer,
        Z=None if buffer.Z is None else buffer.Z.clone(),
    )


def _run_memvr(
    model, hidden, *, gamma, alpha, window, image_positions=IMAGE_POS_T, top_k=10
):
    """One forward under MemVR. ``gamma`` may be negative to force a fire."""
    wrapper = _MockWrapper(model)
    hp = MemVRHyperparams(gamma=max(gamma, 0.0), alpha=alpha, window=window, top_k=top_k)
    with enable_memvr(wrapper, hp) as buffer:
        buffer.gamma = gamma
        buffer.update_image_positions(image_positions.clone())
        outs = _sweep(model, hidden)
        snap = _snapshot(buffer)
    return snap, outs


def _sparc_active(layer) -> bool:
    """SPARC installs ``MethodType(partial(forward_llama, ...), attn)``."""
    fwd = layer.self_attn.forward
    return isinstance(getattr(fwd, "__func__", None), functools.partial)


# ---------------------------------------------------------------- 1. entropy


def test_uniform_top10_entropy_is_one():
    logits = torch.zeros(1, 10)
    assert float(normalized_topk_entropy(logits, top_k=10)) == pytest.approx(1.0, abs=1e-6)


def test_one_hot_entropy_is_near_zero():
    logits = torch.full((1, 12), -50.0)
    logits[0, 0] = 50.0
    assert float(normalized_topk_entropy(logits, top_k=10)) < 1e-3


def test_entropy_uses_only_the_top_k():
    """Ten equal top logits plus a long tail: entropy is 1.0 from the top-10."""
    logits = torch.cat([torch.zeros(1, 10), torch.full((1, 30), -100.0)], dim=1)
    assert float(normalized_topk_entropy(logits, top_k=10)) == pytest.approx(1.0, abs=1e-6)


def test_entropy_is_clamped_to_the_unit_interval():
    for seed in range(5):
        logits = torch.randn(1, 50, generator=torch.Generator().manual_seed(seed))
        val = float(normalized_topk_entropy(logits, top_k=10))
        assert 0.0 <= val <= 1.0


# ---------------------------------------------------------------- 2. window


def test_default_window_is_the_depth_fraction():
    assert resolve_effective_window(24) == (4, 12)
    assert resolve_effective_window(32) == (5, 16)
    assert resolve_effective_window(12) == (2, 6)


def test_window_end_is_capped_at_l_minus_two():
    # Override end beyond the cap: it clamps to L-2, start is preserved.
    assert resolve_effective_window(10, window=(3, 20)) == (3, 8)


def test_window_override_is_respected_within_bounds():
    assert resolve_effective_window(24, window=(2, 7)) == (2, 7)


def test_window_empty_after_the_cap_raises():
    with pytest.raises(ValueError, match="empty after the L-2 cap"):
        resolve_effective_window(4, window=(3, 3))


# ---------------------------------------------------------------- 3. adapter / NaN guard


def _adapter_operands(seed=0):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(1, 3, HIDDEN, generator=gen)
    ffn = torch.randn(1, 3, HIDDEN, generator=gen)
    Z = torch.randn(len(IMAGE_POSITIONS), HIDDEN, generator=gen)
    up = torch.randn(INTER, HIDDEN, generator=gen)
    down = torch.randn(HIDDEN, INTER, generator=gen)
    return x, ffn, Z, up, down


def test_adapter_mix_alpha_zero_returns_the_ffn_bit_identical():
    x, ffn, Z, up, down = _adapter_operands()
    assert memvr_adapter_mix(x, ffn, Z, up, down, alpha=0.0) is ffn


def test_adapter_mix_none_Z_returns_the_ffn():
    x, ffn, _, up, down = _adapter_operands()
    assert memvr_adapter_mix(x, ffn, None, up, down, alpha=0.5) is ffn


def test_adapter_mix_zero_visual_tokens_returns_the_ffn():
    x, ffn, _, up, down = _adapter_operands()
    Z = torch.zeros(len(IMAGE_POSITIONS), HIDDEN)
    assert memvr_adapter_mix(x, ffn, Z, up, down, alpha=0.5) is ffn


def test_adapter_mix_zero_input_returns_the_ffn():
    """x=0 makes the adapter all-zero; the mean|adapter|==0 guard fires."""
    _, ffn, Z, up, down = _adapter_operands()
    x = torch.zeros(1, 3, HIDDEN)
    assert memvr_adapter_mix(x, ffn, Z, up, down, alpha=0.5) is ffn


def test_adapter_mix_moves_the_output_and_stays_finite():
    x, ffn, Z, up, down = _adapter_operands()
    out = memvr_adapter_mix(x, ffn, Z, up, down, alpha=0.5)
    assert out.shape == ffn.shape
    assert torch.isfinite(out).all()
    assert not torch.equal(out, ffn)


# ---------------------------------------------------------------- 4. trigger


def test_fires_inside_the_window_and_records_the_trigger():
    model = _build_model()
    snap, _ = _run_memvr(model, _prefill(), gamma=-1.0, alpha=0.5, window=(2, 3))
    assert snap.n_fires_total == 1
    assert snap.fire_layer == 2  # the first in-window layer
    assert snap.fired_in_prefill is True
    assert snap.armed_layer is None  # layer 3 injected and disarmed


def test_does_not_fire_outside_the_window():
    model = _build_model()
    # Window starts at 3: layers 0-2 must not fire even with gamma below all.
    snap, _ = _run_memvr(model, _prefill(), gamma=-1.0, alpha=0.0, window=(3, 3))
    assert snap.n_fires_total == 1
    assert snap.fire_layer == 3


def test_at_most_one_fire_per_forward():
    model = _build_model()
    snap, _ = _run_memvr(model, _prefill(), gamma=-1.0, alpha=0.0, window=(2, 4))
    assert snap.n_fires_total == 1
    assert snap.fire_layer == 2


def test_rearms_on_the_next_forward():
    model = _build_model()
    wrapper = _MockWrapper(model)
    hp = MemVRHyperparams(gamma=0.0, alpha=0.0, window=(2, 3))
    with enable_memvr(wrapper, hp) as buffer:
        buffer.gamma = -1.0
        buffer.update_image_positions(IMAGE_POS_T.clone())
        _sweep(model, _prefill())
        _sweep(model, _decode())
        snap = _snapshot(buffer)
    assert snap.n_fires_total == 2  # once per forward
    assert snap.fire_layer == 2
    assert snap.fired_in_prefill is True


def test_injection_localizes_to_the_armed_layer():
    model = _build_model()
    hidden = _prefill()
    _, base = _run_memvr(model, hidden, gamma=2.0, alpha=0.5, window=(2, 3))  # no fire
    _, fired = _run_memvr(model, hidden, gamma=-1.0, alpha=0.5, window=(2, 3))  # inject @3
    for i in (0, 1, 2):
        assert torch.equal(base[i], fired[i]), f"layer {i} moved but should not have"
    assert not torch.equal(base[3], fired[3]), "armed layer 3 output did not move"


def test_boundary_window_injects_at_the_last_valid_layer():
    model = _build_model()  # L=6, so L-1=5 is the last layer, L-2=4 the cap
    snap, _ = _run_memvr(model, _prefill(), gamma=-1.0, alpha=0.5, window=(4, 4))
    assert snap.n_fires_total == 1
    assert snap.fire_layer == 4  # arms layer 5, a real layer; no crash


# ---------------------------------------------------------------- 5. Z capture


def test_captures_visual_tokens_at_the_prefill():
    model = _build_model()
    hidden = _prefill(seq=6)
    snap, _ = _run_memvr(model, hidden, gamma=2.0, alpha=0.0, window=(2, 3))
    assert snap.Z is not None
    assert snap.Z.shape == (len(IMAGE_POSITIONS), HIDDEN)
    assert snap.Z.dtype is torch.float32
    assert torch.allclose(snap.Z, hidden[0].index_select(0, IMAGE_POS_T).float())


def test_visual_tokens_are_not_recaptured_on_decode():
    model = _build_model()
    wrapper = _MockWrapper(model)
    hp = MemVRHyperparams(gamma=2.0, alpha=0.0, window=(2, 3))
    with enable_memvr(wrapper, hp) as buffer:
        buffer.update_image_positions(IMAGE_POS_T.clone())
        _sweep(model, _prefill())
        z_prefill = buffer.Z.clone()
        _sweep(model, _decode())
        z_decode = buffer.Z.clone()
    assert torch.equal(z_prefill, z_decode)


def test_capture_requires_image_positions():
    model = _build_model()
    wrapper = _MockWrapper(model)
    hp = MemVRHyperparams(gamma=2.0, alpha=0.0, window=(2, 3))
    with enable_memvr(wrapper, hp):
        with pytest.raises(RuntimeError, match="update_image_positions"):
            _sweep(model, _prefill())


# ---------------------------------------------------------------- 6. neutrality


def test_gamma_one_never_fires_and_is_bit_identical():
    model = _build_model()
    hidden = _prefill()
    base = _sweep(model, hidden)  # no MemVR installed
    snap, memvr = _run_memvr(model, hidden, gamma=1.0, alpha=0.5, window=(2, 3))
    assert snap.n_fires_total == 0
    for i, (b, m) in enumerate(zip(base, memvr, strict=True)):
        assert torch.equal(b, m), f"layer {i} differs under gamma=1.0"


def test_no_hooks_persist_outside_the_context():
    model = _build_model()
    wrapper = _MockWrapper(model)
    layers = decoder_of(model).layers
    assert all(len(layer._forward_hooks) == 0 for layer in layers)
    assert len(layers[0]._forward_pre_hooks) == 0
    with enable_memvr(wrapper, MemVRHyperparams(gamma=0.75, alpha=0.12, window=(2, 3))):
        assert all(len(layer._forward_hooks) == 1 for layer in layers)
        assert len(layers[0]._forward_pre_hooks) == 1
    assert all(len(layer._forward_hooks) == 0 for layer in layers)
    assert len(layers[0]._forward_pre_hooks) == 0


# ---------------------------------------------------------------- 7. restoration


def test_context_manager_restores_on_normal_exit():
    model = _build_model()
    wrapper = _MockWrapper(model)
    layers = decoder_of(model).layers
    with enable_memvr(wrapper, MemVRHyperparams(gamma=0.75, alpha=0.12, window=(2, 3))):
        assert all(
            _mlp_module_of(layer).forward.__name__ == "_memvr_mlp_forward"
            for layer in layers
        )
    assert all(_mlp_module_of(layer).forward.__name__ == "forward" for layer in layers)
    assert all(len(layer._forward_hooks) == 0 for layer in layers)
    assert len(layers[0]._forward_pre_hooks) == 0


def test_context_manager_restores_on_exception():
    model = _build_model()
    wrapper = _MockWrapper(model)
    layers = decoder_of(model).layers
    with pytest.raises(RuntimeError, match="boom"):
        with enable_memvr(wrapper, MemVRHyperparams(gamma=0.75, alpha=0.12, window=(2, 3))):
            raise RuntimeError("boom")
    assert all(_mlp_module_of(layer).forward.__name__ == "forward" for layer in layers)
    assert all(len(layer._forward_hooks) == 0 for layer in layers)
    assert len(layers[0]._forward_pre_hooks) == 0


# ---------------------------------------------------------------- 8. coexistence with SPARC


def test_memvr_coexists_with_enable_sparc():
    from PIL import Image

    from vr_modality_bias.experiment.sparc import SparcHyperparams, enable_sparc

    model = _build_model(block_cls=_FullBlock)
    ids = [0, IMG_ID, IMG_ID, 0, IMG_ID, 0]  # image tokens at 1, 2, 4
    wrapper = _CoexWrapper(model, ids)
    layers = decoder_of(model).layers

    sparc_hp = SparcHyperparams(
        alpha=1.3, tau=1e9, selected_layer=3, se_layers=(0, N_LAYERS - 1)
    )
    memvr_hp = MemVRHyperparams(gamma=0.75, alpha=0.5, window=(2, 3))

    with enable_sparc(
        wrapper, hparams=sparc_hp, probe_image=Image.new("RGB", (8, 8)), prompt="q"
    ) as sparc_buf:
        sparc_buf.reset()
        sparc_buf.update_input_len(len(ids) - len(IMAGE_POSITIONS))
        sparc_buf.update_image_positions(IMAGE_POS_T.clone())
        assert all(_sparc_active(layer) for layer in layers)

        with enable_memvr(wrapper, memvr_hp) as memvr_buf:
            memvr_buf.gamma = -1.0  # force a MemVR fire during the shared forward
            memvr_buf.update_image_positions(IMAGE_POS_T.clone())
            assert all(
                _mlp_module_of(layer).forward.__name__ == "_memvr_mlp_forward"
                for layer in layers
            )
            assert all(len(layer._forward_hooks) == 1 for layer in layers)

            cache = DynamicCache()
            h = _prefill(seq=len(ids))
            with torch.no_grad():
                for layer in layers:
                    h = layer(
                        hidden_states=h,
                        past_key_values=cache,
                        position_embeddings=_pos(h.shape[1]),
                    )[0]
            assert memvr_buf.n_fires_total >= 1  # MemVR fired while SPARC was active

        # MemVR removed; SPARC still installed.
        assert all(_mlp_module_of(layer).forward.__name__ == "forward" for layer in layers)
        assert all(len(layer._forward_hooks) == 0 for layer in layers)
        assert all(_sparc_active(layer) for layer in layers)

    # Both removed.
    assert not any(_sparc_active(layer) for layer in layers)


class _CoexProcessor:
    def __init__(self, ids):
        self._ids = ids

    def apply_chat_template(self, *_args, **_kwargs):
        return "<prefix>"

    def __call__(self, text=None, images=None, return_tensors="pt"):
        return {"input_ids": torch.tensor([self._ids])}


class _CoexWrapper(_MockWrapper):
    """``_MockWrapper`` plus the processor bits ``enable_sparc``'s probe needs."""

    def __init__(self, model, ids):
        super().__init__(model)
        self._processor = _CoexProcessor(ids)

    @staticmethod
    def _build_messages(prompt, image=None):
        return [{"role": "user", "content": prompt}]


# ---------------------------------------------------------------- 9. get_final_norm discovery

from vr_modality_bias.models.internvl import InternVLWrapper  # noqa: E402
from vr_modality_bias.models.llava import LlavaWrapper  # noqa: E402
from vr_modality_bias.models.qwen_vl import QwenVLWrapper  # noqa: E402
from vr_modality_bias.models.smolvlm import SmolVLMWrapper  # noqa: E402


def test_smolvlm_discovers_the_final_norm_interleaved_path():
    wrapper = SmolVLMWrapper()
    norm = nn.Identity()
    wrapper._model = SimpleNamespace(
        model=SimpleNamespace(text_model=SimpleNamespace(norm=norm))
    )
    assert wrapper._discover_final_norm() is norm


def test_qwen_discovers_the_final_norm_language_model_path():
    wrapper = QwenVLWrapper()
    norm = nn.Identity()
    wrapper._model = SimpleNamespace(
        model=SimpleNamespace(language_model=SimpleNamespace(norm=norm))
    )
    assert wrapper._discover_final_norm() is norm


@pytest.mark.parametrize(
    "wrapper_cls", [SmolVLMWrapper, QwenVLWrapper, LlavaWrapper, InternVLWrapper]
)
def test_get_final_norm_before_load_raises(wrapper_cls):
    with pytest.raises(RuntimeError, match="Model not loaded"):
        wrapper_cls().get_final_norm()


# ---------------------------------------------------------------- 10. hyperparams


def test_hyperparams_defaults_match_the_official_recipe():
    hp = MemVRHyperparams()
    assert hp.gamma == 0.75
    assert hp.alpha == 0.12
    assert hp.window is None
    assert hp.top_k == 10


def test_hyperparams_reject_alpha_outside_the_unit_interval():
    with pytest.raises(ValueError, match="alpha"):
        MemVRHyperparams(alpha=1.5)


def test_hyperparams_reject_a_negative_gamma():
    with pytest.raises(ValueError, match="gamma"):
        MemVRHyperparams(gamma=-0.1)


def test_hyperparams_reject_a_zero_top_k():
    with pytest.raises(ValueError, match="top_k"):
        MemVRHyperparams(top_k=0)
