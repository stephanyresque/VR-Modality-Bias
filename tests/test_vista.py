"""Tests for the VISTA core (``utils/vista.py`` + ``experiment/vista.py``), v1.

Coverage:

1. Steering formula: the adaptive anti-visual weight (doubles when anti-visual,
   1.0 when aligned/orthogonal), a constructed steer value, and norm preservation.
2. Short-circuit: lam=0 and vsv=None return the MLP output bit-for-bit.
3. Window: out-of-window layers untouched, None = all, resolution + validation.
4. compute_vsv: shape [L, d], float32, last position of unequal-length sequences,
   embedding excluded.
5. build_negative_inputs: strips the image item, asserts no image token survives.
6. sla_mix_logits: alpha=0 -> final logits, alpha=1 -> window mean, blend by alpha.
7. Restoration by the context manager on normal exit and on exception.
8. Coexistence with enable_sparc; coexistence with MemVR installed-off; mutual
   exclusion when both MemVR and VISTA are armed.
9. Instrumentation populated after a steered forward.
"""

from __future__ import annotations

import functools
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from vr_modality_bias.experiment.vista import (
    VistaHyperparams,
    enable_vista,
    resolve_sla_window,
    resolve_vsv_window,
)
from vr_modality_bias.utils.attn import decoder_of
from vr_modality_bias.utils.vista import (
    VistaBuffer,
    _extract_memvr_buffer,
    _mlp_module_of,
    _strip_image_items,
    build_negative_inputs,
    compute_vsv,
    sla_mix_logits,
    vista_adaptive_factor,
    vista_steer,
)

# ---------------------------------------------------------------- fixtures

HIDDEN = 8
INTER = 16
VOCAB = 20
N_LAYERS = 6
IMG_ID = 99

# For the SPARC coexistence mock: N_HEADS * HEAD_DIM == HIDDEN.
N_HEADS = 2
HEAD_DIM = 4


class _TinyMLP(nn.Module):
    def __init__(self, hidden: int = HIDDEN, inter: int = INTER):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _MLPBlock(nn.Module):
    def __init__(self, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.mlp = _TinyMLP()

    def forward(self, hidden_states, **kwargs):
        return hidden_states + self.mlp(hidden_states)


class _TinyAttn(nn.Module):
    """SPARC-compatible attention (matches test_memvr's mock)."""

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
        self.norm = nn.LayerNorm(HIDDEN)


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


def _prefill(seq: int = 5, seed: int = 3) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(1, seq, HIDDEN, generator=gen)


def _vsv(seed: int = 1) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(N_LAYERS, HIDDEN, generator=gen)


def _pos(q_len: int):
    return torch.ones(1, q_len, HEAD_DIM), torch.zeros(1, q_len, HEAD_DIM)


@torch.no_grad()
def _sweep(model: _Model, hidden: torch.Tensor):
    outs = []
    h = hidden
    for layer in decoder_of(model).layers:
        out = layer(h)
        h = out[0] if isinstance(out, (tuple, list)) else out
        outs.append(h)
    return outs


def _sparc_active(layer) -> bool:
    fwd = layer.self_attn.forward
    return isinstance(getattr(fwd, "__func__", None), functools.partial)


# ---------------------------------------------------------------- 1. steering formula


def test_adaptive_factor_doubles_when_the_state_is_anti_visual():
    v = torch.tensor([1.0, 0.0, 0.0, 0.0])
    x = torch.tensor([[-2.0, 0.0, 0.0, 0.0]])  # x points against v
    assert float(vista_adaptive_factor(x, v).reshape(-1)[0]) == pytest.approx(2.0)


def test_adaptive_factor_is_one_when_the_state_is_aligned():
    v = torch.tensor([1.0, 0.0, 0.0, 0.0])
    x = torch.tensor([[2.0, 0.0, 0.0, 0.0]])  # x aligned with v
    assert float(vista_adaptive_factor(x, v).reshape(-1)[0]) == pytest.approx(1.0)


def test_adaptive_factor_is_one_when_orthogonal():
    v = torch.tensor([1.0, 0.0, 0.0, 0.0])
    x = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    assert float(vista_adaptive_factor(x, v).reshape(-1)[0]) == pytest.approx(1.0)


def test_steer_constructed_orthogonal_case():
    """x orthogonal to v: adaptive factor 1, so y = lam * v_hat, then renormalize."""
    v = torch.tensor([1.0, 0.0, 0.0, 0.0])
    x = torch.tensor([[0.0, 3.0, 0.0, 0.0]])
    out = vista_steer(x, v, lam=0.5)
    denom = (0.5**2 + 1.0) ** 0.5
    expected = torch.tensor([[0.5 / denom, 1.0 / denom, 0.0, 0.0]]) * 3.0
    assert torch.allclose(out, expected, atol=1e-5)


def test_steer_preserves_the_original_norm():
    torch.manual_seed(0)
    x = torch.randn(1, 4, HIDDEN)
    v = torch.randn(HIDDEN)
    out = vista_steer(x, v, lam=0.5)
    assert torch.allclose(out.norm(dim=-1), x.norm(dim=-1), atol=1e-5)


# ---------------------------------------------------------------- 2. short-circuit


def test_steer_lam_zero_is_bit_identical():
    x = torch.randn(1, 3, HIDDEN)
    v = torch.randn(HIDDEN)
    assert vista_steer(x, v, lam=0.0) is x


def test_steer_none_vsv_is_bit_identical():
    x = torch.randn(1, 3, HIDDEN)
    assert vista_steer(x, None, lam=0.5) is x


def test_no_vsv_leaves_the_sweep_bit_identical():
    model = _build_model()
    hidden = _prefill()
    base = _sweep(model, hidden)
    with enable_vista(_MockWrapper(model), VistaHyperparams(lam=0.5)):
        steered = _sweep(model, hidden)  # never armed (no set_vsv)
    for b, s in zip(base, steered, strict=True):
        assert torch.equal(b, s)


def test_lam_zero_leaves_the_sweep_bit_identical():
    model = _build_model()
    hidden = _prefill()
    base = _sweep(model, hidden)
    with enable_vista(_MockWrapper(model), VistaHyperparams(lam=0.0)) as buf:
        buf.set_vsv(_vsv())
        steered = _sweep(model, hidden)
    for b, s in zip(base, steered, strict=True):
        assert torch.equal(b, s)


# ---------------------------------------------------------------- 3. window


def test_out_of_window_layers_are_untouched():
    model = _build_model()
    hidden = _prefill()
    base = _sweep(model, hidden)
    with enable_vista(_MockWrapper(model), VistaHyperparams(lam=0.5, window=(2, 3))) as buf:
        buf.set_vsv(_vsv())
        steered = _sweep(model, hidden)
    # Layers before the window are byte-identical; the window's first layer moves.
    assert torch.equal(base[0], steered[0])
    assert torch.equal(base[1], steered[1])
    assert not torch.equal(base[2], steered[2])


def test_window_none_steers_every_layer():
    model = _build_model()
    hidden = _prefill()
    base = _sweep(model, hidden)
    with enable_vista(_MockWrapper(model), VistaHyperparams(lam=0.5)) as buf:
        buf.set_vsv(_vsv(seed=2))
        steered = _sweep(model, hidden)
    assert not torch.equal(base[0], steered[0])  # layer 0 steered too


def test_resolve_vsv_window_none_means_all_layers():
    assert resolve_vsv_window(24) is None


def test_resolve_vsv_window_override_is_clamped():
    assert resolve_vsv_window(24, window=(0, 100)) == (0, 23)


def test_resolve_vsv_window_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        resolve_vsv_window(24, window=(20, 5))


def test_resolve_sla_window_is_depth_proportional():
    assert resolve_sla_window(24) == (19, 22)
    assert resolve_sla_window(32) == (25, 30)


def test_resolve_sla_window_override_is_clamped():
    assert resolve_sla_window(24, window=(10, 100)) == (10, 23)


# ---------------------------------------------------------------- 4. compute_vsv


class _VsvModel(nn.Module):
    """Per-layer hidden = emb * (layer + 2); distinct per layer, non-cancelling."""

    def __init__(self, n_layers=N_LAYERS, d=HIDDEN, vocab=VOCAB):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self._n_layers = n_layers
        self.lm_head = nn.Linear(d, vocab, bias=False)

    def forward(self, input_ids=None, output_hidden_states=False, use_cache=False, **kw):
        h0 = self.emb(input_ids)
        hs = [h0]
        for i in range(self._n_layers):
            hs.append(h0 * float(i + 2))
        return SimpleNamespace(hidden_states=tuple(hs), logits=self.lm_head(hs[-1]))


def test_compute_vsv_shape_dtype_last_position_and_excludes_embedding():
    torch.manual_seed(0)
    model = _VsvModel()
    model.eval()
    wrapper = SimpleNamespace(_model=model)
    pos_inputs = {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}  # length 5
    neg_inputs = {"input_ids": torch.tensor([[6, 7, 8]])}  # length 3
    vsv = compute_vsv(wrapper, pos_inputs, neg_inputs)

    assert vsv.shape == (N_LAYERS, HIDDEN)  # L rows -> embedding excluded
    assert vsv.dtype is torch.float32
    diff = model.emb.weight[5] - model.emb.weight[8]  # last tokens of each sequence
    for layer in range(N_LAYERS):
        assert torch.allclose(vsv[layer], diff * float(layer + 2), atol=1e-5)


# ---------------------------------------------------------------- 5. negatives


class _NegProcessor:
    def __init__(self, ids, record=None):
        self._ids = ids
        self._record = record

    def apply_chat_template(self, messages, **kwargs):
        if self._record is not None:
            self._record.append(messages)
        return "prefix"

    def __call__(self, text=None, images=None, return_tensors="pt"):
        return {"input_ids": torch.tensor([self._ids])}


class _NegWrapper:
    def __init__(self, ids, record=None, image_token_id=IMG_ID):
        self._processor = _NegProcessor(ids, record)
        self._model = SimpleNamespace(config=SimpleNamespace(image_token_id=image_token_id))

    @staticmethod
    def _build_messages(prompt, image=None):
        return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]


def test_strip_image_items_removes_the_image_dict():
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "q"}]}]
    assert _strip_image_items(messages) == [
        {"role": "user", "content": [{"type": "text", "text": "q"}]}
    ]


def test_build_negative_inputs_strips_the_image_and_passes_the_assert():
    record: list = []
    inputs = build_negative_inputs(_NegWrapper(ids=[1, 2, 3], record=record), "is there a dog?")
    assert "input_ids" in inputs
    content = record[0][0]["content"]
    assert all(item.get("type") != "image" for item in content)


def test_build_negative_inputs_raises_when_an_image_token_survives():
    with pytest.raises(AssertionError, match="image tokens"):
        build_negative_inputs(_NegWrapper(ids=[1, IMG_ID, 3]), "q")


# ---------------------------------------------------------------- 6. SLA


def _sla_operands(seed=0):
    torch.manual_seed(seed)
    hidden_states = [torch.randn(HIDDEN) for _ in range(N_LAYERS)]
    lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)
    logits = torch.randn(VOCAB)
    return hidden_states, lm_head, logits


def test_sla_alpha_zero_returns_the_final_logits():
    hs, lm, logits = _sla_operands()
    assert sla_mix_logits(hs, lm, logits, alpha=0.0, window=(1, 3)) is logits


def test_sla_alpha_one_returns_the_window_mean():
    hs, lm, logits = _sla_operands()
    out = sla_mix_logits(hs, lm, logits, alpha=1.0, window=(1, 3))
    expected = torch.stack([lm(hs[layer]).float() for layer in (1, 2, 3)], dim=0).mean(dim=0)
    assert torch.allclose(out, expected, atol=1e-5)


def test_sla_blends_by_alpha():
    hs, lm, logits = _sla_operands(seed=1)
    alpha = 0.3
    out = sla_mix_logits(hs, lm, logits, alpha=alpha, window=(1, 3))
    mean = torch.stack([lm(hs[layer]).float() for layer in (1, 2, 3)], dim=0).mean(dim=0)
    expected = alpha * mean + (1.0 - alpha) * logits
    assert torch.allclose(out, expected, atol=1e-5)


# ---------------------------------------------------------------- 7. restoration


def test_context_manager_restores_on_normal_exit():
    model = _build_model()
    layers = decoder_of(model).layers
    with enable_vista(_MockWrapper(model), VistaHyperparams(lam=0.5)) as buf:
        assert all(_mlp_module_of(layer).forward.__name__ == "_vista_mlp_forward" for layer in layers)
        assert len(layers[0]._forward_pre_hooks) == 1
        assert buf.installed is True
    assert all(_mlp_module_of(layer).forward.__name__ == "forward" for layer in layers)
    assert len(layers[0]._forward_pre_hooks) == 0
    assert buf.installed is False


def test_context_manager_restores_on_exception():
    model = _build_model()
    layers = decoder_of(model).layers
    with pytest.raises(RuntimeError, match="boom"):
        with enable_vista(_MockWrapper(model), VistaHyperparams(lam=0.5)):
            raise RuntimeError("boom")
    assert all(_mlp_module_of(layer).forward.__name__ == "forward" for layer in layers)
    assert len(layers[0]._forward_pre_hooks) == 0


# ---------------------------------------------------------------- 8. coexistence


def test_vista_coexists_with_enable_sparc():
    from PIL import Image

    from vr_modality_bias.experiment.sparc import SparcHyperparams, enable_sparc

    model = _build_model(block_cls=_FullBlock)
    ids = [0, IMG_ID, IMG_ID, 0, IMG_ID, 0]
    wrapper = _CoexWrapper(model, ids)
    layers = decoder_of(model).layers

    sparc_hp = SparcHyperparams(
        alpha=1.3, tau=1e9, selected_layer=3, se_layers=(0, N_LAYERS - 1)
    )
    with enable_sparc(
        wrapper, hparams=sparc_hp, probe_image=Image.new("RGB", (8, 8)), prompt="q"
    ) as sparc_buf:
        sparc_buf.reset()
        sparc_buf.update_input_len(len(ids) - 3)
        sparc_buf.update_image_positions(torch.tensor([1, 2, 4]))
        assert all(_sparc_active(layer) for layer in layers)

        with enable_vista(wrapper, VistaHyperparams(lam=0.5)) as vsv_buf:
            vsv_buf.set_vsv(_vsv(seed=4))
            assert all(
                _mlp_module_of(layer).forward.__name__ == "_vista_mlp_forward"
                for layer in layers
            )
            cache = DynamicCache()
            h = _prefill(seq=len(ids))
            with torch.no_grad():
                for layer in layers:
                    h = layer(
                        hidden_states=h,
                        past_key_values=cache,
                        position_embeddings=_pos(h.shape[1]),
                    )[0]
            assert vsv_buf.n_steered_forwards >= 1

        assert all(_mlp_module_of(layer).forward.__name__ == "forward" for layer in layers)
        assert all(_sparc_active(layer) for layer in layers)

    assert not any(_sparc_active(layer) for layer in layers)


def test_vista_coexists_with_memvr_installed_off():
    from vr_modality_bias.experiment.memvr import MemVRHyperparams, enable_memvr

    model = _build_model()
    wrapper = _MockWrapper(model)
    layers = decoder_of(model).layers
    hidden = _prefill(seq=5)

    with enable_memvr(wrapper, MemVRHyperparams(gamma=1.0, alpha=0.12, window=(2, 3))) as mv_buf:
        mv_buf.update_image_positions(torch.tensor([1, 2, 4]))  # for MemVR's Z capture
        with enable_vista(wrapper, VistaHyperparams(lam=0.5)) as vsv_buf:
            vsv_buf.set_vsv(_vsv(seed=5))
            _sweep(model, hidden)  # gamma=1.0 -> MemVR never arms -> no collision
            assert mv_buf.n_fires_total == 0
            assert vsv_buf.n_steered_forwards >= 1

    assert all(_mlp_module_of(layer).forward.__name__ == "forward" for layer in layers)
    assert len(layers[0]._forward_hooks) == 0
    assert len(layers[0]._forward_pre_hooks) == 0


def test_memvr_and_vista_both_armed_raises():
    from vr_modality_bias.experiment.memvr import MemVRHyperparams, enable_memvr

    model = _build_model()
    wrapper = _MockWrapper(model)
    hidden = _prefill(seq=5)

    with enable_memvr(wrapper, MemVRHyperparams(gamma=0.0, alpha=0.12, window=(1, 2))) as mv_buf:
        mv_buf.gamma = -1.0  # force MemVR to fire and arm
        mv_buf.update_image_positions(torch.tensor([1, 2, 4]))
        with enable_vista(wrapper, VistaHyperparams(lam=0.5)) as vsv_buf:
            vsv_buf.set_vsv(_vsv(seed=6))
            with pytest.raises(RuntimeError, match="both armed"):
                _sweep(model, hidden)


class _CoexProcessor:
    def __init__(self, ids):
        self._ids = ids

    def apply_chat_template(self, *_args, **_kwargs):
        return "<prefix>"

    def __call__(self, text=None, images=None, return_tensors="pt"):
        return {"input_ids": torch.tensor([self._ids])}


class _CoexWrapper(_MockWrapper):
    def __init__(self, model, ids):
        super().__init__(model)
        self._processor = _CoexProcessor(ids)

    @staticmethod
    def _build_messages(prompt, image=None):
        return [{"role": "user", "content": prompt}]


def test_extract_memvr_buffer_finds_the_buffer_and_none_otherwise():
    from vr_modality_bias.experiment.memvr import MemVRHyperparams, enable_memvr

    model = _build_model()
    wrapper = _MockWrapper(model)
    layers = decoder_of(model).layers
    # A plain mlp forward is not a MemVR wrapper.
    assert _extract_memvr_buffer(_mlp_module_of(layers[0]).forward) is None
    with enable_memvr(wrapper, MemVRHyperparams(gamma=1.0, alpha=0.12, window=(2, 3))) as mv_buf:
        extracted = _extract_memvr_buffer(_mlp_module_of(layers[0]).forward)
        assert extracted is mv_buf


# ---------------------------------------------------------------- 9. instrumentation


def test_instrumentation_is_populated_after_a_steered_forward():
    model = _build_model()
    hidden = _prefill(seq=5)
    with enable_vista(_MockWrapper(model), VistaHyperparams(lam=0.5)) as buf:
        vsv = _vsv(seed=3)
        buf.set_vsv(vsv)
        _sweep(model, hidden)
        assert buf.n_steered_forwards == 1
        assert buf.lambda_sim_count == N_LAYERS  # window None -> every layer, prefill
        assert buf.lambda_sim_mean() >= 1.0
        assert buf.vsv_norm_mean == pytest.approx(float(vsv.norm(dim=-1).mean()))


# ---------------------------------------------------------------- hyperparams


def test_hyperparams_defaults_match_the_official_recipe():
    hp = VistaHyperparams()
    assert hp.lam == 0.01
    assert hp.window is None
    assert hp.sla is False
    assert hp.sla_alpha == 0.3
    assert hp.sla_window is None


def test_hyperparams_reject_a_negative_lam():
    with pytest.raises(ValueError, match="lam"):
        VistaHyperparams(lam=-0.1)


def test_hyperparams_reject_sla_alpha_outside_the_unit_interval():
    with pytest.raises(ValueError, match="sla_alpha"):
        VistaHyperparams(sla_alpha=1.5)
