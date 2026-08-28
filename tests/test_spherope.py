from __future__ import annotations

import math

import pytest
import torch
from PIL import Image

from vr_modality_bias.experiment.spherope import SPHEROPE_MODES, enable_spherope, vision_tower_of
from vr_modality_bias.utils.spherope import (
    SpheRoPETextRotary,
    SpheRoPEVisionRotary,
    circular_pad_image,
    pad_columns,
    sfc_width_half_angles,
    width_freq_slots,
)


def _inv_freq(dim: int, theta: float = 10000.0) -> torch.Tensor:
    return 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))


def _grid(n_rows: int, n_cols: int):
    rows, cols = torch.meshgrid(torch.arange(n_rows), torch.arange(n_cols), indexing="ij")
    return rows.flatten(), cols.flatten()


def _wrapped(angles: torch.Tensor) -> torch.Tensor:
    return torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)


# ---------------------------------------------------------------- the angles


def test_the_width_axis_wraps_around_in_every_frequency():
    n_rows, n_cols = 9, 32
    rows, cols = _grid(n_rows, n_cols)
    inv_freq = _inv_freq(40)

    inside = sfc_width_half_angles(rows, cols, n_rows, n_cols, inv_freq)
    beyond = sfc_width_half_angles(rows, cols + n_cols, n_rows, n_cols, inv_freq)

    assert torch.allclose(_wrapped(inside), _wrapped(beyond), atol=1e-5)


def test_a_linear_width_axis_does_not_wrap():
    n_rows, n_cols = 9, 32
    rows, cols = _grid(n_rows, n_cols)
    inv_freq = _inv_freq(40)

    inside = cols.float().unsqueeze(1) * inv_freq.unsqueeze(0)
    beyond = (cols + n_cols).float().unsqueeze(1) * inv_freq.unsqueeze(0)

    assert not torch.allclose(_wrapped(inside), _wrapped(beyond), atol=1e-3)


def test_every_column_of_a_pole_row_shares_the_spherical_embedding():
    n_rows, n_cols = 9, 32
    rows, cols = _grid(n_rows, n_cols)
    inv_freq = _inv_freq(40)

    angles = sfc_width_half_angles(rows, cols, n_rows, n_cols, inv_freq, max_error=0.0)
    north = angles[rows == 0]
    equator = angles[rows == n_rows // 2]

    assert torch.allclose(north, north[:1].expand_as(north), atol=1e-5)
    assert not torch.allclose(equator, equator[:1].expand_as(equator), atol=1e-3)


def test_high_frequencies_take_the_harmonic_path_and_low_ones_the_sphere():
    n_rows, n_cols = 9, 32
    rows, cols = _grid(n_rows, n_cols)
    inv_freq = _inv_freq(40)
    fundamental = 2 * math.pi / n_cols
    valid = (inv_freq / fundamental >= 1.0) & (
        torch.abs(inv_freq / fundamental - torch.round(inv_freq / fundamental))
        / (inv_freq / fundamental) <= 0.06
    )
    split = int(torch.where(~valid)[0][0])
    assert 0 < split < inv_freq.numel()

    angles = sfc_width_half_angles(rows, cols, n_rows, n_cols, inv_freq)
    harmonic = torch.round(inv_freq[:split] / fundamental) * fundamental
    expected = cols.float().unsqueeze(1) * harmonic.unsqueeze(0)

    assert torch.allclose(angles[:, :split], expected, atol=1e-4)
    north = angles[rows == 0][:, split:]
    assert torch.allclose(north, north[:1].expand_as(north), atol=1e-5)


def test_padded_columns_copy_the_embedding_of_the_column_they_wrap():
    n_rows, n_cols, pad = 9, 40, 4
    rows, cols = _grid(n_rows, n_cols)
    inv_freq = _inv_freq(40)

    angles = sfc_width_half_angles(rows, cols, n_rows, n_cols, inv_freq, pad_cols=pad)
    width = n_cols - 2 * pad
    row = rows == 3
    left_pad = angles[row][:pad]
    right_source = angles[row][width:width + pad]
    right_pad = angles[row][width + pad:]
    left_source = angles[row][pad:2 * pad]

    assert torch.allclose(_wrapped(left_pad), _wrapped(right_source), atol=1e-5)
    assert torch.allclose(_wrapped(right_pad), _wrapped(left_source), atol=1e-5)


def test_the_angles_are_finite_and_float32():
    rows, cols = _grid(4, 6)
    angles = sfc_width_half_angles(rows, cols, 4, 6, _inv_freq(20))

    assert angles.dtype == torch.float32
    assert torch.isfinite(angles).all()


# ---------------------------------------------------------------- wrappers


class _VisionRotary(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.inv_freq = torch.nn.Buffer(_inv_freq(dim), persistent=False)

    def forward(self, position_ids):
        return (position_ids.unsqueeze(-1) * self.inv_freq).flatten(1)


def test_the_vision_wrapper_keeps_the_height_half_and_replaces_the_width_half():
    original = _VisionRotary(40)
    wrapper = SpheRoPEVisionRotary(original)
    rows, cols = _grid(6, 12)
    position_ids = torch.stack([rows, cols], dim=1)

    base = original(position_ids)
    out = wrapper(position_ids)

    n_freq = original.inv_freq.numel()
    assert out.shape == base.shape
    assert torch.equal(out[:, :n_freq], base[:, :n_freq])
    assert not torch.allclose(out[:, n_freq:], base[:, n_freq:])
    assert wrapper.calls == 1


class _TextRotary(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.inv_freq = torch.nn.Buffer(_inv_freq(dim), persistent=False)
        self.attention_scaling = 1.0

    def forward(self, x, position_ids):
        inv = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        freqs = (inv @ position_ids[:, :, None, :].float()).transpose(2, 3)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


def _mrope_position_ids(n_text_before: int, n_rows: int, n_cols: int, n_text_after: int):
    t, h, w = [], [], []
    for i in range(n_text_before):
        t.append(i); h.append(i); w.append(i)
    start = n_text_before
    for r in range(n_rows):
        for c in range(n_cols):
            t.append(start); h.append(start + r); w.append(start + c)
    nxt = start + max(n_rows, n_cols)
    for i in range(n_text_after):
        t.append(nxt + i); h.append(nxt + i); w.append(nxt + i)
    ids = torch.tensor([t, h, w]).unsqueeze(1)
    image_positions = torch.arange(n_text_before, n_text_before + n_rows * n_cols)
    return ids, image_positions


def test_width_freq_slots_follow_the_mrope_section():
    assert width_freq_slots([16, 24, 24]) == (40, 64)
    assert width_freq_slots([2, 3, 3]) == (5, 8)


def test_the_text_wrapper_touches_only_the_width_slots_of_the_image_tokens():
    original = _TextRotary(16)
    wrapper = SpheRoPETextRotary(original, [2, 3, 3])
    ids, img = _mrope_position_ids(3, 4, 6, 2)
    wrapper.arm(img)
    x = torch.zeros(1, ids.shape[-1], 8)

    cos0, sin0 = original(x, ids)
    cos1, sin1 = wrapper(x, ids)

    assert cos1.shape == cos0.shape
    text = torch.ones(ids.shape[-1], dtype=torch.bool)
    text[img] = False
    assert torch.equal(cos1[:, :, text], cos0[:, :, text])
    assert torch.equal(cos1[0], cos0[0]) and torch.equal(cos1[1], cos0[1])
    lo, hi = 5, 8
    half = 8
    untouched = [i for i in range(16) if not (lo <= i < hi or half + lo <= i < half + hi)]
    assert torch.equal(cos1[2, 0, img][:, untouched], cos0[2, 0, img][:, untouched])
    assert not torch.allclose(cos1[2, 0, img][:, lo:hi], cos0[2, 0, img][:, lo:hi])
    assert torch.equal(cos1[2, 0, img][:, lo:hi], cos1[2, 0, img][:, half + lo:half + hi])
    assert torch.equal(sin1[2, 0, img][:, lo:hi], sin1[2, 0, img][:, half + lo:half + hi])
    assert wrapper.calls == 1


def test_the_text_wrapper_leaves_decode_steps_and_disarmed_calls_alone():
    original = _TextRotary(16)
    wrapper = SpheRoPETextRotary(original, [2, 3, 3])
    ids, img = _mrope_position_ids(3, 4, 6, 2)
    x = torch.zeros(1, ids.shape[-1], 8)

    cos_off, _ = wrapper(x, ids)
    assert torch.equal(cos_off, original(x, ids)[0])

    wrapper.arm(img)
    step = ids[:, :, -1:] + 1
    cos_step, _ = wrapper(x[:, :1], step)
    assert torch.equal(cos_step, original(x[:, :1], step)[0])
    assert wrapper.calls == 0

    wrapper.disarm()
    assert torch.equal(wrapper(x, ids)[0], original(x, ids)[0])


# ---------------------------------------------------------------- context manager


@pytest.fixture(scope="module")
def tiny_qwen():
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

    cfg = Qwen2_5_VLConfig(
        vision_config=dict(
            depth=2, hidden_size=32, intermediate_size=64, num_heads=2,
            out_hidden_size=64, patch_size=14, spatial_merge_size=2, window_size=28,
            fullatt_block_indexes=[1], hidden_act="silu", spatial_patch_size=14,
            temporal_patch_size=2, in_channels=3,
        ),
        text_config=dict(
            hidden_size=64, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, vocab_size=1000,
            max_position_embeddings=4096, rope_theta=1000000.0,
            rope_scaling={"type": "mrope", "mrope_section": [2, 3, 3]},
        ),
    )
    torch.manual_seed(0)
    return Qwen2_5_VLForConditionalGeneration(cfg).eval()


class _Wrapper:
    def __init__(self, model):
        self._model = model


def test_off_installs_nothing(tiny_qwen):
    vision = vision_tower_of(tiny_qwen)
    before = vision.rotary_pos_emb
    with enable_spherope(_Wrapper(tiny_qwen), mode="off", image_positions=torch.arange(4)) as h:
        assert vision.rotary_pos_emb is before
        assert h["vision"] is None and h["text"] is None


@pytest.mark.parametrize("mode", ["vit", "llm", "both"])
def test_the_wrappers_are_installed_and_restored(tiny_qwen, mode):
    vision = vision_tower_of(tiny_qwen)
    decoder = tiny_qwen.model.language_model
    before_v, before_t = vision.rotary_pos_emb, decoder.rotary_emb

    with enable_spherope(_Wrapper(tiny_qwen), mode=mode, image_positions=torch.arange(4)) as h:
        assert (vision.rotary_pos_emb is not before_v) == (mode in ("vit", "both"))
        assert (decoder.rotary_emb is not before_t) == (mode in ("llm", "both"))
        assert (h["vision"] is not None) == (mode in ("vit", "both"))
        assert (h["text"] is not None) == (mode in ("llm", "both"))

    assert vision.rotary_pos_emb is before_v
    assert decoder.rotary_emb is before_t


def test_restoration_survives_an_exception(tiny_qwen):
    vision = vision_tower_of(tiny_qwen)
    before = vision.rotary_pos_emb
    with pytest.raises(RuntimeError):
        with enable_spherope(_Wrapper(tiny_qwen), mode="both", image_positions=torch.arange(4)):
            raise RuntimeError("boom")
    assert vision.rotary_pos_emb is before


def test_an_unknown_mode_is_rejected(tiny_qwen):
    with pytest.raises(ValueError, match="mode"):
        with enable_spherope(_Wrapper(tiny_qwen), mode="sphere", image_positions=torch.arange(4)):
            pass


def test_a_family_without_2d_rope_is_rejected():
    class _Visual(torch.nn.Module):
        pass

    class _Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2)])

    class _Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_model = _Visual()
            self.text_model = _Decoder()

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _Inner()

    with pytest.raises(ValueError, match="no 2D rotary"):
        with enable_spherope(_Wrapper(_Model()), mode="vit", image_positions=torch.arange(1)):
            pass


def test_the_modes_are_the_four_documented_ones():
    assert SPHEROPE_MODES == ("off", "vit", "llm", "both")


def test_the_tiny_vision_tower_runs_with_the_spherical_rope(tiny_qwen):
    vision = vision_tower_of(tiny_qwen)
    grid = torch.tensor([[1, 4, 8]])
    n_patches = int(grid.prod())
    torch.manual_seed(1)
    pixels = torch.randn(n_patches, 3 * 2 * 14 * 14)

    with torch.no_grad():
        plain = vision(pixels, grid_thw=grid)
        with enable_spherope(_Wrapper(tiny_qwen), mode="vit", image_positions=torch.arange(n_patches // 4), pad_cols_vit=2) as h:
            sphe = vision(pixels, grid_thw=grid)
        again = vision(pixels, grid_thw=grid)

    plain_t = plain if isinstance(plain, torch.Tensor) else plain.last_hidden_state
    sphe_t = sphe if isinstance(sphe, torch.Tensor) else sphe.last_hidden_state
    again_t = again if isinstance(again, torch.Tensor) else again.last_hidden_state
    assert plain_t.shape == sphe_t.shape
    assert not torch.allclose(plain_t, sphe_t)
    assert torch.equal(plain_t, again_t)
    assert h["vision"].calls == 1


def test_the_tiny_text_model_runs_with_the_spherical_rope(tiny_qwen):
    decoder = tiny_qwen.model.language_model
    ids, img = _mrope_position_ids(3, 2, 4, 2)
    torch.manual_seed(2)
    embeds = torch.randn(1, ids.shape[-1], 64)

    with torch.no_grad():
        plain = decoder(inputs_embeds=embeds, position_ids=ids, use_cache=False).last_hidden_state
        with enable_spherope(_Wrapper(tiny_qwen), mode="llm", image_positions=img, pad_cols_llm=1) as h:
            sphe = decoder(inputs_embeds=embeds, position_ids=ids, use_cache=False).last_hidden_state
        again = decoder(inputs_embeds=embeds, position_ids=ids, use_cache=False).last_hidden_state

    assert not torch.allclose(plain, sphe)
    assert torch.equal(plain, again)
    assert torch.allclose(plain[:, :3], sphe[:, :3])
    assert h["text"].calls == 1


# ---------------------------------------------------------------- circular padding


def test_circular_pad_copies_the_opposite_edges():
    image = Image.new("RGB", (10, 4))
    pixels = image.load()
    for x in range(10):
        for y in range(4):
            pixels[x, y] = (x * 20, y * 50, 0)

    padded = circular_pad_image(image, 3)
    out = padded.load()

    assert padded.size == (16, 4)
    for y in range(4):
        for i in range(3):
            assert out[i, y] == pixels[7 + i, y]
            assert out[13 + i, y] == pixels[i, y]
        for x in range(10):
            assert out[3 + x, y] == pixels[x, y]


def test_circular_pad_of_zero_returns_the_same_image():
    image = Image.new("RGB", (10, 4))
    assert circular_pad_image(image, 0) is image


def test_pad_columns_scale_the_pixel_pad_to_the_token_grid():
    assert pad_columns(112, 1760, 126, merge=2) == 8
    assert pad_columns(112, 1760, 126) == 8
    assert pad_columns(0, 1760, 126) == 0
    assert pad_columns(100, 1000, 55, merge=2) % 2 == 0
