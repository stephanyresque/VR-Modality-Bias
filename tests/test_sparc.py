from __future__ import annotations

from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.nn as nn
from PIL import Image

from vr_modality_bias.experiment.sparc import (
    SparcHyperparams,
    enable_sparc,
    probe_image_token_index,
)


# ---------------------------------------------------------------- mock model


class _MockSelfAttn(nn.Module):
    """Minimal stand-in for an attention block. The ``forward`` is a sentinel
    so we can detect whether SPARC swapped it or not."""

    def __init__(self):
        super().__init__()
        # Bind a sentinel forward to this instance via MethodType — same
        # pattern the real layers use.
        self.forward = MethodType(self._original_forward, self)

    def _original_forward(self, *args, **kwargs):
        return ("ORIGINAL", args, kwargs)


class _MockDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _MockSelfAttn()


class _MockLanguageModel(nn.Module):
    def __init__(self, n_layers: int = 4):
        super().__init__()
        self.layers = nn.ModuleList([_MockDecoderLayer() for _ in range(n_layers)])


class _MockTopLevelModel(nn.Module):
    """Mirrors ``model.model.language_model`` path the SPARC code walks."""

    def __init__(self, n_layers: int = 4):
        super().__init__()
        self.model = SimpleNamespace(language_model=_MockLanguageModel(n_layers))
        self.config = SimpleNamespace(image_token_id=42)


class _MockTokenizer:
    pad_token_id = 0


class _MockProcessor:
    tokenizer = _MockTokenizer()

    def apply_chat_template(self, *_args, **_kwargs):
        return "<prefix>"

    def __call__(self, text, images=None, return_tensors="pt"):
        # Pretend the chat template gave us a 10-token prefix with image_token_id (42)
        # placed at positions 2..5 (4 image patches).
        ids = torch.tensor([[1, 2, 42, 42, 42, 42, 7, 8, 9, 10]])
        # Real BatchEncoding is dict-like; mimic that.
        return {"input_ids": ids}


class _MockWrapper:
    def __init__(self, n_layers: int = 4):
        self.model_id = "mock/test"
        self._model = _MockTopLevelModel(n_layers=n_layers)
        self._processor = _MockProcessor()
        self._device = torch.device("cpu")

    @staticmethod
    def _build_messages(prompt, image):
        return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]


def _blank_image() -> Image.Image:
    return Image.new("RGB", (8, 8))


# ---------------------------------------------------------------- tests


def test_sparc_hyperparams_rejects_alpha_one():
    with pytest.raises(ValueError):
        SparcHyperparams(alpha=1.0)


def test_sparc_hyperparams_rejects_alpha_below_one():
    with pytest.raises(ValueError):
        SparcHyperparams(alpha=0.8)


def test_sparc_hyperparams_accepts_alpha_above_one():
    h = SparcHyperparams(alpha=1.3)
    assert h.alpha == 1.3
    assert h.tau == 2.0
    assert h.selected_layer == 15
    assert h.se_layers == (0, 31)
    assert h.beta == 0.0


def test_sparc_hyperparams_as_dict_is_serialisable():
    h = SparcHyperparams(alpha=1.2, tau=1.5, selected_layer=10, se_layers=(5, 20), beta=0.1)
    d = h.as_dict()
    assert d == {
        "alpha": 1.2,
        "tau": 1.5,
        "selected_layer": 10,
        "se_layers": [5, 20],
        "beta": 0.1,
    }
    import json
    json.dumps(d)  # must not raise


def test_probe_image_token_index_returns_first_image_position():
    wrapper = _MockWrapper(n_layers=4)
    idx, input_len, n_patches = probe_image_token_index(wrapper, _blank_image(), "hello")
    # Mock returns ids [1, 2, 42, 42, 42, 42, 7, 8, 9, 10] — image_token_id=42 at pos 2..5
    assert idx == 2
    assert n_patches == 4
    assert input_len == 10 - 4  # total minus image patches


def test_enable_sparc_swaps_forward_for_each_layer():
    wrapper = _MockWrapper(n_layers=4)
    hparams = SparcHyperparams(alpha=1.3)

    # Snapshot original sentinel forward IDs.
    decoder = wrapper._model.model.language_model
    original_ids = [id(layer.self_attn.forward) for layer in decoder.layers]

    with enable_sparc(wrapper, hparams=hparams, probe_image=_blank_image(), prompt="hi") as buffer:
        # Inside the with-block, every layer's forward must have been replaced.
        for layer, orig_id in zip(decoder.layers, original_ids):
            assert id(layer.self_attn.forward) != orig_id, (
                "SPARC did not monkey-patch this layer's attention forward."
            )
        # Buffer is yielded and ready to use.
        assert buffer is not None
        assert hasattr(buffer, "reset")
        assert hasattr(buffer, "update_input_len")


def test_enable_sparc_restores_originals_on_normal_exit():
    wrapper = _MockWrapper(n_layers=4)
    hparams = SparcHyperparams(alpha=1.3)

    decoder = wrapper._model.model.language_model
    original_forwards = [layer.self_attn.forward for layer in decoder.layers]

    with enable_sparc(wrapper, hparams=hparams, probe_image=_blank_image(), prompt="hi"):
        pass  # do nothing — exit cleanly

    # After the with-block, every layer's forward must be the original one.
    for layer, original in zip(decoder.layers, original_forwards):
        assert layer.self_attn.forward is original, (
            "SPARC failed to restore the original attention forward on exit."
        )

    # Calling the restored forward should still produce the original sentinel.
    out = decoder.layers[0].self_attn.forward("arg")
    assert out[0] == "ORIGINAL"


def test_enable_sparc_restores_originals_on_exception():
    """Even when the with-block raises, the originals must come back.

    This protects against the classic foot-gun where a measurement crash
    leaves the model patched, then the next "baseline" run is silently
    a SPARC run.
    """
    wrapper = _MockWrapper(n_layers=4)
    hparams = SparcHyperparams(alpha=1.3)

    decoder = wrapper._model.model.language_model
    original_forwards = [layer.self_attn.forward for layer in decoder.layers]

    with pytest.raises(RuntimeError, match="injected"):
        with enable_sparc(wrapper, hparams=hparams, probe_image=_blank_image(), prompt="hi"):
            raise RuntimeError("injected")

    for layer, original in zip(decoder.layers, original_forwards):
        assert layer.self_attn.forward is original, (
            "SPARC must restore originals even when the with-block raised."
        )


def test_enable_sparc_yields_a_buffer_per_block():
    """Re-entering the context manager gives a fresh buffer each time."""
    wrapper = _MockWrapper(n_layers=4)
    hparams = SparcHyperparams(alpha=1.3)

    with enable_sparc(wrapper, hparams=hparams, probe_image=_blank_image(), prompt="hi") as b1:
        pass
    with enable_sparc(wrapper, hparams=hparams, probe_image=_blank_image(), prompt="hi") as b2:
        pass

    assert b1 is not b2


# ---------------------------------------------------------------- per-id mask buffer tests
#
# These tests pin the buffer's translation of local image-block indices
# (returned by ``(ratio >= tau).nonzero()``) to GLOBAL input_ids positions.
# Two paths:
#   * unified per-id mask (post-fix, ``image_positions`` set on the buffer)
#   * legacy contiguous-block fallback (pre-fix, ``image_token_index`` int)
# The unified path is the only correct one for Idefics3 / SmolVLM whose
# image-placeholder tokens are interleaved with row/column separators.


def test_buffer_per_id_mask_translates_local_indices_to_global():
    """When ``image_positions`` is set, ``update_indices1`` must gather."""
    from vr_modality_bias.utils.attn import SelectedIndexBuffer

    buf = SelectedIndexBuffer()
    # Idefics3-style non-contiguous layout: image tokens at positions
    # 100, 101, 105, 106 (gap at 102-104 = separator tokens).
    buf.update_image_positions(torch.tensor([100, 101, 105, 106], dtype=torch.long))

    # Selector picked LOCAL positions 0 and 2 inside the image block,
    # i.e. the 1st and 3rd image-placeholder tokens (global 100 and 105).
    local = torch.tensor([[0], [2]], dtype=torch.long)
    buf.update_indices1(local)

    assert torch.equal(buf.indices1, torch.tensor([100, 105], dtype=torch.long)), (
        f"Per-id mask translated local indices wrong: got {buf.indices1}"
    )


def test_buffer_per_id_mask_identical_to_legacy_for_contiguous_layout():
    """For a contiguous block (Qwen), per-id mask must match legacy slice."""
    from vr_modality_bias.utils.attn import SelectedIndexBuffer

    # Qwen-like: 5 image tokens at positions 30..34.
    image_token_index = 30
    image_positions = torch.tensor([30, 31, 32, 33, 34], dtype=torch.long)
    local = torch.tensor([[0], [2], [4]], dtype=torch.long)  # 1st, 3rd, 5th

    buf_new = SelectedIndexBuffer()
    buf_new.update_image_positions(image_positions)
    buf_new.update_indices1(local)

    buf_legacy = SelectedIndexBuffer()
    buf_legacy.update_indices1(local, image_token_index=image_token_index)

    assert torch.equal(buf_new.indices1, buf_legacy.indices1), (
        "Per-id mask must give the same result as the legacy contiguous "
        "slice when image positions actually ARE contiguous (Qwen case)."
    )
    assert torch.equal(buf_new.indices1, torch.tensor([30, 32, 34], dtype=torch.long))


def test_buffer_legacy_path_requires_image_token_index():
    """Without ``image_positions`` and without an explicit index, raise."""
    from vr_modality_bias.utils.attn import SelectedIndexBuffer

    buf = SelectedIndexBuffer()
    local = torch.tensor([[0]], dtype=torch.long)
    with pytest.raises(ValueError, match="image_positions"):
        buf.update_indices1(local)  # neither path available


def test_buffer_reset_clears_image_positions():
    """``reset()`` must clear the per-id mask so the next image starts clean."""
    from vr_modality_bias.utils.attn import SelectedIndexBuffer

    buf = SelectedIndexBuffer()
    buf.update_image_positions(torch.tensor([1, 2, 3], dtype=torch.long))
    buf.update_input_len(50)
    buf.reset()

    assert buf.image_positions is None
    assert buf.input_len == 0
    assert buf.indices1 == []
    assert buf.indices2 == []
    assert buf.num_image_patches is None


# ---------------------------------------------------------------- LLaVA-1.5 layout
# Post-Block-2 the only registered family is "llama"; the LLaVA-1.5
# checkpoint exposes the decoder at ``model.model.language_model.layers``.
# These tests pin both the layout discovery and the family marker.


def test_decoder_of_resolves_llava_layout():
    """decoder_of must walk model.model.language_model.layers for LLaVA-1.5."""
    import torch.nn as nn
    from vr_modality_bias.utils.attn import decoder_of

    class _FakeLayer(nn.Module):
        pass

    decoder = SimpleNamespace(layers=nn.ModuleList([_FakeLayer() for _ in range(32)]))
    inner = SimpleNamespace(language_model=decoder)
    fake_model = SimpleNamespace(model=inner)

    found = decoder_of(fake_model)
    assert found is decoder
    assert len(found.layers) == 32


def test_decoder_of_resolves_legacy_nested_llava_layout():
    """Older LLaVA shapes nest the decoder one level deeper: ``language_model.model.layers``."""
    import torch.nn as nn
    from vr_modality_bias.utils.attn import decoder_of

    class _FakeLayer(nn.Module):
        pass

    decoder = SimpleNamespace(layers=nn.ModuleList([_FakeLayer() for _ in range(32)]))
    # language_model is a LlamaForCausalLM-style wrapper; .model holds the decoder.
    language_model = SimpleNamespace(model=decoder)
    inner = SimpleNamespace(language_model=language_model)
    fake_model = SimpleNamespace(model=inner)

    found = decoder_of(fake_model)
    assert found is decoder


def test_detect_model_family_returns_llama_for_llava_class_name():
    """``LlavaForConditionalGeneration`` must be recognised as the llama family."""
    from vr_modality_bias.utils.attn import detect_model_family

    # Synthesize a class whose name starts with "Llava" — that's all the
    # marker check looks at; no torch modules involved.
    fake = type("LlavaForConditionalGeneration", (), {})()
    assert detect_model_family(fake) == "llama"
