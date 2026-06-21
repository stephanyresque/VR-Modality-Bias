from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn
from PIL import Image

from vr_modality_bias.experiment.forced_decoding import collect_forced_decoding


# ---------------------------------------------------------------- mock model


def _encode(pos: int, layer: int, dim: int) -> float:
    """Deterministic synthetic hidden state.

    Encoding chosen so that a shift of even one position changes every
    cell of the output tensor by at least 1000 — completely unambiguous.
    """
    return float(pos * 1000 + (layer + 1) * 10 + dim)


class _MockTokenizer:
    pad_token_id = 0

    def decode(self, ids, skip_special_tokens=False):
        return "".join(f"<{int(i)}>" for i in ids)


class _MockProcessorOutput(dict):
    """Dict subclass with .to() so we look like a transformers BatchEncoding."""

    def to(self, *args, **kwargs):
        out = _MockProcessorOutput()
        for k, v in self.items():
            out[k] = v.to(*args, **kwargs) if isinstance(v, torch.Tensor) else v
        return out


class _MockProcessor:
    """Returns deterministic input_ids based on the text length only."""

    tokenizer = _MockTokenizer()

    def __init__(self, prefix_len: int, caption_len: int):
        # The mock will emit input_ids = [10, 11, 12, ...] of length prefix_len
        # for the prefix-only call, and prefix_len + caption_len for the
        # full call. The exact values are irrelevant; we only check shape.
        self._prefix_len = prefix_len
        self._caption_len = caption_len

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False):
        # Caller appends ``caption_ref.strip()`` to whatever we return. The
        # content doesn't matter as long as the length we feed to the model
        # via __call__ matches.
        return "<prefix>"

    def __call__(self, text, images=None, return_tensors="pt"):
        # The caller tokenises twice: once for prefix only, once for
        # "prefix + caption_ref". We detect which via the string length
        # (caption was appended → longer).
        assert isinstance(text, list) and len(text) == 1
        rendered = text[0]
        if rendered == "<prefix>":
            length = self._prefix_len
        else:
            length = self._prefix_len + self._caption_len
        input_ids = torch.arange(10, 10 + length, dtype=torch.int64).view(1, length)
        attn = torch.ones((1, length), dtype=torch.int64)
        # Mimic Qwen's vision keys at the prefix step too (the real
        # processor puts them there) so collect_forced_decoding finds them.
        out = _MockProcessorOutput()
        out["input_ids"] = input_ids
        out["attention_mask"] = attn
        out["pixel_values"] = torch.zeros((1, 3, 8, 8))
        out["image_grid_thw"] = torch.tensor([[1, 2, 2]])
        return out


class _MockModel(nn.Module):
    """Mocks a Qwen-VL-style model — just enough surface for forced decoding.

    ``forward`` returns hidden_states populated from
    :func:`_encode`, and ``prepare_inputs_for_generation`` slices ``input_ids``
    based on the cache length so the loop hits ``forward`` with only the
    new token at each step.
    """

    def __init__(self, n_layers: int = 3, hidden_dim: int = 4) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            image_token_id=999_999,  # not in our mock input_ids
            hidden_size=hidden_dim,
        )
        self.model = SimpleNamespace(
            language_model=SimpleNamespace(layers=[]),
        )  # so ``decoder_of`` returns something — not used here
        self._n_layers = n_layers
        self._hidden_dim = hidden_dim

    @property
    def device(self):
        return torch.device("cpu")

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        pixel_values=None,
        image_grid_thw=None,
        use_cache=True,
        is_first_iteration=False,
        **kwargs,
    ):
        cache_len = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cache_len > 0:
            new_ids = input_ids[:, cache_len:]
            cache_position = torch.arange(cache_len, input_ids.shape[1])
        else:
            new_ids = input_ids
            cache_position = torch.arange(input_ids.shape[1])
        out: dict[str, Any] = {
            "input_ids": new_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "cache_position": cache_position,
            "use_cache": use_cache,
        }
        if is_first_iteration:
            out["pixel_values"] = pixel_values
            out["image_grid_thw"] = image_grid_thw
        return out

    def forward(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        use_cache=False,
        output_hidden_states=False,
        return_dict=False,
        pixel_values=None,
        image_grid_thw=None,
        **kwargs,
    ):
        assert output_hidden_states, "Forced decoding must request hidden states."
        assert return_dict, "Forced decoding must request return_dict."
        seq_len = input_ids.shape[1]
        # Mirror what the real Qwen-2.5-VL model does: derive the absolute
        # position of each new token from the cache length plus an arange.
        # This is what ``compute_3d_position_ids`` does in the ``else`` branch
        # (no attention_mask) — see the module docstring of forced_decoding.
        cache_len = past_key_values.get_seq_length() if past_key_values is not None else 0
        positions = list(range(cache_len, cache_len + seq_len))

        # Embedding output (we never read this).
        embedding = torch.zeros((1, seq_len, self._hidden_dim))
        layers_out = [embedding]
        for layer_idx in range(self._n_layers):
            h = torch.zeros((1, seq_len, self._hidden_dim))
            for i, p in enumerate(positions):
                for d in range(self._hidden_dim):
                    h[0, i, d] = _encode(p, layer_idx, d)
            layers_out.append(h)

        # Advance the cache so prepare_inputs_for_generation slices
        # correctly on the next call. We mock K/V with zeros — the real
        # values are irrelevant to forced_decoding.
        if past_key_values is not None and use_cache:
            for layer_idx in range(self._n_layers):
                dummy_k = torch.zeros((1, 1, seq_len, 1))
                dummy_v = torch.zeros((1, 1, seq_len, 1))
                past_key_values.update(dummy_k, dummy_v, layer_idx)

        return SimpleNamespace(
            hidden_states=tuple(layers_out),
            past_key_values=past_key_values,
            logits=torch.zeros((1, seq_len, 50)),
        )


class _MockWrapper:
    """Drop-in stand-in for QwenVLWrapper that the function actually consumes."""

    def __init__(self, n_layers: int = 3, hidden_dim: int = 4, *, prefix_len: int = 5, caption_len: int = 4):
        self.model_id = "mock/test"
        self._model = _MockModel(n_layers=n_layers, hidden_dim=hidden_dim)
        self._processor = _MockProcessor(prefix_len=prefix_len, caption_len=caption_len)
        self._device = torch.device("cpu")
        self._n_layers = n_layers

    @property
    def n_layers(self) -> int:
        return self._n_layers

    @staticmethod
    def _build_messages(prompt, image):
        return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]


def _blank_image() -> Image.Image:
    arr = torch.zeros((8, 8, 3), dtype=torch.uint8).numpy()
    return Image.fromarray(arr, mode="RGB")


# ---------------------------------------------------------------- tests


def test_output_shape_matches_run_teacher_forcing_layout():
    wrapper = _MockWrapper(n_layers=3, hidden_dim=4, prefix_len=5, caption_len=4)
    result = collect_forced_decoding(
        wrapper, _blank_image(), "describe", "abcd",
        output_dtype=torch.float32,
    )
    # caption_start = prefix_len = 5; caption_len comes from the
    # difference between the full and prefix tokenisations (4 here).
    assert result.caption_start == 5
    assert result.caption_len == 4
    # Total seq_len = 5 + 4 = 9. Hidden states tensor: (n_layers, 9, 4).
    assert result.hidden_states.shape == (3, 9, 4)
    assert result.input_ids.shape == (9,)


def test_prefill_fills_indices_zero_to_caption_start_minus_one():
    """Indices 0..caption_start-1 must contain the prefill outputs."""
    wrapper = _MockWrapper(n_layers=3, hidden_dim=4, prefix_len=5, caption_len=4)
    result = collect_forced_decoding(
        wrapper, _blank_image(), "describe", "abcd",
        output_dtype=torch.float32,
    )
    # The prefill at position p, layer l, dim d should equal _encode(p, l, d).
    # ``hidden_states`` is (n_layers, seq_len, hidden_dim); positions in
    # ``[0, 5)`` are prefill territory.
    for layer in range(3):
        for pos in range(5):
            for dim in range(4):
                assert result.hidden_states[layer, pos, dim].item() == _encode(pos, layer, dim), (
                    f"prefill slot (layer={layer}, pos={pos}, dim={dim}) wrong"
                )


def test_each_step_lands_at_its_own_absolute_index():
    """The j-th step (which feeds the token at absolute position caption_start + j)
    must write to that same absolute index."""
    wrapper = _MockWrapper(n_layers=3, hidden_dim=4, prefix_len=5, caption_len=4)
    result = collect_forced_decoding(
        wrapper, _blank_image(), "describe", "abcd",
        output_dtype=torch.float32,
    )
    caption_start = result.caption_start
    for j in range(result.caption_len):
        abs_pos = caption_start + j
        for layer in range(3):
            for dim in range(4):
                expected = _encode(abs_pos, layer, dim)
                got = result.hidden_states[layer, abs_pos, dim].item()
                assert got == expected, (
                    f"step j={j} (abs_pos={abs_pos}, layer={layer}, dim={dim}): "
                    f"expected {expected}, got {got}"
                )


def test_anti_off_by_one_predictive_state_alignment():
    """The critical alignment test that motivated Phase 1's index spec.

    For every caption token ``j``, ``compute_kl_matrix`` reads the
    predictive state at the **absolute** index ``caption_start - 1 + j``.
    We assert that:

      * For ``j == 0``: that slot was populated by the **prefill** (its
        encoding uses position ``caption_start - 1``), NOT by any step.
      * For ``j >= 1``: that slot was populated by the step whose
        ``cache_position`` was exactly ``caption_start - 1 + j``, i.e. the
        step that fed ``caption_ref[j - 1]``.

    A shift of one position in any direction would change the encoded
    value by 1000 — so this test catches both kinds of off-by-one.
    """
    wrapper = _MockWrapper(n_layers=3, hidden_dim=4, prefix_len=5, caption_len=4)
    result = collect_forced_decoding(
        wrapper, _blank_image(), "describe", "abcd",
        output_dtype=torch.float32,
    )
    caption_start = result.caption_start

    for j in range(result.caption_len):
        predictive_idx = caption_start - 1 + j
        for layer in range(3):
            for dim in range(4):
                expected_for_correct_alignment = _encode(predictive_idx, layer, dim)
                got = result.hidden_states[layer, predictive_idx, dim].item()
                # Off-by-one detection: if the loop accidentally wrote
                # this slot using a different position, we'd get a value
                # off by at least 1000 (the position multiplier).
                assert got == expected_for_correct_alignment, (
                    f"PREDICTIVE STATE MISALIGNED for caption token j={j}: "
                    f"slot (layer={layer}, abs_pos={predictive_idx}, dim={dim}) "
                    f"should hold {expected_for_correct_alignment}; got {got}. "
                    "The loop is off by at least one position — fix scatter "
                    "logic in forced_decoding._scatter_prefill_into / _scatter_step_into."
                )


def test_no_extra_or_missing_positions_in_output():
    """Every absolute index in [0, seq_len) must have been populated.

    A slot left at zero would mean a step was missed; a slot containing
    something other than _encode(p, ...) means a step wrote to the wrong
    position. Either is fatal.
    """
    wrapper = _MockWrapper(n_layers=2, hidden_dim=3, prefix_len=4, caption_len=3)
    result = collect_forced_decoding(
        wrapper, _blank_image(), "describe", "xyz",
        output_dtype=torch.float32,
    )
    seq_len = result.hidden_states.shape[1]
    for layer in range(2):
        for pos in range(seq_len):
            for dim in range(3):
                expected = _encode(pos, layer, dim)
                got = result.hidden_states[layer, pos, dim].item()
                assert got == expected, (
                    f"Slot (layer={layer}, pos={pos}, dim={dim}) = {got}, "
                    f"expected {expected}"
                )


def test_returns_caption_ref_input_ids_unchanged():
    """input_ids on the result must match the full tokenisation byte-by-byte."""
    wrapper = _MockWrapper(n_layers=2, hidden_dim=3, prefix_len=4, caption_len=3)
    result = collect_forced_decoding(
        wrapper, _blank_image(), "describe", "xyz",
        output_dtype=torch.float32,
    )
    # The mock processor uses arange(10, 10+length); length here is 7.
    expected = torch.arange(10, 17, dtype=torch.int64)
    assert torch.equal(result.input_ids, expected)


def test_caption_len_zero_raises():
    """An empty caption_ref → caption_len == 0 → reject loudly."""
    wrapper = _MockWrapper(n_layers=2, hidden_dim=3, prefix_len=4, caption_len=0)
    with pytest.raises(RuntimeError):
        collect_forced_decoding(
            wrapper, _blank_image(), "describe", "",
            output_dtype=torch.float32,
        )


def test_unloaded_model_raises():
    """Wrapper with no loaded model should raise instead of crashing inside."""

    class _Bare:
        _model = None
        _processor = None
        _device = None

    with pytest.raises(RuntimeError, match="Model not loaded"):
        collect_forced_decoding(_Bare(), _blank_image(), "p", "c")


def test_metadata_marks_collection_method_and_sparc_inactive():
    wrapper = _MockWrapper(n_layers=2, hidden_dim=3, prefix_len=4, caption_len=3)
    result = collect_forced_decoding(
        wrapper, _blank_image(), "describe", "xyz",
        output_dtype=torch.float32,
    )
    assert result.metadata["collection_method"] == "forced_decoding"
    assert result.metadata["sparc_active"] is False
    assert result.metadata["n_layers"] == 2
    assert result.metadata["hidden_dim"] == 3
