from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from vr_modality_bias.io.storage import (
    hidden_states_filename,
    load_hidden_states,
    save_hidden_states,
)
from vr_modality_bias.models.base import HiddenStatesResult


def _make_synthetic_result(
    *,
    n_layers: int = 4,
    seq_len: int = 12,
    hidden_dim: int = 8,
    caption_start: int = 5,
    with_attention_mask: bool = True,
) -> HiddenStatesResult:
    """A small but realistic result for round-trip tests."""
    torch.manual_seed(0)
    hidden = torch.randn(n_layers, seq_len, hidden_dim, dtype=torch.float16)
    input_ids = torch.arange(seq_len, dtype=torch.int64) + 100
    attention_mask = (
        torch.ones(seq_len, dtype=torch.int8) if with_attention_mask else None
    )
    return HiddenStatesResult(
        hidden_states=hidden,
        input_ids=input_ids,
        caption_start=caption_start,
        caption_len=seq_len - caption_start,
        metadata={
            "model_id": "synthetic/test",
            "hidden_dim": hidden_dim,
            "n_layers": n_layers,
        },
        attention_mask=attention_mask,
    )


def test_hidden_states_filename_canonical_format():
    assert hidden_states_filename("000000000139", "A") == "000000000139__A.h5"
    assert hidden_states_filename("img_42", "B") == "img_42__B.h5"


def test_hidden_states_filename_rejects_unknown_condition():
    with pytest.raises(ValueError):
        hidden_states_filename("000000000139", "C")


def test_roundtrip_preserves_arrays_and_metadata(tmp_path: Path):
    result = _make_synthetic_result()
    path = tmp_path / hidden_states_filename("img_test", "A")
    extra = {
        "image_id": "img_test",
        "prompt_key": "caption_short",
        "seed_global": 42,
        "noise_seed": 12345,
        "caption_ref": "A red rectangle.",
        "timestamp_iso": "2026-05-27T14:30:00+00:00",
    }
    save_hidden_states(path, result, condition="A", extra_attrs=extra)
    assert path.is_file()

    loaded = load_hidden_states(path)
    assert loaded.hidden_states.shape == result.hidden_states.shape
    assert loaded.hidden_states.dtype == torch.float16
    np.testing.assert_array_equal(
        loaded.hidden_states.numpy(), result.hidden_states.numpy()
    )
    assert torch.equal(loaded.input_ids, result.input_ids)
    assert loaded.caption_start == result.caption_start
    assert loaded.caption_len == result.caption_len
    assert loaded.attention_mask is not None
    assert torch.equal(loaded.attention_mask, result.attention_mask)

    # canonical attrs
    assert loaded.metadata["condition"] == "A"
    assert loaded.metadata["image_id"] == "img_test"
    assert loaded.metadata["prompt_key"] == "caption_short"
    assert loaded.metadata["seed_global"] == 42
    assert loaded.metadata["noise_seed"] == 12345
    assert loaded.metadata["caption_ref"] == "A red rectangle."
    assert loaded.metadata["hidden_dim"] == result.hidden_states.shape[-1]


def test_attention_mask_is_optional(tmp_path: Path):
    result = _make_synthetic_result(with_attention_mask=False)
    path = tmp_path / hidden_states_filename("img_no_mask", "B")
    save_hidden_states(
        path,
        result,
        condition="B",
        extra_attrs={"image_id": "img_no_mask"},
    )
    loaded = load_hidden_states(path)
    assert loaded.attention_mask is None


def test_save_rejects_unknown_condition(tmp_path: Path):
    result = _make_synthetic_result()
    with pytest.raises(ValueError):
        save_hidden_states(tmp_path / "bad.h5", result, condition="C")


def test_save_creates_parent_dirs(tmp_path: Path):
    result = _make_synthetic_result()
    nested = tmp_path / "deep" / "deeper" / "img__A.h5"
    save_hidden_states(nested, result, condition="A", extra_attrs={"image_id": "img"})
    assert nested.is_file()


def test_save_persists_arbitrary_extra_attrs(tmp_path: Path):
    result = _make_synthetic_result()
    path = tmp_path / "img__A.h5"
    save_hidden_states(
        path,
        result,
        condition="A",
        extra_attrs={"image_id": "img", "extra_debug_field": "hello"},
    )
    loaded = load_hidden_states(path)
    assert loaded.metadata["extra_debug_field"] == "hello"
