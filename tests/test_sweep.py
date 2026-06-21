"""Tests for the caption-sweep extension (scripts 09 + 10).

Covers everything that can be validated CPU-only:
    - registry exposes the post-Block-2 model keys,
    - the three caption prompts are active (no None),
    - decode_caption_tokens slices and decodes per-position,
    - METRICS_SCHEMA has the new nullable caption_tokens field,
    - write/read parquet survives both populated and absent caption_tokens.

Model loading (LLaVA-1.5-7B) is **not** exercised here — that's DGX
territory. The Block-2 migration retired SmolVLM and Qwen2.5-VL, so the
sweep-orchestrator tests that used those keys are gone with them (the
orchestrator in scripts/10_run_sweep.py is listed as meaningless until a
new LLaVA sweep config is wired).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch


# ---------------------------------------------------------------- registry


def test_registry_lists_only_llava_post_block2():
    """Block 2 retired SmolVLM and Qwen2.5-VL; LLaVA-1.5-7B is the sole key."""
    from vr_modality_bias.models.registry import list_models

    assert list_models() == ["llava-1.5-7b"]


def test_registry_builds_llava_without_loading():
    """Building the wrapper must not touch the GPU / download weights."""
    from vr_modality_bias.models.llava import LlavaWrapper
    from vr_modality_bias.models.registry import build_model

    wrapper = build_model("llava-1.5-7b")
    assert isinstance(wrapper, LlavaWrapper)
    assert "llava" in wrapper.model_id.lower()
    # Not yet loaded — n_layers must error.
    with pytest.raises(RuntimeError):
        _ = wrapper.n_layers


# ----------------------------------------------------------------- prompts


def test_caption_prompts_are_active():
    from vr_modality_bias.data.prompts import PROMPTS, get_prompt

    for key in ("caption_short", "caption_medium", "caption_long"):
        assert PROMPTS[key] is not None, f"{key} must be active"
        assert isinstance(get_prompt(key), str)
        assert len(get_prompt(key)) > 20


def test_caption_long_removes_do_not_speculate_clause():
    """Long-caption cell is meant to surface hallucination — no muzzle."""
    from vr_modality_bias.data.prompts import get_prompt

    text = get_prompt("caption_long")
    assert "speculate" not in text.lower()


# ---------------------------------------------------------- token decoding


class _FakeTokenizer:
    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        # Trivial mapping: token id -> "tok<id>"; "[id]" if special and not skipped
        assert len(ids) == 1
        tid = int(ids[0])
        if tid < 0 and skip_special_tokens:
            return ""
        return f"tok{tid}"


class _FakeProcessor:
    def __init__(self):
        self.tokenizer = _FakeTokenizer()


class _FakeWrapper:
    def __init__(self):
        self._processor = _FakeProcessor()


def test_decode_caption_tokens_returns_one_string_per_caption_position():
    from vr_modality_bias.utils.tokens import decode_caption_tokens

    wrapper = _FakeWrapper()
    input_ids = torch.tensor([10, 11, 12, 20, 21, 22, 23], dtype=torch.int64)
    tokens = decode_caption_tokens(wrapper, input_ids, caption_start=3)
    assert tokens == ["tok20", "tok21", "tok22", "tok23"]


def test_decode_caption_tokens_empty_caption():
    from vr_modality_bias.utils.tokens import decode_caption_tokens

    wrapper = _FakeWrapper()
    input_ids = torch.tensor([1, 2, 3], dtype=torch.int64)
    tokens = decode_caption_tokens(wrapper, input_ids, caption_start=3)
    assert tokens == []


def test_decode_caption_tokens_rejects_bad_caption_start():
    from vr_modality_bias.utils.tokens import decode_caption_tokens

    wrapper = _FakeWrapper()
    input_ids = torch.tensor([1, 2, 3], dtype=torch.int64)
    with pytest.raises(ValueError):
        decode_caption_tokens(wrapper, input_ids, caption_start=-1)
    with pytest.raises(ValueError):
        decode_caption_tokens(wrapper, input_ids, caption_start=99)


def test_decode_caption_tokens_rejects_non_1d():
    from vr_modality_bias.utils.tokens import decode_caption_tokens

    wrapper = _FakeWrapper()
    ids = torch.zeros((2, 3), dtype=torch.int64)
    with pytest.raises(ValueError):
        decode_caption_tokens(wrapper, ids, caption_start=0)


def test_decode_caption_tokens_errors_without_processor():
    from vr_modality_bias.utils.tokens import decode_caption_tokens

    class Bare:
        pass

    with pytest.raises(RuntimeError):
        decode_caption_tokens(Bare(), torch.zeros(3, dtype=torch.int64), 0)


# ------------------------------------------------------------- parquet schema


def test_metrics_schema_has_caption_tokens_field():
    from vr_modality_bias.io.results import METRICS_SCHEMA

    field = METRICS_SCHEMA.field("caption_tokens")
    assert field.nullable, "caption_tokens must be nullable for backwards compat"


def test_write_metrics_with_caption_tokens_roundtrips(tmp_path: Path):
    from vr_modality_bias.io.results import (
        METRICS_SCHEMA,
        read_metrics_table,
        write_metrics_table,
    )

    rows = [
        {
            "image_id": "img_001",
            "caption_len": 4,
            "n_layers": 2,
            "hidden_dim": 8,
            "caption_ref": "A red box.",
            "kl": [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
            "cos_dist": [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            "residual_ratio": 0.5,
            "model_id": "test/model",
            "prompt_key": "caption_short",
            "seed_global": 42,
            "noise_seed": 12345,
            "timestamp_iso": "2026-06-01T12:00:00+00:00",
            "caption_tokens": ["A", " red", " box", "."],
        }
    ]
    path = tmp_path / "metrics.parquet"
    write_metrics_table(rows, path)

    table = pq.read_table(path)
    assert table.schema.equals(METRICS_SCHEMA)
    back = read_metrics_table(path)
    assert back[0]["caption_tokens"] == ["A", " red", " box", "."]


def test_write_metrics_without_caption_tokens_writes_null(tmp_path: Path):
    """Old-path rows (no caption_tokens) write as null and stay readable."""
    from vr_modality_bias.io.results import read_metrics_table, write_metrics_table

    rows = [
        {
            "image_id": "img_legacy",
            "caption_len": 2,
            "n_layers": 1,
            "hidden_dim": 4,
            "caption_ref": "X.",
            "kl": [[0.1, 0.2]],
            "cos_dist": [[0.0, 0.0]],
            "residual_ratio": 0.1,
            "model_id": "m",
            "prompt_key": "caption_short",
            "seed_global": 0,
            "noise_seed": 0,
            "timestamp_iso": "2026-01-01T00:00:00+00:00",
            # caption_tokens omitted
        }
    ]
    path = tmp_path / "legacy.parquet"
    write_metrics_table(rows, path)
    back = read_metrics_table(path)
    assert back[0]["caption_tokens"] is None


# -------------------------------------------------------- discard semantics


def test_discard_flag_default_is_false_in_baseline_config():
    """The baseline must keep old behaviour: no auto-discard of h5."""
    import yaml

    cfg = yaml.safe_load(
        Path("configs/baseline.yaml").read_text(encoding="utf-8")
    )
    assert not bool(
        cfg.get("io", {}).get("discard_hidden_states_after_metrics", False)
    )


# Note: the sweep orchestrator tests (expand_cells, sweep-config validation)
# were dropped in the Block-2 migration. They relied on the SmolVLM and
# Qwen2.5-VL keys + their six per-length configs, all of which are gone.
# scripts/10_run_sweep.py is documented as orphaned until a new LLaVA
# sweep wiring lands.
