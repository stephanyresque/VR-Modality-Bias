"""Tests for the caption-sweep extension (run_caption_sweep.py + run_sweep.py).

Covers everything that can be validated CPU-only:
    - registry has the three model keys after the extension,
    - the three caption prompts are active (no None),
    - decode_caption_tokens slices and decodes per-position,
    - METRICS_SCHEMA has the new nullable caption_tokens field,
    - write/read parquet survives both populated and absent caption_tokens,
    - sweep orchestrator's combination expansion (run_sweep.py) is correct.

Model loading (SmolVLM-2.2B / Qwen2.5-VL-7B) is **not** exercised here —
that's DGX territory.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch


_SCRIPT_10 = Path(__file__).parent.parent / "scripts" / "run_sweep.py"


def _load_script_10():
    spec = importlib.util.spec_from_file_location("script_10", _SCRIPT_10)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- registry


def test_registry_has_all_three_model_keys():
    from vr_modality_bias.models.registry import list_models

    keys = list_models()
    assert "smolvlm-256m" in keys
    assert "smolvlm-2.2b" in keys
    assert "qwen2.5-vl-7b" in keys


def test_registry_builds_smolvlm_2_2b_without_loading():
    from vr_modality_bias.models.registry import build_model
    from vr_modality_bias.models.smolvlm import SmolVLMWrapper

    wrapper = build_model("smolvlm-2.2b")
    assert isinstance(wrapper, SmolVLMWrapper)
    assert "SmolVLM" in wrapper.model_id
    # Not yet loaded — n_layers must error
    with pytest.raises(RuntimeError):
        _ = wrapper.n_layers


def test_registry_builds_qwen_without_loading():
    from vr_modality_bias.models.qwen_vl import QwenVLWrapper
    from vr_modality_bias.models.registry import build_model

    wrapper = build_model("qwen2.5-vl-7b")
    assert isinstance(wrapper, QwenVLWrapper)
    assert "Qwen" in wrapper.model_id
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


# ----------------------------------------------------- sweep orchestrator


def test_expand_cells_full_matrix_when_no_filters():
    script = _load_script_10()
    cells = script.expand_cells(models=None, lengths=None)
    assert len(cells) == 6
    pairs = {(m, length) for m, length, _ in cells}
    assert pairs == {
        ("smolvlm-2.2b", "short"),
        ("smolvlm-2.2b", "medium"),
        ("smolvlm-2.2b", "long"),
        ("qwen2.5-vl-7b", "short"),
        ("qwen2.5-vl-7b", "medium"),
        ("qwen2.5-vl-7b", "long"),
    }


def test_expand_cells_filters_by_model():
    script = _load_script_10()
    cells = script.expand_cells(models=["smolvlm-2.2b"], lengths=None)
    assert len(cells) == 3
    assert all(m == "smolvlm-2.2b" for m, _, _ in cells)


def test_expand_cells_filters_by_length():
    script = _load_script_10()
    cells = script.expand_cells(models=None, lengths=["long"])
    assert len(cells) == 2
    assert all(length == "long" for _, length, _ in cells)


def test_expand_cells_filters_by_both():
    script = _load_script_10()
    cells = script.expand_cells(models=["qwen2.5-vl-7b"], lengths=["short", "medium"])
    assert len(cells) == 2
    pairs = {(m, length) for m, length, _ in cells}
    assert pairs == {
        ("qwen2.5-vl-7b", "short"),
        ("qwen2.5-vl-7b", "medium"),
    }


def test_expand_cells_rejects_unknown_model():
    script = _load_script_10()
    with pytest.raises(ValueError) as exc_info:
        script.expand_cells(models=["mystery-7b"], lengths=None)
    assert "mystery-7b" in str(exc_info.value)


def test_expand_cells_rejects_unknown_length():
    script = _load_script_10()
    with pytest.raises(ValueError) as exc_info:
        script.expand_cells(models=None, lengths=["epic"])
    assert "epic" in str(exc_info.value)


def test_expand_cells_returns_existing_configs():
    """Every (model, length) cell must point to a real config file on disk."""
    script = _load_script_10()
    cells = script.expand_cells(models=None, lengths=None)
    for model, length, config_path in cells:
        assert config_path.is_file(), (
            f"config for ({model}, {length}) missing on disk: {config_path}"
        )


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


def test_discard_flag_true_in_sweep_configs():
    """Every sweep cell config must opt in to per-image discard."""
    import yaml

    sweep_configs = [
        "configs/run_smolvlm22_short.yaml",
        "configs/run_smolvlm22_medium.yaml",
        "configs/run_smolvlm22_long.yaml",
        "configs/run_qwen7b_short.yaml",
        "configs/run_qwen7b_medium.yaml",
        "configs/run_qwen7b_long.yaml",
    ]
    for path in sweep_configs:
        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        assert (
            cfg.get("io", {}).get("discard_hidden_states_after_metrics") is True
        ), f"{path} must have io.discard_hidden_states_after_metrics: true"


def test_sweep_configs_have_expected_n_images():
    """All six sweep cells use 50 images (per spec)."""
    import yaml

    for path in Path("configs").glob("run_*.yaml"):
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert int(cfg["dataset"]["n_images"]) == 50, (
            f"{path}: n_images should be 50, got {cfg['dataset']['n_images']}"
        )


def test_sweep_configs_have_matching_prompt_keys():
    """prompt_key must match the length suffix in the config name."""
    import yaml

    expected = {
        "short": "caption_short",
        "medium": "caption_medium",
        "long": "caption_long",
    }
    for path in Path("configs").glob("run_*.yaml"):
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        suffix = path.stem.rsplit("_", 1)[-1]
        assert cfg["task"]["prompt_key"] == expected[suffix], path
