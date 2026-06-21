from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from vr_modality_bias.io.results import (
    METRICS_SCHEMA,
    compute_summary_stats,
    read_metrics_table,
    write_metrics_table,
    write_summary_csv,
    write_summary_json,
)


def _sample_row(
    image_id: str = "img_001",
    *,
    n_layers: int = 3,
    caption_len: int = 4,
    hidden_dim: int = 8,
    residual_ratio: float = 0.42,
) -> dict:
    rng = np.random.default_rng(0)
    return {
        "image_id": image_id,
        "caption_len": caption_len,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "caption_ref": "A red shape on a table.",
        "kl": rng.random((n_layers, caption_len), dtype=np.float32).astype(np.float32),
        "cos_dist": rng.random((n_layers, caption_len), dtype=np.float32).astype(
            np.float32
        ),
        "residual_ratio": residual_ratio,
        "model_id": "HuggingFaceTB/SmolVLM-256M-Instruct",
        "prompt_key": "caption_short",
        "seed_global": 42,
        "noise_seed": 1_234_567_890,
        "timestamp_iso": "2026-05-27T14:30:00+00:00",
    }


def test_write_and_read_roundtrip_preserves_typed_nested_lists(tmp_path: Path):
    rows = [_sample_row("img_001"), _sample_row("img_002", residual_ratio=0.71)]
    path = tmp_path / "metrics.parquet"
    n = write_metrics_table(rows, path)
    assert n == 2

    table = pq.read_table(path)
    assert table.schema.equals(METRICS_SCHEMA)

    back = read_metrics_table(path)
    assert len(back) == 2
    assert back[0]["image_id"] == "img_001"
    assert back[1]["image_id"] == "img_002"
    assert back[0]["caption_len"] == rows[0]["caption_len"]
    assert back[0]["n_layers"] == rows[0]["n_layers"]
    # kl/cos_dist come back as Python list-of-list-of-float
    kl_back = np.asarray(back[0]["kl"], dtype=np.float32)
    np.testing.assert_array_equal(kl_back, np.asarray(rows[0]["kl"], dtype=np.float32))
    cos_back = np.asarray(back[0]["cos_dist"], dtype=np.float32)
    np.testing.assert_array_equal(
        cos_back, np.asarray(rows[0]["cos_dist"], dtype=np.float32)
    )


def test_write_handles_nan_residual_ratio(tmp_path: Path):
    rows = [_sample_row("img_nan", residual_ratio=float("nan"))]
    path = tmp_path / "metrics.parquet"
    write_metrics_table(rows, path)
    back = read_metrics_table(path)
    assert math.isnan(back[0]["residual_ratio"])


def test_write_handles_numpy_matrix_input(tmp_path: Path):
    """Accepts NumPy arrays directly for kl/cos_dist, not just lists."""
    row = _sample_row("img_np")
    row["kl"] = np.full((3, 4), 0.25, dtype=np.float32)
    row["cos_dist"] = np.zeros((3, 4), dtype=np.float32)
    path = tmp_path / "metrics.parquet"
    write_metrics_table([row], path)
    back = read_metrics_table(path)
    np.testing.assert_array_equal(
        np.asarray(back[0]["kl"], dtype=np.float32),
        np.full((3, 4), 0.25, dtype=np.float32),
    )


def test_compute_summary_stats_on_known_values():
    """For residual_ratio = [0.1, 0.2, 0.3, 0.4, 0.5]:
    median = 0.3, q25 = 0.2, q75 = 0.4, iqr = 0.2.
    """
    rows = [
        {"residual_ratio": v, "model_id": "m", "prompt_key": "caption_short"}
        for v in [0.1, 0.2, 0.3, 0.4, 0.5]
    ]
    stats = compute_summary_stats(rows)
    rr = stats["residual_ratio"]
    assert stats["n_images"] == 5
    assert stats["n_residual_ratio_finite_in_range"] == 5
    assert math.isclose(rr["median"], 0.3, rel_tol=1e-9)
    assert math.isclose(rr["q25"], 0.2, rel_tol=1e-9)
    assert math.isclose(rr["q75"], 0.4, rel_tol=1e-9)
    assert math.isclose(rr["iqr"], 0.2, rel_tol=1e-9)
    assert math.isclose(rr["min"], 0.1, rel_tol=1e-9)
    assert math.isclose(rr["max"], 0.5, rel_tol=1e-9)


def test_compute_summary_stats_includes_head_tail_ratio():
    """head_tail_ratio stats should be reported alongside residual_ratio."""
    rows = [
        {"residual_ratio": 0.5, "head_tail_ratio": v, "model_id": "m"}
        for v in [0.2, 0.4, 0.6, 0.8, 1.0]
    ]
    stats = compute_summary_stats(rows)
    htr = stats["head_tail_ratio"]
    assert stats["n_head_tail_ratio_finite"] == 5
    assert math.isclose(htr["median"], 0.6, rel_tol=1e-9)
    assert math.isclose(htr["min"], 0.2, rel_tol=1e-9)
    assert math.isclose(htr["max"], 1.0, rel_tol=1e-9)


def test_compute_summary_stats_head_tail_ratio_no_range_filter():
    """head_tail_ratio is unbounded above; values >1 must be kept in stats."""
    rows = [
        {"residual_ratio": 0.5, "head_tail_ratio": v}
        for v in [0.5, 1.0, 1.5, 3.0]
    ]
    stats = compute_summary_stats(rows)
    htr = stats["head_tail_ratio"]
    assert stats["n_head_tail_ratio_finite"] == 4
    assert math.isclose(htr["max"], 3.0, rel_tol=1e-9)
    assert math.isclose(htr["median"], 1.25, rel_tol=1e-9)


def test_compute_summary_stats_head_tail_ratio_handles_nan():
    rows = [
        {"residual_ratio": 0.5, "head_tail_ratio": v}
        for v in [0.3, float("nan"), 0.7, float("nan"), 0.9]
    ]
    stats = compute_summary_stats(rows)
    assert stats["n_head_tail_ratio_finite"] == 3
    assert math.isclose(stats["head_tail_ratio"]["median"], 0.7, rel_tol=1e-9)


def test_compute_summary_stats_head_tail_ratio_missing_column_is_none_safe():
    """Old-style rows without head_tail_ratio shouldn't crash compute_summary_stats."""
    rows = [{"residual_ratio": 0.5, "model_id": "m"} for _ in range(3)]
    stats = compute_summary_stats(rows)
    assert stats["n_head_tail_ratio_finite"] == 0
    assert stats["head_tail_ratio"]["median"] is None


def test_compute_summary_stats_excludes_nan_and_out_of_range():
    rows = [{"residual_ratio": v} for v in [0.1, float("nan"), 0.5, 1.5, -0.3, 0.7]]
    stats = compute_summary_stats(rows)
    assert stats["n_images"] == 6
    assert stats["n_residual_ratio_finite"] == 5  # nan excluded
    assert stats["n_residual_ratio_finite_in_range"] == 3  # 0.1, 0.5, 0.7


def test_compute_summary_stats_empty_input():
    stats = compute_summary_stats([])
    assert stats["n_images"] == 0
    assert stats["residual_ratio"]["median"] is None
    assert stats["residual_ratio"]["iqr"] is None


def test_compute_summary_stats_single_element_has_no_std():
    """``std`` with ``ddof=1`` is undefined for n=1 — must be ``None``."""
    stats = compute_summary_stats([{"residual_ratio": 0.42}])
    assert stats["residual_ratio"]["std"] is None
    assert stats["residual_ratio"]["median"] == 0.42


def test_write_summary_csv_emits_expected_columns(tmp_path: Path):
    rows = [_sample_row("img_001"), _sample_row("img_002", residual_ratio=0.7)]
    path = tmp_path / "summary.csv"
    n = write_summary_csv(rows, path)
    assert n == 2

    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        out_rows = list(reader)
    assert {r["image_id"] for r in out_rows} == {"img_001", "img_002"}
    assert "residual_ratio" in out_rows[0]
    # head_tail_ratio must be in summary.csv alongside residual_ratio.
    assert "head_tail_ratio" in out_rows[0]
    # Matrices must not appear in summary.csv (scalar-only)
    assert "kl" not in out_rows[0]
    assert "cos_dist" not in out_rows[0]


def test_write_summary_json_serialises_nan_as_null(tmp_path: Path):
    stats = compute_summary_stats(
        [{"residual_ratio": float("nan"), "model_id": "m", "prompt_key": "p"}]
    )
    path = tmp_path / "summary.json"
    write_summary_json(stats, path)

    raw = path.read_text(encoding="utf-8")
    parsed = json.loads(raw)
    assert parsed["n_images"] == 1
    assert parsed["residual_ratio"]["median"] is None


def test_write_summary_json_reports_range_metadata(tmp_path: Path):
    stats = compute_summary_stats([{"residual_ratio": 0.3}])
    path = tmp_path / "summary.json"
    write_summary_json(stats, path)
    parsed = json.loads(path.read_text(encoding="utf-8"))
    assert parsed["range"] == {"lo": 0.0, "hi": 1.0}


# ================================================================
# share_tail — post-Block-3 headline metric
# Round-trip through the parquet schema (populated + null cases) and
# summary stats wiring. Cousin tests of test_compute_summary_stats_*.
# ================================================================


def test_write_and_read_preserves_share_tail(tmp_path):
    """A populated share_tail value survives parquet round-trip."""
    from vr_modality_bias.io.results import read_metrics_table, write_metrics_table

    row = {
        "image_id": "img-1",
        "caption_len": 12,
        "n_layers": 28,
        "hidden_dim": 768,
        "caption_ref": "ref",
        "kl": [[0.1, 0.2, 0.3]],
        "cos_dist": [[0.4, 0.5, 0.6]],
        "residual_ratio": 0.42,
        "share_tail": 0.37,             # bounded [0,1], SPARC-proof
        "head_tail_ratio": 0.81,        # legacy column kept around
        "model_id": "llava-hf/llava-1.5-7b-hf",
        "prompt_key": "caption_long",
        "seed_global": 0,
        "noise_seed": 1,
        "timestamp_iso": "2026-01-01T00:00:00+00:00",
        "caption_tokens": None,
    }
    path = tmp_path / "out.parquet"
    write_metrics_table([row], path)
    back = read_metrics_table(path)
    assert len(back) == 1
    assert back[0]["share_tail"] == pytest.approx(0.37, rel=1e-5)


def test_write_accepts_missing_share_tail_as_null(tmp_path):
    """Rows produced by code paths predating Block 3 omit share_tail; must round-trip as None."""
    from vr_modality_bias.io.results import read_metrics_table, write_metrics_table

    row = {
        "image_id": "img-2",
        "caption_len": 12,
        "n_layers": 28,
        "hidden_dim": 768,
        "caption_ref": "ref",
        "kl": [[0.0]],
        "cos_dist": [[0.0]],
        "residual_ratio": 0.4,
        # share_tail intentionally absent
        "head_tail_ratio": 0.7,
        "model_id": "llava-hf/llava-1.5-7b-hf",
        "prompt_key": "caption_long",
        "seed_global": 0,
        "noise_seed": 1,
        "timestamp_iso": "2026-01-01T00:00:00+00:00",
        "caption_tokens": None,
    }
    path = tmp_path / "out.parquet"
    write_metrics_table([row], path)
    back = read_metrics_table(path)
    assert back[0]["share_tail"] is None


def test_compute_summary_stats_reports_share_tail_section():
    """share_tail must show up in the summary alongside the legacy htr block."""
    from vr_modality_bias.io.results import compute_summary_stats

    rows = [
        {"residual_ratio": 0.4, "share_tail": v, "head_tail_ratio": 1.0, "model_id": "m"}
        for v in (0.1, 0.3, 0.5, 0.7, 0.9)
    ]
    stats = compute_summary_stats(rows)
    assert stats["n_share_tail_finite"] == 5
    st = stats["share_tail"]
    assert math.isclose(st["median"], 0.5, rel_tol=1e-9)
    assert st["min"] == pytest.approx(0.1)
    assert st["max"] == pytest.approx(0.9)
