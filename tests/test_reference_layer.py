"""Tests for Ponto 4: the reference-layer rule (metrics.reference_layer) and the
curve aggregation of the offline CLI (scripts/select_reference_layer.py).
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from loguru import logger

from vr_modality_bias.metrics.reference_layer import (
    ReferenceLayerResult,
    reference_layer_from_curve,
)

_SCRIPTS = Path(__file__).parent.parent / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"_script_{name}", _SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def select_script():
    return _load_script("select_reference_layer")


def _write_parquet(path: Path, kl_rows: list, image_ids: list | None = None) -> None:
    """Write a synthetic diagnostic parquet with the nested ``kl`` column.

    ``kl_rows`` is one matrix per image row: ``kl_rows[img][layer]`` is that
    layer's per-position vector (variable length across images).
    """
    cols: dict = {"kl": kl_rows}
    if image_ids is not None:
        cols["image_id"] = image_ids
    pq.write_table(pa.table(cols), path)


# ---------------------------------------------------------------- the rule


def test_monotonic_increasing_curve():
    result = reference_layer_from_curve([1, 2, 3, 4, 5, 6, 7, 8], theta=0.5)
    assert isinstance(result, ReferenceLayerResult)
    assert result.recommended_layer == 3  # 4/8 = 0.5, the first crossing
    assert result.argmax_layer == 7
    assert result.deep_block_start == 5  # floor(2*8/3)
    assert result.normalized_curve[3] == pytest.approx(0.5)


def test_single_peak_in_the_middle():
    result = reference_layer_from_curve([0, 1, 2, 3, 4, 3, 2, 1, 0], theta=0.5)
    assert result.recommended_layer == 2  # 2/4 = 0.5
    assert result.argmax_layer == 4
    assert result.deep_block_start == 6  # floor(2*9/3)


def test_plateau_tie_at_threshold_returns_the_first_layer():
    # Normalized: [0.25, 0.5, 0.5, 0.5, 1.0]; the tie at 0.5 resolves to layer 1.
    result = reference_layer_from_curve([1, 2, 2, 2, 4], theta=0.5)
    assert result.recommended_layer == 1
    assert result.argmax_layer == 4


def test_flat_curve_recommends_the_first_layer():
    result = reference_layer_from_curve([5, 5, 5, 5], theta=0.5)
    assert result.recommended_layer == 0
    assert result.normalized_curve == [1.0, 1.0, 1.0, 1.0]


def test_sparse_nan_is_ignored_and_warns():
    curve = [1.0, float("nan"), 6.0, 8.0, float("nan"), 2.0]
    messages: list = []
    handler_id = logger.add(messages.append, level="WARNING")
    try:
        result = reference_layer_from_curve(curve, theta=0.5)
    finally:
        logger.remove(handler_id)

    # Finite max is 8 at layer 3; normalized [0.125, nan, 0.75, 1.0, nan, 0.25].
    assert result.recommended_layer == 2  # 0.75 crosses before the peak
    assert result.argmax_layer == 3
    assert math.isnan(result.normalized_curve[1])
    assert math.isnan(result.normalized_curve[4])
    assert result.normalized_curve[2] == pytest.approx(0.75)
    assert any("non-finite" in str(m) for m in messages)


def test_smolvlm_like_profile_recommends_before_the_deep_block():
    # 24 layers: low early, rising through the middle, peak in the last third.
    curve = [
        0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.14,
        0.19, 0.25, 0.32, 0.40, 0.49, 0.58, 0.68, 0.78,
        0.87, 0.94, 0.98, 1.00, 0.99, 0.95, 0.88, 0.80,
    ]
    result = reference_layer_from_curve(curve, theta=0.5)
    assert len(curve) == 24
    assert result.deep_block_start == 16  # floor(2*24/3)
    assert result.argmax_layer == 19
    assert result.recommended_layer == 13  # 0.49 at 12 (<0.5), 0.58 at 13
    assert result.recommended_layer < result.deep_block_start


def test_theta_one_is_accepted_and_selects_the_argmax():
    result = reference_layer_from_curve([1, 2, 4, 2], theta=1.0)
    assert result.recommended_layer == 2  # only the max normalizes to 1.0
    assert result.argmax_layer == 2


def test_accepts_a_numpy_array():
    result = reference_layer_from_curve(np.array([1.0, 2.0, 4.0, 8.0]), theta=0.5)
    assert result.recommended_layer == 2  # 4/8 = 0.5


# ---------------------------------------------------------------- guards


def test_empty_curve_raises():
    with pytest.raises(ValueError, match="empty"):
        reference_layer_from_curve([])


def test_all_zero_curve_raises():
    with pytest.raises(ValueError, match="zero"):
        reference_layer_from_curve([0.0, 0.0, 0.0])


def test_no_finite_values_raises():
    with pytest.raises(ValueError, match="finite"):
        reference_layer_from_curve([float("nan"), float("inf")])


@pytest.mark.parametrize("theta", [0.0, -0.1, 1.5, 2.0])
def test_theta_outside_the_unit_interval_raises(theta):
    with pytest.raises(ValueError, match="theta"):
        reference_layer_from_curve([1.0, 2.0, 3.0], theta=theta)


# ---------------------------------------------------------------- aggregation


def test_aggregate_curve_is_median_over_images_of_per_layer_means(select_script, tmp_path):
    path = tmp_path / "metrics.parquet"
    # Variable position length across images (the real caption_len variation).
    img1 = [[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]]  # per-layer mean [3, 7, 11]
    img2 = [[0.0, 3.0, 6.0], [3.0, 6.0, 9.0], [6.0, 9.0, 12.0]]  # [3, 6, 9]
    _write_parquet(path, [img1, img2], image_ids=["a", "b"])

    curve = select_script.aggregate_curve_from_files([str(path)])
    # Median per layer over the two images: [median(3,3), median(7,6), median(11,9)].
    assert curve == pytest.approx([3.0, 6.5, 10.0])


def test_aggregate_curve_pools_rows_across_files(select_script, tmp_path):
    path1 = tmp_path / "a.parquet"
    path2 = tmp_path / "b.parquet"
    _write_parquet(path1, [[[2.0], [4.0], [6.0]]])  # image mean [2, 4, 6]
    _write_parquet(path2, [[[4.0], [8.0], [10.0]]])  # image mean [4, 8, 10]
    curve = select_script.aggregate_curve_from_files([str(path1), str(path2)])
    assert curve == pytest.approx([3.0, 6.0, 8.0])


def test_aggregate_feeds_the_reference_rule(select_script, tmp_path):
    path = tmp_path / "metrics.parquet"
    img1 = [[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]]
    img2 = [[0.0, 3.0, 6.0], [3.0, 6.0, 9.0], [6.0, 9.0, 12.0]]
    _write_parquet(path, [img1, img2])
    curve = select_script.aggregate_curve_from_files([str(path)])
    result = reference_layer_from_curve(curve, theta=0.5)
    # curve [3, 6.5, 10]; normalized [0.3, 0.65, 1.0]; first >= 0.5 is layer 1.
    assert result.recommended_layer == 1
    assert result.argmax_layer == 2
    assert result.deep_block_start == 2


def test_aggregate_rejects_a_missing_kl_column(select_script, tmp_path):
    path = tmp_path / "bad.parquet"
    pq.write_table(pa.table({"not_kl": [1, 2, 3]}), path)
    with pytest.raises(ValueError, match="json"):
        select_script.aggregate_curve_from_files([str(path)])


def test_aggregate_rejects_inconsistent_n_layers(select_script, tmp_path):
    path1 = tmp_path / "a.parquet"
    path2 = tmp_path / "b.parquet"
    _write_parquet(path1, [[[1.0, 2.0], [3.0, 4.0]]])  # 2 layers
    _write_parquet(path2, [[[1.0], [2.0], [3.0]]])  # 3 layers
    with pytest.raises(ValueError, match="n_layers"):
        select_script.aggregate_curve_from_files([str(path1), str(path2)])


def test_aggregate_rejects_no_files(select_script):
    with pytest.raises(ValueError, match="no parquet files"):
        select_script.aggregate_curve_from_files([])


def test_mean_over_positions_rejects_a_ragged_cell(select_script):
    with pytest.raises(ValueError, match="json"):
        select_script.mean_over_positions([[1.0, 2.0], [3.0]])


# ---------------------------------------------------------------- json + parser


def test_load_curve_from_json(select_script, tmp_path):
    path = tmp_path / "curve.json"
    path.write_text(json.dumps([0.1, 0.5, 1.0]), encoding="utf-8")
    assert select_script.load_curve_from_json(path) == [0.1, 0.5, 1.0]


def test_load_curve_from_json_rejects_a_non_list(select_script, tmp_path):
    path = tmp_path / "curve.json"
    path.write_text(json.dumps({"curve": [1, 2]}), encoding="utf-8")
    with pytest.raises(ValueError, match="list"):
        select_script.load_curve_from_json(path)


def test_parser_requires_exactly_one_source(select_script):
    with pytest.raises(SystemExit):
        select_script.build_parser().parse_args([])
    with pytest.raises(SystemExit):
        select_script.build_parser().parse_args(["--metrics-glob", "x", "--json", "y"])


def test_parser_theta_default_is_half(select_script):
    args = select_script.build_parser().parse_args(["--json", "curve.json"])
    assert args.theta == 0.5
