"""Per-run result writers — ``metrics.parquet``, ``summary.csv``, ``summary.json``"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

__all__ = [
    "METRICS_SCHEMA",
    "compute_summary_stats",
    "read_metrics_table",
    "write_metrics_table",
    "write_summary_csv",
    "write_summary_json",
]


METRICS_SCHEMA = pa.schema(
    [
        pa.field("image_id", pa.string()),
        pa.field("caption_len", pa.int32()),
        pa.field("n_layers", pa.int32()),
        pa.field("hidden_dim", pa.int32()),
        pa.field("caption_ref", pa.string()),
        pa.field("kl", pa.list_(pa.list_(pa.float32()))),
        pa.field("cos_dist", pa.list_(pa.list_(pa.float32()))),
        pa.field("residual_ratio", pa.float32()),
        pa.field("model_id", pa.string()),
        pa.field("prompt_key", pa.string()),
        pa.field("seed_global", pa.int32()),
        pa.field("noise_seed", pa.int64()),
        pa.field("timestamp_iso", pa.string()),
    ]
)


def _matrix_to_nested_list(arr: np.ndarray | None) -> list[list[float]]:
    if arr is None:
        return []
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim != 2:
        raise ValueError(f"expected 2-D matrix, got shape {a.shape}")
    return a.tolist()


def write_metrics_table(rows: Iterable[dict[str, Any]], path: Path) -> int:
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = list(rows)
    columns: dict[str, list[Any]] = {field.name: [] for field in METRICS_SCHEMA}
    for row in rows:
        for field in METRICS_SCHEMA:
            value = row.get(field.name)
            if field.name in ("kl", "cos_dist"):
                value = _matrix_to_nested_list(value) if value is not None else []
            columns[field.name].append(value)

    table = pa.Table.from_pydict(columns, schema=METRICS_SCHEMA)
    pq.write_table(table, path)
    return len(rows)


def read_metrics_table(path: Path) -> list[dict[str, Any]]:
    """Read a Parquet file written by :func:`write_metrics_table` into row dicts."""
    return pq.read_table(Path(path)).to_pylist()


_SUMMARY_CSV_COLUMNS: tuple[str, ...] = (
    "image_id",
    "caption_len",
    "n_layers",
    "residual_ratio",
    "model_id",
    "prompt_key",
    "caption_ref",
)


def write_summary_csv(rows: Iterable[dict[str, Any]], path: Path) -> int:
    """Write one CSV row per image, scalar columns only."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(_SUMMARY_CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in _SUMMARY_CSV_COLUMNS})
            n += 1
    return n


def compute_summary_stats(
    rows: Iterable[dict[str, Any]],
    *,
    range_lo: float = 0.0,
    range_hi: float = 1.0,
) -> dict[str, Any]:

    rows = list(rows)
    values = np.asarray(
        [row.get("residual_ratio") for row in rows], dtype=np.float64
    )

    finite_mask = np.isfinite(values)
    in_range_mask = finite_mask & (values >= range_lo) & (values <= range_hi)
    eligible = values[in_range_mask]

    def _safe(stat_fn, default=None):
        if eligible.size == 0:
            return default
        return float(stat_fn(eligible))

    q25 = _safe(lambda a: np.quantile(a, 0.25))
    q75 = _safe(lambda a: np.quantile(a, 0.75))
    median = _safe(np.median)
    iqr = (q75 - q25) if (q25 is not None and q75 is not None) else None

    head_row = rows[0] if rows else {}

    return {
        "n_images": int(len(rows)),
        "n_residual_ratio_finite": int(finite_mask.sum()),
        "n_residual_ratio_finite_in_range": int(in_range_mask.sum()),
        "range": {"lo": float(range_lo), "hi": float(range_hi)},
        "residual_ratio": {
            "median": median,
            "q25": q25,
            "q75": q75,
            "iqr": iqr,
            "min": _safe(np.min),
            "max": _safe(np.max),
            "mean": _safe(np.mean),
            "std": _safe(lambda a: np.std(a, ddof=1)) if eligible.size > 1 else None,
        },
        "model_id": head_row.get("model_id"),
        "prompt_key": head_row.get("prompt_key"),
        "seed_global": head_row.get("seed_global"),
    }


def write_summary_json(summary: dict[str, Any], path: Path) -> Path:
    """Persist ``summary`` to ``path`` (UTF-8, 2-space indented)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(o: Any) -> Any:
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, float) and not math.isfinite(o):
            return None
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serialisable")

    with path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False, default=_default)
        fh.write("\n")
    return path
