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
        # Per-token deep-block-mean KL, length == caption_len. Nullable so
        # parquets predating Block 4 stay readable. Stored explicitly (not
        # recomputed from ``kl[deep_block, :].mean(0)`` at read time)
        # because Fig 2 (attenuation curves per length) plots it directly
        # and we don't want every figure script to re-import deep_block.
        pa.field("deep_curve", pa.list_(pa.float32()), nullable=True),
        pa.field("residual_ratio", pa.float32()),
        # Length-invariant attenuation indicator (mean(tail) / mean(head) on
        # the deep-block KL curve). Complements residual_ratio, which
        # saturates near 1 for long captions even on flat curves. Nullable
        # so older parquets (baseline scripts/05) remain readable.
        # DEPRECATED post-Block-3: inflates under SPARC (the ratio is
        # unbounded above; SPARC's multiplicative amplification pushes it
        # to 50+/NaN). Use ``share_tail`` as the headline attenuation
        # metric — see metrics/residual.py. Kept for back-compat reads.
        pa.field("head_tail_ratio", pa.float32(), nullable=True),
        # Post-Block-3 headline attenuation metric: fraction of deep-KL
        # mass that sits in the tail half of the caption. Bounded [0, 1],
        # invariant under positive multiplicative scaling (SPARC-proof).
        # Nullable so parquets written before Block-3 stay readable.
        pa.field("share_tail", pa.float32(), nullable=True),
        pa.field("model_id", pa.string()),
        pa.field("prompt_key", pa.string()),
        pa.field("seed_global", pa.int32()),
        pa.field("noise_seed", pa.int64()),
        pa.field("timestamp_iso", pa.string()),
        pa.field("caption_tokens", pa.list_(pa.string()), nullable=True),
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
            elif field.name == "deep_curve":
                # Accept numpy arrays / lists / torch tensors. Coerce to
                # plain ``list[float]`` so pyarrow stores it as the
                # ``list<float32>`` schema field (kept nullable).
                if value is not None:
                    value = [float(v) for v in np.asarray(value, dtype=np.float32).tolist()]
            elif field.name == "caption_tokens":

                if value is None:
                    pass
                else:
                    value = [str(t) for t in value]
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
    "share_tail",      # post-Block-3 headline; bounded [0,1], SPARC-proof
    "head_tail_ratio",  # deprecated; kept in CSV until orphan scripts retire
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


def _finite_stats(values: np.ndarray) -> dict[str, Any]:
    """Median/IQR/min/max/mean/std over the finite subset of ``values``."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "median": None, "q25": None, "q75": None, "iqr": None,
            "min": None, "max": None, "mean": None, "std": None,
        }
    q25 = float(np.quantile(finite, 0.25))
    q75 = float(np.quantile(finite, 0.75))
    return {
        "median": float(np.median(finite)),
        "q25": q25,
        "q75": q75,
        "iqr": q75 - q25,
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite, ddof=1)) if finite.size > 1 else None,
    }


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

    # share_tail is bounded [0, 1] by construction; we report finite stats
    # without any extra filtering. This is the POST-BLOCK-3 headline metric.
    share_tail_values = np.asarray(
        [row.get("share_tail") for row in rows], dtype=np.float64
    )
    share_tail_stats = _finite_stats(share_tail_values)
    n_share_tail_finite = int(np.isfinite(share_tail_values).sum())

    # head_tail_ratio is DEPRECATED but still computed/written by the
    # legacy code paths; keep its summary section so old viewers don't
    # crash. ``None``s and missing keys are tolerated.
    htr_values = np.asarray(
        [row.get("head_tail_ratio") for row in rows], dtype=np.float64
    )
    htr_stats = _finite_stats(htr_values)
    n_htr_finite = int(np.isfinite(htr_values).sum())

    head_row = rows[0] if rows else {}

    return {
        "n_images": int(len(rows)),
        "n_residual_ratio_finite": int(finite_mask.sum()),
        "n_residual_ratio_finite_in_range": int(in_range_mask.sum()),
        "n_share_tail_finite": n_share_tail_finite,
        "n_head_tail_ratio_finite": n_htr_finite,
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
        "share_tail": share_tail_stats,
        "head_tail_ratio": htr_stats,
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
