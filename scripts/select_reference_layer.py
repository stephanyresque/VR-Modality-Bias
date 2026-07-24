#!/usr/bin/env python
"""Offline (Ponto 4): derive SPARC's reference layer from the diagnostic's
per-layer KL curve and print the recommended --selected-layer.

Run: python scripts/select_reference_layer.py --metrics-glob 'results/runs/sweep_*/metrics.parquet'
     (or: python scripts/select_reference_layer.py --json curve.json)
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from pyprojroot import here

try:
    from vr_modality_bias.metrics.reference_layer import reference_layer_from_curve
except ModuleNotFoundError:
    sys.path.insert(0, str(here()))

    from src.vr_modality_bias.metrics.reference_layer import reference_layer_from_curve


_JSON_HINT = "Pass the already-aggregated curve via --json instead."


def mean_over_positions(kl, source: str = "<parquet>") -> list[float]:
    """Per-layer mean over positions for one image row's ``kl`` cell.

    ``kl`` is the nested list stored by the diagnostic: ``kl[layer]`` is the
    per-position KL vector (length == caption_len, variable across images).
    A cell that is not a rectangular ``(n_layers, n_positions)`` matrix is an
    unexpected format, not something to guess at.
    """
    if kl is None or len(kl) == 0:
        raise ValueError(f"{source}: empty 'kl' cell. {_JSON_HINT}")
    try:
        matrix = np.asarray(kl, dtype=np.float64)
    except (ValueError, TypeError):
        raise ValueError(
            f"{source}: 'kl' cell is not a rectangular (n_layers, n_positions) "
            f"matrix. {_JSON_HINT}"
        )
    if matrix.ndim != 2:
        raise ValueError(
            f"{source}: 'kl' cell has {matrix.ndim} dim(s), expected 2 "
            f"(n_layers, n_positions). {_JSON_HINT}"
        )
    return [float(v) for v in matrix.mean(axis=1)]


def aggregate_curve_from_files(files) -> list[float]:
    """Aggregate the per-model KL curve from diagnostic parquet files.

    Per image (one parquet row): mean over positions, per layer. Across images
    (all rows in all files): median per layer. Every matched row is an image;
    files that mix different ``n_layers`` are rejected rather than merged.
    """
    files = [str(f) for f in files]
    if not files:
        raise ValueError("no parquet files matched the glob.")

    per_image: list[list[float]] = []
    for path in files:
        table = pq.read_table(path)
        if "kl" not in table.column_names:
            raise ValueError(
                f"{path}: no 'kl' column; this is not a diagnostic "
                f"metrics.parquet. {_JSON_HINT}"
            )
        for kl in table.column("kl").to_pylist():
            per_image.append(mean_over_positions(kl, source=path))

    if not per_image:
        raise ValueError("matched parquet files contain no rows.")

    lengths = {len(v) for v in per_image}
    if len(lengths) != 1:
        raise ValueError(
            f"inconsistent n_layers across images: {sorted(lengths)}. The glob "
            "likely mixes models; point it at a single model's runs."
        )

    stacked = np.asarray(per_image, dtype=np.float64)
    return [float(v) for v in np.median(stacked, axis=0)]


def load_curve_from_json(path: Path) -> list[float]:
    """Read an already-aggregated curve: a top-level JSON list of floats."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(
            f"{path}: expected a top-level JSON list of floats, got {type(data).__name__}."
        )
    try:
        return [float(x) for x in data]
    except (TypeError, ValueError):
        raise ValueError(f"{path}: JSON list has a non-numeric entry.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--metrics-glob", type=str, default=None,
        help="Glob of diagnostic metrics.parquet files (nested 'kl' column).")
    source.add_argument("--json", type=Path, default=None,
        help="A JSON file with the already-aggregated curve (list of floats).")
    parser.add_argument("--theta", type=float, default=0.5,
        help="Threshold on the normalized curve (default 0.5).")
    return parser


def _print_report(curve: list[float], theta: float, source_desc: str) -> None:
    result = reference_layer_from_curve(curve, theta=theta)
    norm = result.normalized_curve

    print("=" * 70)
    print("REFERENCE LAYER (Ponto 4)")
    print("=" * 70)
    print(f"  source   : {source_desc}")
    print(f"  n_layers : {len(curve)}")
    print(f"  theta    : {theta}")
    print("  curve (raw KL / normalized), * = recommended, ^ = argmax, | = deep-block start:")
    for i, (raw, nrm) in enumerate(zip(curve, norm)):
        marks = "".join([
            "*" if i == result.recommended_layer else " ",
            "^" if i == result.argmax_layer else " ",
            "|" if i == result.deep_block_start else " ",
        ])
        nrm_str = "nan" if nrm != nrm else f"{nrm:.4f}"
        print(f"    {marks} layer {i:>2}: {raw:>12.6f}  {nrm_str}")
    print("-" * 70)
    print(f"  recommended selected_layer : {result.recommended_layer} "
          f"(first layer with k_norm >= {theta})")
    print(f"  argmax (peak influence)    : {result.argmax_layer}")
    print(f"  deep-block start (2L/3)    : {result.deep_block_start}")
    print("=" * 70)


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.metrics_glob is not None:
            files = sorted(glob.glob(args.metrics_glob))
            curve = aggregate_curve_from_files(files)
            source_desc = f"--metrics-glob {args.metrics_glob!r} ({len(files)} file(s))"
        else:
            curve = load_curve_from_json(args.json)
            source_desc = f"--json {args.json}"
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    _print_report(curve, theta=args.theta, source_desc=source_desc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
