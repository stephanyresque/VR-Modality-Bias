#!/usr/bin/env python
"""Aggregate ``metrics.parquet`` into ``summary.csv`` and ``summary.json``.

Run: make baseline  (or: python scripts/summarize.py --config configs/baseline.yaml)
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from vr_modality_bias.io.results import (
    compute_summary_stats,
    read_metrics_table,
    write_summary_csv,
    write_summary_json,
)
from vr_modality_bias.utils.config import load_config
from vr_modality_bias.utils.logging import configure_logging, get_logger
from vr_modality_bias.utils.profiling import (
    dir_size_bytes,
    format_bytes,
    format_seconds,
    summarize_seconds,
)
from vr_modality_bias.utils.runs import current_run_dir


def _read_collect_diagnostics(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    entries: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def _build_diagnostics(entries: list[dict], run_dir: Path) -> dict:
    """Build the §7 Phase 6 diagnostics payload from per-pair entries."""
    seconds = [e["seconds"] for e in entries]
    vram = [e["vram_peak_bytes"] for e in entries]
    bytes_pair = [e["bytes_pair"] for e in entries]

    seconds_stats = summarize_seconds(seconds)
    payload: dict = {
        "n_pairs": len(entries),
        "time_total_seconds": float(seconds_stats["total"]),
        "time_median_seconds_per_pair": (
            float(seconds_stats["median"])
            if seconds_stats["median"] is not None
            else None
        ),
        "time_mean_seconds_per_pair": (
            float(seconds_stats["mean"])
            if seconds_stats["mean"] is not None
            else None
        ),
        "vram_peak_bytes": int(max(vram, default=0)),
        "disk_total_bytes_pairs": int(sum(bytes_pair)),
        "disk_median_bytes_per_pair": (
            float(statistics.median(bytes_pair)) if bytes_pair else None
        ),
        "disk_total_bytes_run": int(dir_size_bytes(run_dir)),
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()


    cfg = load_config(args.config)
    run_dir = current_run_dir(cfg["run"]["output_root"], cfg["run"]["name"])
    log_file = run_dir / "logs" / "summarize.log"
    configure_logging(log_file=log_file)
    log = get_logger(__name__)
    log.info("Run dir: %s", run_dir)

    metrics_path = run_dir / "metrics.parquet"
    if not metrics_path.is_file():
        raise FileNotFoundError(
            f"{metrics_path} missing — run scripts/compute_metrics.py first."
        )

    summary_csv = run_dir / "summary.csv"
    summary_json = run_dir / "summary.json"
    diagnostics_json = run_dir / "diagnostics.json"
    already_done = summary_csv.exists() or summary_json.exists() or diagnostics_json.exists()
    if already_done and not args.overwrite:
        log.info(
            "summary.csv / summary.json / diagnostics.json already present — "
            "pass --overwrite to regenerate."
        )
        return 0

    rows = read_metrics_table(metrics_path)
    if args.limit:
        rows = rows[: args.limit]
    log.info("Loaded %d metric row(s) from %s.", len(rows), metrics_path)

    n = write_summary_csv(rows, summary_csv)
    log.info("Wrote %d row(s) to %s.", n, summary_csv)

    stats = compute_summary_stats(rows)
    write_summary_json(stats, summary_json)
    log.info("Wrote %s.", summary_json)

    rr = stats["residual_ratio"]
    log.info(
        "residual_ratio summary — n_finite_in_range=%d/%d, "
        "median=%s, q25=%s, q75=%s, iqr=%s",
        stats["n_residual_ratio_finite_in_range"],
        stats["n_images"],
        rr.get("median"),
        rr.get("q25"),
        rr.get("q75"),
        rr.get("iqr"),
    )

    # §7 Phase 6 — cost diagnostics rollup.
    diagnostics_collect = run_dir / "diagnostics_collect.jsonl"
    entries = _read_collect_diagnostics(diagnostics_collect)
    if entries:
        diagnostics = _build_diagnostics(entries, run_dir)
        with diagnostics_json.open("w", encoding="utf-8") as fh:
            json.dump(diagnostics, fh, indent=2)
            fh.write("\n")
        log.info(
            "Diagnostics — pairs=%d, total=%s, median/pair=%s, VRAM peak=%s, "
            "disk pairs=%s (median/pair=%s), disk run=%s",
            diagnostics["n_pairs"],
            format_seconds(diagnostics["time_total_seconds"]),
            format_seconds(diagnostics["time_median_seconds_per_pair"]),
            format_bytes(diagnostics["vram_peak_bytes"]),
            format_bytes(diagnostics["disk_total_bytes_pairs"]),
            format_bytes(diagnostics["disk_median_bytes_per_pair"])
            if diagnostics["disk_median_bytes_per_pair"] is not None
            else "n/a",
            format_bytes(diagnostics["disk_total_bytes_run"]),
        )
    else:
        log.info(
            "No per-pair diagnostics at %s — cost rollup skipped (rerun scripts/04 to populate).",
            diagnostics_collect,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
