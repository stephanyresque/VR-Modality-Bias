#!/usr/bin/env python
"""Render the four inspection plots (KL/cosine heatmaps and token curves) for the active run.

Run: make baseline  (or: python scripts/make_plots.py --config configs/baseline.yaml)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from vr_modality_bias.experiment.plots import (
    average_matrices,
    average_token_curve,
    plot_heatmap,
    plot_token_curve,
)
from vr_modality_bias.io.results import read_metrics_table
from vr_modality_bias.utils.config import load_config
from vr_modality_bias.utils.logging import configure_logging, get_logger
from vr_modality_bias.utils.runs import current_run_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_dir = current_run_dir(cfg["run"]["output_root"], cfg["run"]["name"])
    log_file = run_dir / "logs" / "make_plots.log"
    configure_logging(log_file=log_file)
    log = get_logger(__name__)
    log.info("Run dir: %s", run_dir)

    metrics_path = run_dir / "metrics.parquet"
    if not metrics_path.is_file():
        raise FileNotFoundError(
            f"{metrics_path} missing — run scripts/compute_metrics.py first."
        )

    plots_dir = run_dir / "plots"
    expected = [
        plots_dir / "kl_heatmap.png",
        plots_dir / "cos_heatmap.png",
        plots_dir / "kl_token_curve.png",
        plots_dir / "cos_token_curve.png",
    ]
    if all(p.exists() for p in expected) and not args.overwrite:
        log.info("All four plots already exist — pass --overwrite to regenerate.")
        return 0

    rows = read_metrics_table(metrics_path)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError(f"{metrics_path} is empty.")
    log.info("Loaded %d metric row(s).", len(rows))

    kl_mats = [np.asarray(row["kl"], dtype=np.float32) for row in rows]
    cos_mats = [np.asarray(row["cos_dist"], dtype=np.float32) for row in rows]

    mean_kl = average_matrices(kl_mats)
    mean_cos = average_matrices(cos_mats)

    plot_heatmap(
        mean_kl,
        path=plots_dir / "kl_heatmap.png",
        title=f"Mean KL(B || A) — {len(rows)} image(s)",
        cbar_label="KL divergence (nats)",
        cmap="viridis",
    )
    plot_heatmap(
        mean_cos,
        path=plots_dir / "cos_heatmap.png",
        title=f"Mean cosine distance — {len(rows)} image(s)",
        cbar_label="cos_dist",
        cmap="magma",
    )

    kl_curve = average_token_curve(mean_kl)
    cos_curve = average_token_curve(mean_cos)

    plot_token_curve(
        kl_curve,
        path=plots_dir / "kl_token_curve.png",
        title=f"Deep-block mean KL vs. token — {len(rows)} image(s)",
        y_label="Mean KL (deep block, last third)",
    )
    plot_token_curve(
        cos_curve,
        path=plots_dir / "cos_token_curve.png",
        title=f"Deep-block mean cosine distance vs. token — {len(rows)} image(s)",
        y_label="Mean cos_dist (deep block, last third)",
    )

    log.info("Wrote 4 plot(s) under %s.", plots_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
