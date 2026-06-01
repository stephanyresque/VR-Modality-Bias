#!/usr/bin/env python
"""Single-image illustration of what the run-level mean plots aggregate"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from vr_modality_bias.data.prompts import get_prompt
from vr_modality_bias.experiment.plots import (
    average_token_curve,
    plot_unit_example_panel,
)
from vr_modality_bias.io.results import read_metrics_table
from vr_modality_bias.utils.config import load_config
from vr_modality_bias.utils.logging import configure_logging, get_logger
from vr_modality_bias.utils.runs import current_run_dir

DEFAULT_IMAGE_ID = "000000001584"


def _kl_matrix_from_row(row: dict) -> np.ndarray:
    return np.asarray(row["kl"], dtype=np.float32)


def _find_row(rows: list[dict], image_id: str) -> dict:
    for row in rows:
        if row.get("image_id") == image_id:
            return row
    available = sorted(
        {r.get("image_id") for r in rows if r.get("image_id") is not None}
    )
    raise KeyError(
        f"image_id={image_id!r} not found in metrics.parquet. "
        f"Available ({len(available)}): {available}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--image-id",
        type=str,
        default=DEFAULT_IMAGE_ID,
        help=f"image_id to illustrate (default: {DEFAULT_IMAGE_ID}).",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_dir = current_run_dir(cfg["run"]["output_root"], cfg["run"]["name"])
    log_file = run_dir / "logs" / "08_unit_example.log"
    configure_logging(log_file=log_file)
    log = get_logger(__name__)
    log.info("Run dir: %s", run_dir)

    metrics_path = run_dir / "metrics.parquet"
    if not metrics_path.is_file():
        raise FileNotFoundError(
            f"{metrics_path} missing — run scripts/05_compute_metrics.py first."
        )

    plots_dir = run_dir / "plots"
    out_path = plots_dir / f"unit_example_{args.image_id}.png"
    if out_path.exists() and not args.overwrite:
        log.info("%s already exists — pass --overwrite to regenerate.", out_path)
        return 0

    rows = read_metrics_table(metrics_path)
    row = _find_row(rows, args.image_id)
    kl_matrix = _kl_matrix_from_row(row)
    deep_curve = average_token_curve(kl_matrix)

    caption_ref = str(row.get("caption_ref", ""))
    residual_ratio = float(row.get("residual_ratio", float("nan")))
    prompt_key = str(cfg["task"]["prompt_key"])
    prompt = get_prompt(prompt_key)

    # Try to load the source image; rendering remains valid without it.
    images_dir = Path(cfg["dataset"]["images_dir"])
    image_path = images_dir / f"{args.image_id}.jpg"
    image_array: np.ndarray | None = None
    if image_path.is_file():
        with Image.open(image_path) as raw:
            image_array = np.asarray(raw.convert("RGB"))
    else:
        log.warning(
            "Source image not found at %s — rendering panel with placeholder.",
            image_path,
        )

    plot_unit_example_panel(
        path=out_path,
        image_id=args.image_id,
        kl_matrix=kl_matrix,
        deep_curve=deep_curve,
        prompt=prompt,
        caption_ref=caption_ref,
        residual_ratio=residual_ratio,
        image_array=image_array,
    )
    log.info(
        "[%s] panel saved: caption_len=%d, residual_ratio=%.4f, path=%s",
        args.image_id,
        kl_matrix.shape[1],
        residual_ratio,
        out_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
