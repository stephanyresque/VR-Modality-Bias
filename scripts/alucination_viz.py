#!/usr/bin/env python
"""Generate the individual visualisation pieces for one (run, image_id)."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np

from vr_modality_bias.experiment.plots import (
    average_token_curve,
    plot_heatmap,
    plot_token_curve,
)
from vr_modality_bias.io.results import read_metrics_table
from vr_modality_bias.utils.config import load_config
from vr_modality_bias.utils.logging import configure_logging, get_logger
from vr_modality_bias.utils.runs import current_run_dir

DEFAULT_IMAGE_ID = "000000001503"
DEFAULT_SMOOTH_WINDOW = 15


def _kl_matrix_from_row(row: dict) -> np.ndarray:
    """Reconstruct the ``(n_layers, caption_len)`` matrix from the nested list."""
    return np.asarray(row["kl"], dtype=np.float32)


def _find_row(rows: list[dict], image_id: str) -> dict:
    """Return the row whose ``image_id`` matches, or raise with a clear message."""
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


def _moving_average_padded(curve: np.ndarray, window: int) -> np.ndarray:
    
    arr = np.asarray(curve, dtype=np.float64)
    if window <= 1:
        return arr.copy()
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(arr, (pad_left, pad_right), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def _normalize(curve: np.ndarray) -> np.ndarray:
    """Min-max-normalise a curve into ``[0, 1]``. Constant curve → all zeros."""
    arr = np.asarray(curve, dtype=np.float64)
    if arr.size == 0:
        return arr.copy()
    kl_min = float(np.nanmin(arr))
    kl_max = float(np.nanmax(arr))
    span = kl_max - kl_min
    if not np.isfinite(span) or span <= 0.0:
        return np.zeros_like(arr)
    return (arr - kl_min) / span


def _write_token_kl(
    out_dir: Path,
    tokens: list[str],
    deep_curve: np.ndarray,
) -> tuple[Path, Path]:
    
    n = len(tokens)
    arr = np.asarray(deep_curve, dtype=np.float64)
    if arr.size != n:
        raise ValueError(
            f"tokens length ({n}) != deep_curve length ({arr.size})"
        )
    kl_norm = _normalize(arr)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "token_kl.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, quoting=csv.QUOTE_ALL)
        writer.writerow(["position", "token", "kl_deep", "kl_norm"])
        for i, tok in enumerate(tokens):
            writer.writerow(
                [i, tok, f"{float(arr[i]):.6f}", f"{float(kl_norm[i]):.6f}"]
            )

    json_payload = [
        {
            "position": i,
            "token": tokens[i],
            "kl_deep": float(arr[i]),
            "kl_norm": float(kl_norm[i]),
        }
        for i in range(n)
    ]
    json_path = out_dir / "token_kl.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(json_payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    return csv_path, json_path


def _log_summary(log, caption_ref: str, tokens: list[str], deep_curve: np.ndarray) -> None:
    """Log the caption plus the 5 lowest- and 5 highest-KL tokens for manual triage."""
    log.info("=" * 70)
    log.info("caption_ref:")
    for line in (caption_ref.splitlines() or [caption_ref]):
        log.info("  %s", line)
    log.info("=" * 70)

    arr = np.asarray(deep_curve, dtype=np.float64)
    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        log.info("All deep-curve values are non-finite; cannot rank tokens.")
        return
    order = np.argsort(np.where(finite_mask, arr, np.inf))
    low5 = order[:5]
    order_high = np.argsort(np.where(finite_mask, arr, -np.inf))[::-1]
    high5 = order_high[:5]

    log.info(
        "Top-5 LOWEST kl_deep "
        "(candidate hallucination — model nearly indifferent to image):"
    )
    for idx in low5:
        log.info(
            "  pos=%-4d kl=%.4f  token=%r",
            int(idx),
            float(arr[idx]),
            tokens[int(idx)],
        )
    log.info(
        "Top-5 HIGHEST kl_deep (strong visual anchoring — image really matters):"
    )
    for idx in high5:
        log.info(
            "  pos=%-4d kl=%.4f  token=%r",
            int(idx),
            float(arr[idx]),
            tokens[int(idx)],
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "Explicit run directory. If omitted, resolves to the most recent "
            "run for the run_name declared in --config via <run_name>_LATEST.txt."
        ),
    )
    parser.add_argument(
        "--image-id",
        type=str,
        default=DEFAULT_IMAGE_ID,
        help=f"image_id to render (default: {DEFAULT_IMAGE_ID}).",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=DEFAULT_SMOOTH_WINDOW,
        help=(
            f"Moving-average window for kl_token_curve_smooth.png "
            f"(default: {DEFAULT_SMOOTH_WINDOW}). 1 disables smoothing."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.run_dir is not None:
        run_dir = args.run_dir.resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"--run-dir does not exist: {run_dir}")
    else:
        run_dir = current_run_dir(cfg["run"]["output_root"], cfg["run"]["name"])

    log_file = run_dir / "logs" / "alucination_viz.log"
    configure_logging(log_file=log_file)
    log = get_logger(__name__)
    log.info("Run dir: %s", run_dir)
    log.info("image_id: %s", args.image_id)
    log.info("smooth_window: %d", args.smooth_window)

    metrics_path = run_dir / "metrics.parquet"
    if not metrics_path.is_file():
        raise FileNotFoundError(
            f"{metrics_path} missing — pick a run produced by scripts/09 or scripts/05."
        )

    rows = read_metrics_table(metrics_path)
    row = _find_row(rows, args.image_id)

    caption_tokens = row.get("caption_tokens")
    if not caption_tokens:
        raise RuntimeError(
            f"caption_tokens missing or empty for image_id={args.image_id!r}. "
            "This script needs a run produced by scripts/09 (the sweep "
            "pipeline) — the baseline scripts/05 does not populate the column."
        )
    caption_tokens = list(caption_tokens)

    kl_matrix = _kl_matrix_from_row(row)
    caption_ref = str(row.get("caption_ref", ""))
    caption_len = int(row.get("caption_len", kl_matrix.shape[1]))
    n_layers = int(row.get("n_layers", kl_matrix.shape[0]))

    if len(caption_tokens) != caption_len:
        log.warning(
            "caption_tokens length (%d) != caption_len (%d) — using "
            "caption_tokens length for per-token export.",
            len(caption_tokens),
            caption_len,
        )

    deep_curve = average_token_curve(kl_matrix)
    smoothed = _moving_average_padded(deep_curve, window=args.smooth_window)
    log.info(
        "caption_len=%d, n_layers=%d, kl_deep range=[%.4f, %.4f]",
        caption_len,
        n_layers,
        float(np.nanmin(deep_curve)),
        float(np.nanmax(deep_curve)),
    )

    out_dir = run_dir / "viz" / args.image_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Piece 1 — moving-average-smoothed KL token curve. The raw curve is
    # dropped because the smoothed version reads clearly in the paper; the
    # smoothing window is still configurable via --smooth-window.
    curve_path = out_dir / "kl_token_curve.png"
    if args.overwrite or not curve_path.is_file():
        plot_token_curve(
            smoothed,
            path=curve_path,
            title=f"Deep-block mean KL vs. token — {args.image_id}",
            y_label="Mean KL (deep block, last third)",
        )

    # Piece 2 — token_kl.csv + token_kl.json
    csv_path, json_path = _write_token_kl(out_dir, caption_tokens, deep_curve)

    # Piece 3 — single-image heatmap
    heatmap_path = out_dir / "kl_heatmap.png"
    if args.overwrite or not heatmap_path.is_file():
        plot_heatmap(
            kl_matrix,
            path=heatmap_path,
            title=f"KL(B || A) per (layer, token) — {args.image_id}",
            cbar_label="KL divergence (nats)",
        )

    # Piece 4 — copy of the source image (optional)
    images_dir = Path(cfg["dataset"]["images_dir"])
    src_image = images_dir / f"{args.image_id}.jpg"
    dst_image = out_dir / "input_image.jpg"
    if src_image.is_file():
        if args.overwrite or not dst_image.is_file():
            shutil.copyfile(src_image, dst_image)
        log.info("Source image: %s", dst_image)
    else:
        log.warning(
            "Source image not found at %s — input_image.jpg not generated.",
            src_image,
        )

    log.info("Pieces under %s:", out_dir)
    for path in (curve_path, heatmap_path, csv_path, json_path):
        log.info("  %s", path.name)

    _log_summary(log, caption_ref, caption_tokens, deep_curve)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
