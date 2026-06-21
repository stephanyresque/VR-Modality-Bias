#!/usr/bin/env python
"""Per-image unit examples — one folder per image with separate artefacts.

For every image in ``<run_dir>/metrics.parquet`` (or just ``--image-id``),
writes the following files inside
``<run_dir>/plots/unit_examples/<image_id>/``::

    image.jpg              copy of the source image (skipped if not on disk)
    meta.txt               prompt, caption_ref, residual_ratio, metadata
    kl_heatmap.png         per-(layer, token) KL for this image
    kl_token_curve.png     deep-block-averaged KL over tokens

This way the paper-side organisation (image / prompt / KL / curve in
distinct artefacts) is mechanically separated, not jammed into a single
composite figure.

CPU-only, no model required — reuses ``plot_heatmap``/``plot_token_curve``
and reads data already in ``metrics.parquet``.

CLI:
    python scripts/08_unit_example.py --config configs/baseline.yaml
    python scripts/08_unit_example.py --config configs/baseline.yaml --image-id 000000001584
    python scripts/08_unit_example.py --config configs/baseline.yaml --overwrite
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np

from vr_modality_bias.data.prompts import get_prompt
from vr_modality_bias.experiment.plots import (
    average_token_curve,
    plot_heatmap,
    plot_token_curve,
)
from vr_modality_bias.io.results import read_metrics_table
from vr_modality_bias.utils.config import load_config
from vr_modality_bias.utils.logging import configure_logging, get_logger
from vr_modality_bias.utils.runs import area_root, current_run_dir, length_from_prompt_key


def _kl_matrix_from_row(row: dict) -> np.ndarray:
    """Reconstruct an ``(n_layers, caption_len)`` ``float32`` matrix from the
    nested-list representation stored in the Parquet column."""
    return np.asarray(row["kl"], dtype=np.float32)


def _find_row(rows: list[dict], image_id: str) -> dict:
    """Return the row whose ``image_id`` matches, or raise ``KeyError``.

    The error message lists every available ``image_id`` so the user can
    immediately see which ids are valid for this run.
    """
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


def _format_meta(row: dict, prompt: str) -> str:
    """Format a human-readable ``meta.txt`` payload for one image."""
    rr = row.get("residual_ratio")
    if isinstance(rr, float) and np.isfinite(rr):
        rr_str = f"{rr:.6f}"
    else:
        rr_str = str(rr)
    return (
        f"image_id: {row.get('image_id')}\n"
        f"caption_len: {row.get('caption_len')}\n"
        f"n_layers: {row.get('n_layers')}\n"
        f"hidden_dim: {row.get('hidden_dim')}\n"
        f"residual_ratio: {rr_str}\n"
        f"\n"
        f"model_id: {row.get('model_id')}\n"
        f"prompt_key: {row.get('prompt_key')}\n"
        f"seed_global: {row.get('seed_global')}\n"
        f"noise_seed: {row.get('noise_seed')}\n"
        f"timestamp_iso: {row.get('timestamp_iso')}\n"
        f"\n"
        f"Prompt:\n{prompt}\n"
        f"\n"
        f"caption_ref:\n{row.get('caption_ref')}\n"
    )


def emit_for_row(
    row: dict,
    *,
    out_dir: Path,
    images_dir: Path,
    prompt: str,
    overwrite: bool = False,
    log: logging.Logger | None = None,
) -> Path:
    """Materialise the four artefacts for one image into ``out_dir/<image_id>/``.

    Always writes ``meta.txt``, ``kl_heatmap.png`` and ``kl_token_curve.png``.
    Copies the source ``image.jpg`` only when it exists on disk; logs a
    warning otherwise. Returns the path of the per-image directory.
    """
    log = log or get_logger(__name__)
    image_id = str(row["image_id"])
    target_dir = out_dir / image_id
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. meta.txt
    meta_path = target_dir / "meta.txt"
    if overwrite or not meta_path.is_file():
        meta_path.write_text(_format_meta(row, prompt), encoding="utf-8")

    # 2. Source image (optional — copy only if available)
    src_image = images_dir / f"{image_id}.jpg"
    dst_image = target_dir / "image.jpg"
    if src_image.is_file():
        if overwrite or not dst_image.is_file():
            shutil.copyfile(src_image, dst_image)
    else:
        log.warning("[%s] source image not found at %s", image_id, src_image)

    # 3. KL heatmap (reuses the same style as the run-level plots)
    kl_matrix = _kl_matrix_from_row(row)
    heatmap_path = target_dir / "kl_heatmap.png"
    if overwrite or not heatmap_path.is_file():
        plot_heatmap(
            kl_matrix,
            path=heatmap_path,
            title=f"KL(B || A) per (layer, token) — {image_id}",
            cbar_label="KL divergence (nats)",
        )

    # 4. Deep-block KL curve
    curve_path = target_dir / "kl_token_curve.png"
    if overwrite or not curve_path.is_file():
        curve = average_token_curve(kl_matrix)
        plot_token_curve(
            curve,
            path=curve_path,
            title=f"Deep-block mean KL vs. token — {image_id}",
            y_label="Mean KL (deep block, last third)",
        )

    return target_dir


def _kl_shape_from_row(row: dict) -> list[int]:
    kl = np.asarray(row.get("kl") or [], dtype=np.float32)
    return list(kl.shape) if kl.ndim == 2 else []


def _cos_shape_from_row(row: dict) -> list[int]:
    cos = np.asarray(row.get("cos_dist") or [], dtype=np.float32)
    return list(cos.shape) if cos.ndim == 2 else []


def emit_unit_case_json(
    row: dict,
    *,
    out_dir: Path,
    images_dir: Path,
    metrics_parquet_path: Path,
    prompt: str,
    overwrite: bool = False,
) -> Path:
    """Write the Fig-3 data scaffold for one image: ``unit_cases/<image_id>.json``.

    This is the structured-data twin of the per-image plot folder. It
    references back into ``metrics.parquet`` for the heavy arrays
    (``kl``, ``cos_dist``) instead of duplicating them — figure code
    opens the parquet, picks the row, and combines with this JSON.

    Future-CHAIR slots (``hallucinated_objects``,
    ``hallucinated_token_positions``) are left empty; they'll be
    populated when CHAIR is run on the diagnostic captions.
    """
    image_id = str(row["image_id"])
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{image_id}.json"
    if target.is_file() and not overwrite:
        return target

    image_path = images_dir / f"{image_id}.jpg"
    payload = {
        "image_id": image_id,
        "image_path": str(image_path),
        "prompt": prompt,
        "caption_ref": row.get("caption_ref"),
        "caption_tokens": row.get("caption_tokens"),
        "caption_start": row.get("caption_start"),
        "caption_len": row.get("caption_len"),
        "share_tail": row.get("share_tail"),
        "residual_ratio": row.get("residual_ratio"),
        "head_tail_ratio": row.get("head_tail_ratio"),
        "metrics_parquet": str(metrics_parquet_path),
        "kl_shape": _kl_shape_from_row(row),
        "cos_dist_shape": _cos_shape_from_row(row),
        # Fig-3 alignment between hallucinated objects and token positions.
        # Populated by a future CHAIR-on-diagnostic pass; kept here as
        # empty lists so the JSON shape is stable across runs.
        "hallucinated_objects": [],
        "hallucinated_token_positions": [],
        "notes": (
            "hallucinated_* are populated by a future CHAIR-on-diagnostic "
            "pass; ``kl`` / ``cos_dist`` live in ``metrics_parquet`` to "
            "avoid duplicating large arrays."
        ),
    }
    # JSON-safe: replace numpy scalars / NaN with native types / None.
    def _coerce(v):
        if isinstance(v, np.generic):
            v = v.item()
        if isinstance(v, float) and not np.isfinite(v):
            return None
        return v
    payload = {k: _coerce(v) for k, v in payload.items()}
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--image-id",
        type=str,
        default=None,
        help="If set, render only this image_id. Otherwise iterate every row in metrics.parquet.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    organized_root = area_root(
        cfg["run"]["output_root"],
        area=str(cfg["run"].get("area", "diagnostico")),
        model_key=str(cfg["model"]["key"]),
        length=length_from_prompt_key(str(cfg["task"]["prompt_key"])),
    )
    run_dir = current_run_dir(organized_root, cfg["run"]["name"])
    log_file = run_dir / "logs" / "08_unit_example.log"
    configure_logging(log_file=log_file)
    log = get_logger(__name__)
    log.info("Run dir: %s", run_dir)

    metrics_path = run_dir / "metrics.parquet"
    if not metrics_path.is_file():
        raise FileNotFoundError(
            f"{metrics_path} missing — run scripts/05_compute_metrics.py first."
        )

    rows = read_metrics_table(metrics_path)
    if args.image_id:
        rows = [_find_row(rows, args.image_id)]
    log.info("Rendering unit examples for %d image(s).", len(rows))

    prompt_key = str(cfg["task"]["prompt_key"])
    prompt = get_prompt(prompt_key)

    images_dir = Path(cfg["dataset"]["images_dir"])
    out_dir = run_dir / "plots" / "unit_examples"
    # Fig-3 data goes next to (not inside) the plot folder so figure
    # scripts can find all unit cases at one stable path.
    unit_cases_dir = run_dir / "unit_cases"

    for row in rows:
        target = emit_for_row(
            row,
            out_dir=out_dir,
            images_dir=images_dir,
            prompt=prompt,
            overwrite=args.overwrite,
            log=log,
        )
        json_target = emit_unit_case_json(
            row,
            out_dir=unit_cases_dir,
            images_dir=images_dir,
            metrics_parquet_path=metrics_path,
            prompt=prompt,
            overwrite=args.overwrite,
        )
        rr = row.get("residual_ratio")
        log.info(
            "[%s] folder ready (rr=%s): plots=%s  data=%s",
            row.get("image_id"),
            f"{rr:.4f}" if isinstance(rr, float) and np.isfinite(rr) else rr,
            target,
            json_target,
        )

    log.info("Done. %d unit example folder(s) under %s; data under %s.",
             len(rows), out_dir, unit_cases_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
