#!/usr/bin/env python
"""Compute the three internal metrics for every paired (A, B) under the active run."""

from __future__ import annotations

import argparse
from pathlib import Path

from vr_modality_bias.io.results import write_metrics_table
from vr_modality_bias.io.storage import load_hidden_states
from vr_modality_bias.metrics.cosine import compute_cosine_distance_matrix
from vr_modality_bias.metrics.kl import compute_kl_matrix
from vr_modality_bias.metrics.residual import residual_drift_ratio
from vr_modality_bias.models.registry import build_model
from vr_modality_bias.utils.config import load_config
from vr_modality_bias.utils.device import resolve_dtype, select_device
from vr_modality_bias.utils.logging import configure_logging, get_logger
from vr_modality_bias.utils.runs import current_run_dir


def _discover_image_ids(hidden_states_dir: Path) -> list[str]:
    """Return image_ids that have BOTH ``*__A.h5`` and ``*__B.h5`` under ``dir``."""
    ids: dict[str, set[str]] = {}
    for path in hidden_states_dir.glob("*.h5"):
        stem = path.stem
        if "__" not in stem:
            continue
        image_id, condition = stem.rsplit("__", 1)
        if condition not in {"A", "B"}:
            continue
        ids.setdefault(image_id, set()).add(condition)
    complete = sorted(image_id for image_id, conds in ids.items() if conds == {"A", "B"})
    return complete


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_dir = current_run_dir(cfg["run"]["output_root"], cfg["run"]["name"])
    log_file = run_dir / "logs" / "05_compute_metrics.log"
    configure_logging(log_file=log_file)
    log = get_logger(__name__)
    log.info("Run dir: %s", run_dir)

    metrics_path = run_dir / "metrics.parquet"
    if metrics_path.exists() and not args.overwrite:
        log.info(
            "metrics.parquet already exists (%s) — pass --overwrite to regenerate.",
            metrics_path,
        )
        return 0

    hidden_states_dir = run_dir / "hidden_states"
    if not hidden_states_dir.is_dir():
        raise FileNotFoundError(
            f"{hidden_states_dir} missing — run scripts/04_collect_hidden_states.py first."
        )

    image_ids = _discover_image_ids(hidden_states_dir)
    if args.limit:
        image_ids = image_ids[: args.limit]
    if not image_ids:
        raise RuntimeError(
            f"No (image_id, A, B) pairs found under {hidden_states_dir}."
        )
    log.info("Found %d complete (A, B) pair(s).", len(image_ids))

    # The model is loaded purely to obtain ``lm_head`` for KL projection.
    model = build_model(cfg["model"]["key"])
    model.model_id = str(cfg["model"]["model_id"])
    dtype = resolve_dtype(str(cfg["model"]["dtype"]))
    device = select_device("cuda")
    log.info("Loading model %s on %s (dtype=%s)…", model.model_id, device, dtype)
    if hasattr(model, "_dtype"):
        model._dtype = dtype  # noqa: SLF001
    model.load(device)
    lm_head = model.get_lm_head()
    log.info("Loaded. n_layers=%d", model.n_layers)

    top_k = int(cfg["metrics"]["logits_top_k"])
    t0 = int(cfg["residual"]["t0"])

    rows: list[dict] = []
    for image_id in image_ids:
        result_A = load_hidden_states(hidden_states_dir / f"{image_id}__A.h5")
        result_B = load_hidden_states(hidden_states_dir / f"{image_id}__B.h5")

        if result_A.caption_start != result_B.caption_start:
            raise RuntimeError(
                f"[{image_id}] caption_start differs: "
                f"A={result_A.caption_start} B={result_B.caption_start}"
            )
        if result_A.caption_len != result_B.caption_len:
            raise RuntimeError(
                f"[{image_id}] caption_len differs: "
                f"A={result_A.caption_len} B={result_B.caption_len}"
            )

        kl = compute_kl_matrix(
            lm_head,
            result_A.hidden_states,
            result_B.hidden_states,
            caption_start=int(result_A.caption_start),
            caption_len=int(result_A.caption_len),
            top_k=top_k,
        )
        cos = compute_cosine_distance_matrix(
            result_A.hidden_states,
            result_B.hidden_states,
            caption_start=int(result_A.caption_start),
            caption_len=int(result_A.caption_len),
        )
        rr = residual_drift_ratio(kl, t0=t0)

        meta = result_A.metadata
        rows.append({
            "image_id": image_id,
            "caption_len": int(result_A.caption_len),
            "n_layers": int(result_A.hidden_states.shape[0]),
            "hidden_dim": int(result_A.hidden_states.shape[-1]),
            "caption_ref": str(meta.get("caption_ref", "")),
            "kl": kl,
            "cos_dist": cos,
            "residual_ratio": float(rr),
            "model_id": str(meta.get("model_id", model.model_id)),
            "prompt_key": str(meta.get("prompt_key", "")),
            "seed_global": int(meta.get("seed_global", 0)),
            "noise_seed": int(meta.get("noise_seed", 0)),
            "timestamp_iso": str(meta.get("timestamp_iso", "")),
        })
        log.info("[%s] residual_ratio=%.4f", image_id, rr)

    n = write_metrics_table(rows, metrics_path)
    log.info("Wrote %d row(s) to %s.", n, metrics_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
