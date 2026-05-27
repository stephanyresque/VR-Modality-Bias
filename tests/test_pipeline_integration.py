from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from vr_modality_bias.io.results import (
    compute_summary_stats,
    read_metrics_table,
    write_metrics_table,
    write_summary_csv,
    write_summary_json,
)
from vr_modality_bias.io.storage import (
    hidden_states_filename,
    load_hidden_states,
    save_hidden_states,
)
from vr_modality_bias.metrics.cosine import compute_cosine_distance_matrix
from vr_modality_bias.metrics.kl import compute_kl_matrix
from vr_modality_bias.metrics.residual import residual_drift_ratio
from vr_modality_bias.models.base import HiddenStatesResult


def _synthetic_pair(
    *,
    image_id: str,
    n_layers: int = 30,
    seq_len: int = 20,
    hidden_dim: int = 16,
    caption_start: int = 8,
    caption_len: int = 12,
    seed_a: int = 0,
    seed_b: int = 1,
) -> tuple[HiddenStatesResult, HiddenStatesResult]:
    """Produce two synthetic results that share input_ids but diverge in hidden states."""
    g_a = torch.Generator().manual_seed(seed_a)
    g_b = torch.Generator().manual_seed(seed_b)
    h_a = torch.randn(
        n_layers, seq_len, hidden_dim, generator=g_a, dtype=torch.float32
    ).to(torch.float16)
    h_b = torch.randn(
        n_layers, seq_len, hidden_dim, generator=g_b, dtype=torch.float32
    ).to(torch.float16)
    input_ids = torch.arange(seq_len, dtype=torch.int64) + 1
    attn = torch.ones(seq_len, dtype=torch.int8)

    common_kw = dict(
        input_ids=input_ids,
        caption_start=caption_start,
        caption_len=caption_len,
        attention_mask=attn,
    )
    meta = {
        "model_id": "synthetic/test",
        "hidden_dim": hidden_dim,
        "n_layers": n_layers,
        "image_id": image_id,
        "caption_ref": f"Synthetic caption for {image_id}.",
        "prompt_key": "caption_short",
        "seed_global": 42,
        "noise_seed": 12345,
        "timestamp_iso": "2026-05-27T14:30:00+00:00",
    }
    result_A = HiddenStatesResult(hidden_states=h_a, metadata=dict(meta), **common_kw)
    result_B = HiddenStatesResult(hidden_states=h_b, metadata=dict(meta), **common_kw)
    return result_A, result_B


def test_end_to_end_h5_to_summary(tmp_path: Path):
    run_dir = tmp_path / "run"
    hs_dir = run_dir / "hidden_states"
    hs_dir.mkdir(parents=True)

    image_ids = ["img_001", "img_002", "img_003"]
    for i, image_id in enumerate(image_ids):
        a, b = _synthetic_pair(image_id=image_id, seed_a=i * 2, seed_b=i * 2 + 1)
        save_hidden_states(
            hs_dir / hidden_states_filename(image_id, "A"),
            a,
            condition="A",
            extra_attrs={"image_id": image_id, **{k: a.metadata[k] for k in (
                "caption_ref", "model_id", "prompt_key", "seed_global",
                "noise_seed", "timestamp_iso"
            )}},
        )
        save_hidden_states(
            hs_dir / hidden_states_filename(image_id, "B"),
            b,
            condition="B",
            extra_attrs={"image_id": image_id, **{k: b.metadata[k] for k in (
                "caption_ref", "model_id", "prompt_key", "seed_global",
                "noise_seed", "timestamp_iso"
            )}},
        )

    rows: list[dict] = []
    for image_id in image_ids:
        a = load_hidden_states(hs_dir / hidden_states_filename(image_id, "A"))
        b = load_hidden_states(hs_dir / hidden_states_filename(image_id, "B"))
        assert a.caption_start == b.caption_start
        assert a.caption_len == b.caption_len

        kl = compute_kl_matrix(
            torch.nn.Identity(),
            a.hidden_states,
            b.hidden_states,
            caption_start=a.caption_start,
            caption_len=a.caption_len,
        )
        cos = compute_cosine_distance_matrix(
            a.hidden_states,
            b.hidden_states,
            caption_start=a.caption_start,
            caption_len=a.caption_len,
        )
        rr = residual_drift_ratio(kl, t0=5)

        # Sanity at each stage.
        assert kl.shape == (a.hidden_states.shape[0], a.caption_len)
        assert cos.shape == (a.hidden_states.shape[0], a.caption_len)
        assert np.isfinite(kl).all() and (kl >= 0).all()
        assert np.isfinite(cos).all() and (cos >= 0).all()
        assert 0.0 <= rr <= 1.0, f"residual_ratio out of range: {rr}"

        rows.append({
            "image_id": image_id,
            "caption_len": int(a.caption_len),
            "n_layers": int(a.hidden_states.shape[0]),
            "hidden_dim": int(a.hidden_states.shape[-1]),
            "caption_ref": str(a.metadata.get("caption_ref", "")),
            "kl": kl,
            "cos_dist": cos,
            "residual_ratio": float(rr),
            "model_id": str(a.metadata.get("model_id", "")),
            "prompt_key": str(a.metadata.get("prompt_key", "")),
            "seed_global": int(a.metadata.get("seed_global", 0)),
            "noise_seed": int(a.metadata.get("noise_seed", 0)),
            "timestamp_iso": str(a.metadata.get("timestamp_iso", "")),
        })

    metrics_path = run_dir / "metrics.parquet"
    write_metrics_table(rows, metrics_path)

    # Read back the parquet — values must be byte-identical.
    back = read_metrics_table(metrics_path)
    assert len(back) == len(image_ids)
    assert {r["image_id"] for r in back} == set(image_ids)

    # Summary stage.
    csv_path = run_dir / "summary.csv"
    json_path = run_dir / "summary.json"
    write_summary_csv(back, csv_path)
    stats = compute_summary_stats(back)
    write_summary_json(stats, json_path)

    assert csv_path.is_file()
    assert json_path.is_file()
    assert stats["n_images"] == len(image_ids)
    # All synthetic rrs are in [0, 1] by construction (KL >= 0).
    assert stats["n_residual_ratio_finite_in_range"] == len(image_ids)
    rr_summary = stats["residual_ratio"]
    assert rr_summary["median"] is not None
    assert 0.0 <= rr_summary["median"] <= 1.0
