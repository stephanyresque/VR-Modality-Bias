#!/usr/bin/env python
"""Phase 2 sweep — analysis report, stdout only.

Run: python scripts/phase2_report.py --run-dir results/runs/phase2_alpha_sweep [--no-fluency]
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from pyprojroot import here

try:
    from vr_modality_bias.metrics.residual import deep_block
except ModuleNotFoundError:
    sys.path.insert(0, str(here()))
    from src.vr_modality_bias.metrics.residual import deep_block


LENGTHS_ORDER = ("short", "medium", "long")
DEFAULT_ALPHAS = (1.1, 1.2, 1.3, 1.4, 1.5)

START_COLLAPSE_THRESHOLD = 1e-9


def _load_parquet(run_dir: Path, length: str) -> list[dict]:
    path = run_dir / length / "metrics_sweep.parquet"
    if not path.exists():
        print(f"  WARN: {path} not found, skipping {length}.", file=sys.stderr)
        return []
    return pq.read_table(path).to_pylist()


def _augment_with_decomposition(row: dict, t0: int) -> None:
    """Recompute start/end of the deep KL curve in place (keys `_start`, `_end`).

    These are independent of ``head_tail_ratio`` stored in the parquet so we
    can diagnose WHERE the htr broke when it's NaN.
    """
    kl_raw = row.get("kl") or []
    try:
        kl = np.asarray(kl_raw, dtype=np.float64)
    except (TypeError, ValueError):
        row["_start"] = float("nan")
        row["_end"] = float("nan")
        return
    if kl.ndim != 2 or kl.size == 0:
        row["_start"] = float("nan")
        row["_end"] = float("nan")
        return
    n_layers = kl.shape[0]
    l0, l1 = deep_block(n_layers)
    deep_curve = kl[l0:l1, :].mean(axis=0)
    if deep_curve.size <= t0:
        row["_start"] = float("nan")
        row["_end"] = float("nan")
        return
    row["_start"] = float(deep_curve[:t0].mean())
    row["_end"] = float(deep_curve[t0:].mean())


def _is_finite(v) -> bool:
    if v is None:
        return False
    try:
        return bool(np.isfinite(v))
    except (TypeError, ValueError):
        return False


def _median_iqr(values) -> tuple[float, float, float, int]:
    arr = np.asarray([v for v in values if _is_finite(v)], dtype=np.float64)
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"), 0)
    return (
        float(np.median(arr)),
        float(np.percentile(arr, 25)),
        float(np.percentile(arr, 75)),
        int(arr.size),
    )


def _condition_key(condition: str, alpha) -> tuple[int, float]:
    """Sort key: OFF first (0), then ON (1) by alpha ascending."""
    return (0 if condition == "off" else 1, float(alpha) if alpha is not None else 0.0)


def _print_table(headers: list[str], rows: list[list], aligns: list[str] | None = None) -> None:
    if not rows:
        print("  (no rows)")
        return
    if aligns is None:
        aligns = ["<"] * len(headers)
    str_rows = [[str(c) for c in r] for r in rows]
    widths = [
        max(len(h), *(len(r[i]) for r in str_rows))
        for i, h in enumerate(headers)
    ]
    sep = "  "
    print("  " + sep.join(f"{h:<{w}}" for h, w in zip(headers, widths)))
    print("  " + sep.join("-" * w for w in widths))
    for r in str_rows:
        print("  " + sep.join(f"{c:{a}{w}}" for c, w, a in zip(r, widths, aligns)))


def _section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def report_cell_health(by_length: dict[str, list[dict]]) -> None:
    _section("1. CELL HEALTH — n_valid, n_NaN, cause of NaN")
    print("  collapsed := start (= mean(deep[:t0])) is finite but < {:.0e}".format(START_COLLAPSE_THRESHOLD))
    print("  other     := start is nan/inf, or other reason htr came out NaN")
    print()
    headers = ["length", "condition", "alpha", "n_total", "n_valid", "n_NaN", "%NaN", "n_collapsed", "n_other"]
    aligns =  ["<",      "<",         "<",     ">",       ">",       ">",     ">",    ">",           ">"]
    rows_out = []
    for length in LENGTHS_ORDER:
        rows = by_length.get(length) or []
        if not rows:
            continue
        groups: dict[tuple, list[dict]] = {}
        for r in rows:
            key = (r["condition"], r.get("alpha"))
            groups.setdefault(key, []).append(r)
        for key in sorted(groups.keys(), key=lambda k: _condition_key(*k)):
            cond, alpha = key
            cells = groups[key]
            n_total = len(cells)
            n_valid = sum(1 for c in cells if _is_finite(c.get("head_tail_ratio")))
            n_nan = n_total - n_valid
            n_collapsed = sum(
                1 for c in cells
                if not _is_finite(c.get("head_tail_ratio"))
                and _is_finite(c.get("_start"))
                and float(c["_start"]) < START_COLLAPSE_THRESHOLD
            )
            n_other = n_nan - n_collapsed
            rows_out.append([
                length, cond,
                f"{alpha:.1f}" if alpha is not None else "-",
                n_total, n_valid, n_nan,
                f"{100 * n_nan / n_total:.0f}%" if n_total else "-",
                n_collapsed, n_other,
            ])
    _print_table(headers, rows_out, aligns)


def report_delta_htr(by_length: dict[str, list[dict]]) -> None:
    _section("2. PAIRED Δhtr (htr_ON − htr_OFF) per image")
    print("  Δhtr is computed only for images where BOTH OFF and ON are valid (finite htr).")
    print("  n_paired tells you how many images contributed at each α.")
    print()
    headers = ["length", "alpha", "n_paired", "median", "q1", "q3"]
    aligns =  ["<",      "<",     ">",        ">",      ">",  ">"]
    rows_out = []
    for length in LENGTHS_ORDER:
        rows = by_length.get(length) or []
        if not rows:
            continue
        off_htr: dict[str, float] = {
            r["image_id"]: r["head_tail_ratio"]
            for r in rows
            if r["condition"] == "off" and _is_finite(r.get("head_tail_ratio"))
        }
        alphas_seen = sorted({
            r["alpha"] for r in rows
            if r["condition"] == "on" and r.get("alpha") is not None
        })
        for alpha in alphas_seen:
            deltas: list[float] = []
            for r in rows:
                if r["condition"] != "on" or r.get("alpha") != alpha:
                    continue
                htr_on = r.get("head_tail_ratio")
                if not _is_finite(htr_on):
                    continue
                htr_off = off_htr.get(r["image_id"])
                if htr_off is None:
                    continue
                deltas.append(float(htr_on) - float(htr_off))
            med, q1, q3, n = _median_iqr(deltas)
            rows_out.append([
                length, f"{alpha:.1f}", n,
                f"{med:+.4f}" if _is_finite(med) else "nan",
                f"{q1:+.4f}" if _is_finite(q1) else "nan",
                f"{q3:+.4f}" if _is_finite(q3) else "nan",
            ])
    _print_table(headers, rows_out, aligns)


def report_start_end(by_length: dict[str, list[dict]]) -> None:
    _section("3. START vs END decomposition — median across images")
    print("  start = mean(deep[:t0])   (the htr DENOMINATOR — collapses → htr blows up)")
    print("  end   = mean(deep[t0:])   (the htr NUMERATOR — grows → A/B diverges more)")
    print()
    headers = ["length", "condition", "alpha", "n", "med_start", "med_end"]
    aligns =  ["<",      "<",         "<",     ">", ">",         ">"]
    rows_out = []
    for length in LENGTHS_ORDER:
        rows = by_length.get(length) or []
        if not rows:
            continue
        groups: dict[tuple, list[dict]] = {}
        for r in rows:
            if not (_is_finite(r.get("_start")) and _is_finite(r.get("_end"))):
                continue
            key = (r["condition"], r.get("alpha"))
            groups.setdefault(key, []).append(r)
        for key in sorted(groups.keys(), key=lambda k: _condition_key(*k)):
            cond, alpha = key
            cells = groups[key]
            starts = [float(c["_start"]) for c in cells]
            ends = [float(c["_end"]) for c in cells]
            rows_out.append([
                length, cond,
                f"{alpha:.1f}" if alpha is not None else "-",
                len(cells),
                f"{np.median(starts):.4f}",
                f"{np.median(ends):.4f}",
            ])
    _print_table(headers, rows_out, aligns)


def report_rr(by_length: dict[str, list[dict]]) -> None:
    _section("4. residual_drift_ratio (rr) — bounded [0, 1] sanity check")
    print("  rr can't blow up like htr. If rr tells the same story stably,")
    print("  the htr trend is real; if rr saturates flat near 1 while htr explodes,")
    print("  the htr movement is mostly start-collapse artefact.")
    print()
    headers = ["length", "condition", "alpha", "n", "med_rr", "q1_rr", "q3_rr"]
    aligns =  ["<",      "<",         "<",     ">", ">",      ">",     ">"]
    rows_out = []
    for length in LENGTHS_ORDER:
        rows = by_length.get(length) or []
        if not rows:
            continue
        groups: dict[tuple, list[float]] = {}
        for r in rows:
            rr = r.get("residual_ratio")
            if not _is_finite(rr):
                continue
            key = (r["condition"], r.get("alpha"))
            groups.setdefault(key, []).append(float(rr))
        for key in sorted(groups.keys(), key=lambda k: _condition_key(*k)):
            cond, alpha = key
            rrs = groups[key]
            med, q1, q3, n = _median_iqr(rrs)
            rows_out.append([
                length, cond,
                f"{alpha:.1f}" if alpha is not None else "-",
                n,
                f"{med:.4f}", f"{q1:.4f}", f"{q3:.4f}",
            ])
    _print_table(headers, rows_out, aligns)


def report_fluency(by_length: dict[str, list[dict]], args: argparse.Namespace) -> None:
    if args.no_fluency:
        _section("5. FLUENCY SAMPLE — SKIPPED (--no-fluency)")
        return
    _section("5. FLUENCY SAMPLE — free generation with SPARC, 2 captions per α")
    # Lazy imports so --no-fluency doesn't pay the torch / model cost.
    try:
        from PIL import Image
        from vr_modality_bias.data.prompts import get_prompt
        from vr_modality_bias.experiment.sparc import SparcHyperparams, enable_sparc
        from vr_modality_bias.models.registry import build_model
        from vr_modality_bias.utils.config import load_config
        from vr_modality_bias.utils.device import resolve_dtype, select_device
    except ModuleNotFoundError:
        from src.vr_modality_bias.data.prompts import get_prompt
        from src.vr_modality_bias.experiment.sparc import SparcHyperparams, enable_sparc
        from src.vr_modality_bias.models.registry import build_model
        from src.vr_modality_bias.utils.config import load_config
        from src.vr_modality_bias.utils.device import resolve_dtype, select_device

    cfg = load_config(args.fluency_config)

    image_id = args.fluency_image_id
    if image_id is None:
        long_rows = by_length.get("long") or []
        if not long_rows:
            print("  (no rows in long parquet — pass --fluency-image-id to override)")
            return
        image_id = long_rows[0]["image_id"]

    images_dir = cfg["dataset"]["images_dir"]
    image_path = Path(images_dir) / f"{image_id}.jpg"
    if not image_path.exists():
        print(f"  image {image_path} not found, skipping fluency.")
        return

    image = Image.open(image_path).convert("RGB")
    prompt = get_prompt(str(cfg["task"]["prompt_key"]))

    print(f"  image_id      : {image_id}")
    print(f"  prompt_key    : {cfg['task']['prompt_key']}")
    print(f"  max_new_tokens: {args.max_new_tokens}")
    print(f"  alphas        : {args.alphas}")
    print(f"  runs per α    : {args.n_per_alpha}")
    print()
    print(f"  loading {cfg['model']['model_id']} ({cfg['model']['dtype']})...")

    model_wrapper = build_model(str(cfg["model"]["key"]))
    model_wrapper.model_id = str(cfg["model"]["model_id"])
    dtype = resolve_dtype(str(cfg["model"]["dtype"]))
    device = select_device("cuda")
    if hasattr(model_wrapper, "_dtype"):
        model_wrapper._dtype = dtype  # noqa: SLF001
    model_wrapper.load(device)
    print(f"  model loaded.  n_layers={model_wrapper.n_layers}")
    print()

    gen_kwargs = {
        "do_sample": True,
        "temperature": float(cfg["generation"]["temperature"]),
        "top_p": float(cfg["generation"]["top_p"]),
        "repetition_penalty": float(cfg["generation"]["repetition_penalty"]),
    }

    # Compute input_len for SPARC buffer (= prompt length excluding image patches).
    processor = model_wrapper._processor  # noqa: SLF001
    messages = model_wrapper._build_messages(prompt, image)
    prefix_text = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    prefix_inputs = processor(text=[prefix_text], images=[image], return_tensors="pt")
    caption_start = int(prefix_inputs["input_ids"].shape[-1])
    image_token_id = int(model_wrapper._model.config.image_token_id)
    image_positions = (prefix_inputs["input_ids"][0] == image_token_id).nonzero(as_tuple=True)[0]
    num_image_patches = int(image_positions.numel())
    input_len = caption_start - num_image_patches
    print(f"  caption_start={caption_start}  num_image_patches={num_image_patches}  input_len={input_len}")
    print()

    for alpha in args.alphas:
        hparams = SparcHyperparams(alpha=float(alpha))
        print(f"  ── α = {alpha:.1f} " + "─" * 50)
        with enable_sparc(model_wrapper, hparams=hparams, probe_image=image, prompt=prompt) as buffer:
            for run in range(args.n_per_alpha):
                buffer.reset()
                buffer.update_input_len(input_len)
                cap = model_wrapper.generate_caption(
                    image=image, prompt=prompt,
                    max_new_tokens=args.max_new_tokens,
                    seed=args.fluency_seed + run,
                    generation_kwargs=gen_kwargs,
                )
                print(f"  [run {run + 1}] {cap}")
        print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, required=True,
        help="Directory containing {short,medium,long}/metrics_sweep.parquet.",
    )
    parser.add_argument("--t0", type=int, default=5)
    parser.add_argument("--no-fluency", action="store_true",
        help="Skip section 5 (avoids loading the model).")
    parser.add_argument(
        "--fluency-config", type=Path,
        default=Path("configs/run_qwen7b_long.yaml"),
        help="Config to pull dataset/prompt/generation params from for fluency.",
    )
    parser.add_argument(
        "--fluency-image-id", type=str, default=None,
        help="Image id to use for fluency. Default: first image in long parquet.",
    )
    parser.add_argument(
        "--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS),
        help="α values to sample for fluency (default 1.1..1.5).",
    )
    parser.add_argument("--n-per-alpha", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--fluency-seed", type=int, default=42)
    args = parser.parse_args()

    if not args.run_dir.exists():
        print(f"ERROR: run-dir {args.run_dir} does not exist.", file=sys.stderr)
        return 1

    print("=" * 78)
    print("PHASE 2 SWEEP — ANALYSIS REPORT")
    print("=" * 78)
    print(f"  generated : {datetime.now(tz=timezone.utc).isoformat(timespec='seconds')}")
    print(f"  run_dir   : {args.run_dir}")
    print(f"  t0        : {args.t0}")
    print()

    by_length: dict[str, list[dict]] = {}
    for length in LENGTHS_ORDER:
        rows = _load_parquet(args.run_dir, length)
        for r in rows:
            _augment_with_decomposition(r, args.t0)
        by_length[length] = rows
        print(f"  loaded {length:<7s}: {len(rows)} cells")

    report_cell_health(by_length)
    report_delta_htr(by_length)
    report_start_end(by_length)
    report_rr(by_length)
    report_fluency(by_length, args)

    print()
    print("=" * 78)
    print("END OF REPORT")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
