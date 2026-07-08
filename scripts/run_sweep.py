#!/usr/bin/env python
"""Orchestrator over (model × caption length) cells — runs scripts/run_caption_sweep.py per cell.

Run: python scripts/run_sweep.py [--models ...] [--lengths ...] [--limit N] [--overwrite]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections.abc import Iterable
from pathlib import Path

from vr_modality_bias.utils.logging import configure_logging, get_logger
from vr_modality_bias.utils.profiling import format_seconds

ALL_MODELS: tuple[str, ...] = ("smolvlm-2.2b", "qwen2.5-vl-7b")
ALL_LENGTHS: tuple[str, ...] = ("short", "medium", "long")

_CONFIG_BY_CELL: dict[tuple[str, str], Path] = {
    ("smolvlm-2.2b", "short"): Path("configs/run_smolvlm22_short.yaml"),
    ("smolvlm-2.2b", "medium"): Path("configs/run_smolvlm22_medium.yaml"),
    ("smolvlm-2.2b", "long"): Path("configs/run_smolvlm22_long.yaml"),
    ("qwen2.5-vl-7b", "short"): Path("configs/run_qwen7b_short.yaml"),
    ("qwen2.5-vl-7b", "medium"): Path("configs/run_qwen7b_medium.yaml"),
    ("qwen2.5-vl-7b", "long"): Path("configs/run_qwen7b_long.yaml"),
}


def expand_cells(
    models: Iterable[str] | None,
    lengths: Iterable[str] | None,
) -> list[tuple[str, str, Path]]:
    
    selected_models = tuple(models) if models else ALL_MODELS
    selected_lengths = tuple(lengths) if lengths else ALL_LENGTHS

    bad_models = [m for m in selected_models if m not in ALL_MODELS]
    if bad_models:
        raise ValueError(
            f"Unknown model(s) {bad_models}. Known: {list(ALL_MODELS)}."
        )
    bad_lengths = [length for length in selected_lengths if length not in ALL_LENGTHS]
    if bad_lengths:
        raise ValueError(
            f"Unknown length(s) {bad_lengths}. Known: {list(ALL_LENGTHS)}."
        )

    cells: list[tuple[str, str, Path]] = []
    for model in selected_models:
        for length in selected_lengths:
            config = _CONFIG_BY_CELL.get((model, length))
            if config is None:
                raise ValueError(
                    f"No config registered for cell ({model}, {length})."
                )
            cells.append((model, length, config))
    return cells


def _run_cell(
    config_path: Path,
    *,
    limit: int | None,
    overwrite: bool,
) -> tuple[int, float]:

    cmd = [
        sys.executable,
        "scripts/run_caption_sweep.py",
        "--config",
        str(config_path),
    ]
    if limit:
        cmd.extend(["--limit", str(limit)])
    if overwrite:
        cmd.append("--overwrite")

    started = time.perf_counter()
    result = subprocess.run(cmd, check=False)
    elapsed = time.perf_counter() - started
    return result.returncode, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        choices=list(ALL_MODELS),
        help="Subset of models to run. Default: all.",
    )
    parser.add_argument(
        "--lengths",
        nargs="+",
        default=None,
        choices=list(ALL_LENGTHS),
        help="Subset of caption lengths to run. Default: all.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Per-cell image limit (smoke testing). Default: no limit.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Forward --overwrite to scripts/09.",
    )
    args = parser.parse_args()

    configure_logging()
    log = get_logger(__name__)

    cells = expand_cells(args.models, args.lengths)
    log.info("Sweep plan — %d cell(s):", len(cells))
    for model, length, config in cells:
        log.info("  - %s × %s -> %s", model, length, config)

    results: list[dict] = []
    for model, length, config in cells:
        log.info("=" * 70)
        log.info("Cell start: %s × %s (config=%s)", model, length, config)
        code, elapsed = _run_cell(config, limit=args.limit, overwrite=args.overwrite)
        results.append(
            {
                "model": model,
                "length": length,
                "config": str(config),
                "returncode": code,
                "seconds": elapsed,
            }
        )
        if code == 0:
            log.info(
                "Cell OK: %s × %s in %s", model, length, format_seconds(elapsed)
            )
        else:
            log.error(
                "Cell FAILED (returncode=%d): %s × %s after %s",
                code,
                model,
                length,
                format_seconds(elapsed),
            )

    n_ok = sum(1 for r in results if r["returncode"] == 0)
    n_fail = len(results) - n_ok
    log.info("=" * 70)
    log.info("Sweep done — %d OK, %d FAILED", n_ok, n_fail)
    for r in results:
        status = "OK    " if r["returncode"] == 0 else f"FAIL[{r['returncode']}]"
        log.info(
            "  %s | %-14s × %-7s | %s",
            status,
            r["model"],
            r["length"],
            format_seconds(r["seconds"]),
        )
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
