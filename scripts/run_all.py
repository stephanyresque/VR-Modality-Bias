#!/usr/bin/env python
"""Single resumable orchestrator — Stage A: SmolVLM diagnostic (generate_refs ->
summarize); Stage B: CHAIR evaluation per family (phase3_generate + chair_report).
Same --run-name resumes via orchestrator_state_<run-name>.json.

Run: make run-all  (smoke: make run-all-smoke)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml
from loguru import logger
from pyprojroot import here


# SPARC hparams per family -- Arthur's recipe scaled to each model's depth.
SPARC_HPARAMS_BY_FAMILY: dict[str, dict] = {
    "smolvlm-2.2b": {
        "alpha": 1.05, "beta": 0.1, "tau": 3.0,
        "selected_layer": 15, "se_layers": [0, 24],
    },
    "llava-1.5-7b": {
        "alpha": 1.05, "beta": 0.1, "tau": 3.0,
        "selected_layer": 20, "se_layers": [0, 32],
    },
    "qwen2.5-vl-7b": {
        "alpha": 1.05, "beta": 0.1, "tau": 3.0,
        "selected_layer": 18, "se_layers": [0, 28],
    },
}

LENGTH_CONFIG_PATTERNS: dict[str, str] = {
    "smolvlm-2.2b": "configs/run_smolvlm22_{length}.yaml",
    "llava-1.5-7b": "configs/run_llava_{length}.yaml",
    "qwen2.5-vl-7b": "configs/run_qwen7b_{length}.yaml",
}

LENGTHS_DEFAULT = ("short", "medium", "long")
FAMILIES_DEFAULT = ("smolvlm-2.2b", "llava-1.5-7b", "qwen2.5-vl-7b")

REPETITION_PENALTY = 1.2  # spec -- applied to both baseline and SPARC

# Diagnostic stage runs only on SmolVLM, per the Block-2 spec.
DIAG_FAMILY = "smolvlm-2.2b"
DIAG_RUN_NAME_PREFIX = "diag_smolvlm_v1"


def _iso_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def _state_path(output_root: Path, run_name: str) -> Path:
    return output_root / f"orchestrator_state_{run_name}.json"


def _load_state(path: Path) -> dict:
    if not path.exists():
        return {"diagnostic": {}, "evaluation": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning(f"state file {path} unreadable; starting fresh")
        return {"diagnostic": {}, "evaluation": {}}


def _save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _write_temp_config(
    src_config_path: Path, *, run_name: str, output_root: Path,
    temp_dir: Path,
) -> Path:
    """Clone ``src_config_path`` to a temp dir with ``run.name`` overridden.

    Lets us invoke scripts generate_refs.py -> summarize.py (which read run_name from the config)
    without modifying the shared per-family configs.
    """
    cfg = yaml.safe_load(src_config_path.read_text(encoding="utf-8"))
    cfg["run"]["name"] = run_name
    cfg["run"]["output_root"] = str(output_root)
    temp_dir.mkdir(parents=True, exist_ok=True)
    out = temp_dir / f"{run_name}.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return out


def _run_subprocess(cmd: list[str], *, log_prefix: str) -> int:
    """Run ``cmd``, stream stdout/stderr to logger, return exit code."""
    logger.info(f"[{log_prefix}] $ " + " ".join(cmd))
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            logger.info(f"[{log_prefix}] {line}")
    return proc.wait()


def _eta(done: int, total: int, t_start: float) -> str:
    if done == 0:
        return "?"
    elapsed = time.time() - t_start
    per = elapsed / done
    remaining = max(0, total - done)
    secs = per * remaining
    if secs < 90:
        return f"{secs:.0f}s"
    if secs < 3600:
        return f"{secs / 60:.1f}min"
    return f"{secs / 3600:.2f}h"


def stage_diagnostic_smolvlm(
    *,
    state: dict,
    state_path: Path,
    output_root: Path,
    temp_dir: Path,
    lengths: list[str],
    limit: int,
    overwrite: bool,
) -> int:
    """Stage A -- run scripts generate_refs.py -> summarize.py for SmolVLM on each length.

    Returns nonzero if any underlying step fails (orchestrator continues
    on a per-length basis; failures just don't mark that length done).
    """
    fail_count = 0
    done_map = state.setdefault("diagnostic", {}).setdefault(DIAG_FAMILY, {})

    src_pattern = LENGTH_CONFIG_PATTERNS[DIAG_FAMILY]
    for length in lengths:
        key = length
        if done_map.get(key) and not overwrite:
            logger.info(f"[DIAG smolvlm/{length}] already done -- skipping (state file)")
            continue

        src_cfg_path = Path(src_pattern.format(length=length))
        if not src_cfg_path.is_file():
            logger.error(f"[DIAG smolvlm/{length}] missing config {src_cfg_path}")
            fail_count += 1
            continue

        run_name = f"{DIAG_RUN_NAME_PREFIX}_{length}"
        temp_cfg = _write_temp_config(
            src_cfg_path, run_name=run_name, output_root=output_root,
            temp_dir=temp_dir,
        )

        ok = True
        for stage_idx, script in enumerate((
            "scripts/generate_refs.py",
            "scripts/collect_hidden_states.py",
            "scripts/compute_metrics.py",
            "scripts/summarize.py",
        ), start=1):
            cmd = [sys.executable, script, "--config", str(temp_cfg), "--limit", str(limit)]
            if overwrite and stage_idx == 1:
                cmd.append("--overwrite")
            rc = _run_subprocess(cmd, log_prefix=f"DIAG {length} step{stage_idx}")
            if rc != 0:
                logger.error(f"[DIAG smolvlm/{length}] {script} exited {rc}; "
                             f"halting this length, keeping state.")
                ok = False
                fail_count += 1
                break

        if ok:
            done_map[length] = True
            state["diagnostic"][DIAG_FAMILY] = done_map
            _save_state(state_path, state)
            logger.info(f"[DIAG smolvlm/{length}] DONE -- state saved.")

    return fail_count


def stage_evaluation_chair(
    *,
    state: dict,
    state_path: Path,
    output_root: Path,
    families: list[str],
    lengths: list[str],
    limit: int,
    overwrite: bool,
) -> int:
    """Stage B -- for each family: generate (phase3_generate.py) then CHAIR (chair_report.py)."""
    fail_count = 0
    eval_state = state.setdefault("evaluation", {})

    for family in families:
        if family not in SPARC_HPARAMS_BY_FAMILY:
            logger.error(f"[EVAL {family}] unknown family; skipping.")
            fail_count += 1
            continue

        fam_state = eval_state.setdefault(family, {})
        run_name = f"chair_{family}_v1"
        run_dir = output_root / run_name

        # ---- phase3_generate.py: generate captions (baseline + SPARC ON) ----
        if fam_state.get("generated") and not overwrite:
            logger.info(f"[EVAL {family}] generation already done -- skipping (state).")
        else:
            sparc = SPARC_HPARAMS_BY_FAMILY[family]
            cmd = [
                sys.executable, "scripts/phase3_generate.py",
                "--run-name", run_name,
                "--output-root", str(output_root),
                "--limit", str(limit),
                "--lengths", *lengths,
                "--length-config-pattern", LENGTH_CONFIG_PATTERNS[family],
                "--alpha", str(sparc["alpha"]),
                "--beta", str(sparc["beta"]),
                "--tau", str(sparc["tau"]),
                "--selected-layer", str(sparc["selected_layer"]),
                "--se-layers", str(sparc["se_layers"][0]), str(sparc["se_layers"][1]),
                "--repetition-penalty", str(REPETITION_PENALTY),
            ]
            if overwrite:
                cmd.append("--overwrite")

            rc = _run_subprocess(cmd, log_prefix=f"EVAL {family} gen")
            if rc != 0:
                logger.error(f"[EVAL {family}] generation failed (rc={rc}); "
                             "skipping CHAIR for this family.")
                fail_count += 1
                continue
            fam_state["generated"] = True
            eval_state[family] = fam_state
            _save_state(state_path, state)

        # ---- chair_report.py: CHAIR over the captions ----
        captions = run_dir / "captions.jsonl"
        if not captions.is_file():
            logger.error(f"[EVAL {family}] expected {captions} but it doesn't exist; "
                         "skipping CHAIR.")
            fail_count += 1
            continue
        if fam_state.get("chair") and not overwrite:
            logger.info(f"[EVAL {family}] CHAIR already done -- skipping (state).")
            continue

        cmd = [
            sys.executable, "scripts/chair_report.py",
            "--run-dir", str(run_dir),
            "--auto-download",
        ]
        rc = _run_subprocess(cmd, log_prefix=f"EVAL {family} chair")
        if rc != 0:
            logger.error(f"[EVAL {family}] CHAIR failed (rc={rc}); leaving state untouched.")
            fail_count += 1
            continue
        fam_state["chair"] = True
        eval_state[family] = fam_state
        _save_state(state_path, state)
        logger.info(f"[EVAL {family}] DONE -- chair_results.{{json,csv}} under {run_dir}")

    return fail_count


def _summarize_diagnostic(output_root: Path, lengths: list[str]) -> dict:
    """Collect median share_tail per length from the SmolVLM diagnostic runs."""
    import pyarrow.parquet as pq
    import numpy as np
    out: dict = {}
    for length in lengths:
        # Latest run dir for this length, via the LATEST pointer.
        run_name = f"{DIAG_RUN_NAME_PREFIX}_{length}"
        ptr = output_root / f"{run_name}_LATEST.txt"
        if not ptr.is_file():
            out[length] = {"status": "no-run", "median_share_tail": None}
            continue
        run_dir = Path(ptr.read_text(encoding="utf-8").strip())
        parq = run_dir / "metrics.parquet"
        if not parq.is_file():
            out[length] = {"status": "no-parquet", "median_share_tail": None}
            continue
        rows = pq.read_table(parq).to_pylist()
        st = [r["share_tail"] for r in rows
              if r.get("share_tail") is not None and np.isfinite(r["share_tail"])]
        out[length] = {
            "status": "ok",
            "n_images": len(rows),
            "n_share_tail_finite": len(st),
            "median_share_tail": float(np.median(st)) if st else None,
        }
    return out


def _summarize_evaluation(output_root: Path, families: list[str]) -> dict:
    """Collect chair_results.csv summaries per family."""
    import csv
    out: dict = {}
    for family in families:
        run_dir = output_root / f"chair_{family}_v1"
        csv_path = run_dir / "chair_results.csv"
        if not csv_path.is_file():
            out[family] = {"status": "no-csv", "rows": []}
            continue
        rows = []
        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                rows.append(r)
        out[family] = {"status": "ok", "rows": rows}
    return out


def _print_final_summary(diag_summary: dict, eval_summary: dict) -> None:
    print()
    print("=" * 78)
    print("RUN-ALL FINAL SUMMARY")
    print("=" * 78)
    print()
    print("[diagnostic] SmolVLM share_tail (median per length):")
    for length, info in diag_summary.items():
        st = info.get("median_share_tail")
        st_str = f"{st:.4f}" if isinstance(st, float) else str(st)
        n = info.get("n_images")
        print(f"  {length:<8}  n={n}  median share_tail={st_str}  ({info['status']})")

    print()
    print("[evaluation] CHAIR per family (off vs on):")
    for family, info in eval_summary.items():
        if info["status"] != "ok":
            print(f"  {family}: {info['status']}")
            continue
        print(f"  {family}:")
        # one line per (length, condition_label)
        for r in info["rows"]:
            cs = r.get("chair_s") or ""
            ci = r.get("chair_i") or ""
            pct = r.get("pct_degen") or ""
            n = r.get("n_captions") or ""
            label = r.get("condition_label", r.get("condition", ""))
            length = r.get("length", "")
            cs_str = f"{float(cs):.4f}" if cs not in ("", None) else "--"
            ci_str = f"{float(ci):.4f}" if ci not in ("", None) else "--"
            pct_str = f"{float(pct):.1f}%" if pct not in ("", None) else "--"
            print(f"    [{length:<6}] {label:<10}  n={n:<3}  "
                  f"CHAIR_s={cs_str}  CHAIR_i={ci_str}  degen={pct_str}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-name", type=str, default="run_all_v1",
        help="State-file identifier. Same name -> resume.",
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("results/runs"),
        help="Where every per-stage run dir lands.",
    )
    parser.add_argument(
        "--limit", type=int, default=50,
        help="Images per length. Default 50 (the full set).",
    )
    parser.add_argument(
        "--lengths", nargs="+", default=list(LENGTHS_DEFAULT),
        choices=list(LENGTHS_DEFAULT),
        help="Which lengths to process in BOTH stages.",
    )
    parser.add_argument(
        "--families", nargs="+", default=list(FAMILIES_DEFAULT),
        choices=list(FAMILIES_DEFAULT),
        help="Which families to evaluate in Stage B.",
    )
    parser.add_argument(
        "--skip-diagnostico", action="store_true",
        help="Skip Stage A (SmolVLM diagnostic).",
    )
    parser.add_argument(
        "--skip-avaliacao", action="store_true",
        help="Skip Stage B (CHAIR evaluation).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Force recompute everything (clears the state file).",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke: --limit 1. Default keeps lengths/families unchanged; "
             "combine with --skip-diagnostico and --families <one> for the "
             "minimal sanity (e.g. one image, only Qwen eval).",
    )
    args = parser.parse_args()

    if args.smoke:
        args.limit = 1
        args.run_name = f"{args.run_name}_smoke"

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    temp_dir = output_root / ".tmp_configs"

    log_dir = output_root / f"{args.run_name}_orchestrator_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "run_all.log"
    logger.add(str(log_file), enqueue=True, level="INFO")

    logger.info("=" * 70)
    logger.info(f"Run-all orchestrator -- run_name={args.run_name}")
    logger.info(f"lengths={args.lengths} families={args.families} limit={args.limit}")
    logger.info(f"skip_diagnostico={args.skip_diagnostico} skip_avaliacao={args.skip_avaliacao}")
    logger.info(f"overwrite={args.overwrite} smoke={args.smoke}")
    logger.info(f"output_root={output_root}  log={log_file}")
    logger.info("=" * 70)

    state_path = _state_path(output_root, args.run_name)
    if args.overwrite and state_path.exists():
        logger.info(f"--overwrite: removing state file {state_path}")
        state_path.unlink()
    state = _load_state(state_path)

    t_start = time.time()
    total_fail = 0

    if not args.skip_diagnostico:
        logger.info("--- Stage A: diagnostic (SmolVLM only) ---")
        total_fail += stage_diagnostic_smolvlm(
            state=state, state_path=state_path,
            output_root=output_root, temp_dir=temp_dir,
            lengths=args.lengths, limit=args.limit,
            overwrite=args.overwrite,
        )
    else:
        logger.info("--- Stage A skipped (--skip-diagnostico) ---")

    if not args.skip_avaliacao:
        logger.info("--- Stage B: CHAIR evaluation (per family) ---")
        total_fail += stage_evaluation_chair(
            state=state, state_path=state_path,
            output_root=output_root, families=args.families,
            lengths=args.lengths, limit=args.limit,
            overwrite=args.overwrite,
        )
    else:
        logger.info("--- Stage B skipped (--skip-avaliacao) ---")

    elapsed = time.time() - t_start
    logger.info("=" * 70)
    logger.info(f"Run-all DONE.  failures={total_fail}  elapsed={elapsed / 60:.1f}min")
    logger.info(f"state file: {state_path}")
    logger.info(f"log file  : {log_file}")
    logger.info("=" * 70)

    diag_summary = _summarize_diagnostic(output_root, args.lengths) if not args.skip_diagnostico else {}
    eval_summary = _summarize_evaluation(output_root, args.families) if not args.skip_avaliacao else {}
    _print_final_summary(diag_summary, eval_summary)

    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover (operator-side)
        import traceback
        logger.error(f"top-level failure: {exc}")
        logger.error(traceback.format_exc())
        raise SystemExit(1)
