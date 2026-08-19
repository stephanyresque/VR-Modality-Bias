"""Tests for the three shell orchestrators.

Generation needs a GPU, so nothing here executes it: the scripts are run with
``PYTHON=echo`` so every command they would issue is printed instead, and the
assertions are about how those commands are built. Bash-dependent tests skip
when bash is absent rather than fail.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).parent.parent / "scripts"
_MATRIX = _SCRIPTS / "run_sparc_matrix.sh"
_COMPOSED = _SCRIPTS / "run_composed_matrix.sh"
_EXPERIMENT = _SCRIPTS / "run_experiment.sh"

_ARMS = ("baseline", "arm1_sparc", "arm2_adaptive", "arm3_qcond",
         "arm4_conserve", "arm5_reflayer")
_PILOT = "baseline arm1_sparc arm5_reflayer"

_bash = shutil.which("bash")
needs_bash = pytest.mark.skipif(_bash is None, reason="bash not on PATH")


def _run(script: Path, env_extra: dict, *args) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHON": "echo", **env_extra}
    return subprocess.run(
        [_bash, str(script), *args], capture_output=True, text=True, env=env
    )


def _matrix_env(tmp_path: Path, **extra) -> dict:
    return {
        "DERIVED_LAYER": "12",
        "VOCABULARY": "vocab.json",
        "ANNOTATIONS": "annotations.jsonl",
        "OUTPUT_ROOT": str(tmp_path / "runs").replace("\\", "/"),
        **extra,
    }


def _composed_env(tmp_path: Path, **extra) -> dict:
    return {
        "DERIVED_LAYER": "12",
        "CONFIG": "cfg.yaml",
        "QUESTIONS": "questions.jsonl",
        "DIRECTION_TERMS": "direction_terms.json",
        "OUTPUT_ROOT": str(tmp_path / "runs").replace("\\", "/"),
        **extra,
    }


def _gen_lines(stdout: str, script: str) -> list[str]:
    return [ln for ln in stdout.splitlines() if script in ln]


# ---------------------------------------------------------------- static


def test_all_three_scripts_are_strict_bash():
    for script in (_MATRIX, _COMPOSED, _EXPERIMENT):
        text = script.read_text(encoding="utf-8")
        assert text.startswith("#!/usr/bin/env bash"), script.name
        assert "set -euo pipefail" in text, script.name


@needs_bash
@pytest.mark.parametrize("script", [_MATRIX, _COMPOSED, _EXPERIMENT])
def test_bash_syntax_is_valid(script):
    result = subprocess.run([_bash, "-n", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------- arm choice


@needs_bash
def test_the_description_matrix_still_defaults_to_the_five_arms(tmp_path: Path):
    result = _run(_MATRIX, _matrix_env(tmp_path))

    assert result.returncode == 0, result.stderr
    gen = _gen_lines(result.stdout, "phase3_generate.py")
    assert len(gen) == 5
    assert not any("--run-name baseline " in ln for ln in gen), (
        "baseline is opt-in; the default must stay the five incremental arms"
    )


@needs_bash
def test_the_composed_matrix_defaults_to_the_five_arms(tmp_path: Path):
    result = _run(_COMPOSED, _composed_env(tmp_path))

    assert result.returncode == 0, result.stderr
    assert len(_gen_lines(result.stdout, "composed_generate.py")) == 5


@needs_bash
@pytest.mark.parametrize("script,env_fn,gen_script", [
    (_MATRIX, _matrix_env, "phase3_generate.py"),
    (_COMPOSED, _composed_env, "composed_generate.py"),
])
def test_arms_selects_exactly_the_three_pilot_arms(tmp_path, script, env_fn, gen_script):
    result = _run(script, env_fn(tmp_path, ARMS=_PILOT))

    assert result.returncode == 0, result.stderr
    gen = _gen_lines(result.stdout, gen_script)
    assert len(gen) == 3
    for arm in ("baseline", "arm1_sparc", "arm5_reflayer"):
        assert any(f"--run-name {arm}" in ln for ln in gen), arm
    assert not any("arm2_adaptive" in ln for ln in gen)


@needs_bash
@pytest.mark.parametrize("script,env_fn", [
    (_MATRIX, _matrix_env), (_COMPOSED, _composed_env),
])
def test_an_unknown_arm_aborts_before_anything_runs(tmp_path, script, env_fn):
    result = _run(script, env_fn(tmp_path, ARMS="arm1_sparc arm9_bogus"))

    assert result.returncode != 0
    assert "arm9_bogus" in result.stdout + result.stderr
    assert "phase3_generate.py" not in result.stdout
    assert "composed_generate.py" not in result.stdout


@needs_bash
@pytest.mark.parametrize("script,env_fn,gen_script", [
    (_MATRIX, _matrix_env, "phase3_generate.py"),
    (_COMPOSED, _composed_env, "composed_generate.py"),
])
def test_a_single_arm_runs_alone(tmp_path, script, env_fn, gen_script):
    result = _run(script, env_fn(tmp_path, ARMS="arm3_qcond"))

    gen = _gen_lines(result.stdout, gen_script)
    assert len(gen) == 1
    assert "--qcond --qtop-frac 0.05" in gen[0]


# ---------------------------------------------------------------- arm flags


@needs_bash
def test_the_baseline_arm_generates_only_the_off_condition(tmp_path: Path):
    result = _run(_MATRIX, _matrix_env(tmp_path, ARMS="baseline"))

    line = _gen_lines(result.stdout, "phase3_generate.py")[0]
    assert "--baseline-only" in line, (
        "'no intervention' has to mean no ON pass at all, not an intervention "
        "tuned to zero"
    )
    assert "--adaptive" not in line
    assert "--qcond" not in line


@needs_bash
def test_only_arm5_uses_the_derived_layer(tmp_path: Path):
    result = _run(_MATRIX, _matrix_env(tmp_path, ARMS="arm1_sparc arm5_reflayer"))

    gen = _gen_lines(result.stdout, "phase3_generate.py")
    arm1 = next(ln for ln in gen if "--run-name arm1_sparc" in ln)
    arm5 = next(ln for ln in gen if "--run-name arm5_reflayer" in ln)
    assert "--selected-layer 15" in arm1
    assert "--selected-layer 12" in arm5


@needs_bash
def test_the_composed_track_passes_its_config_and_questions(tmp_path: Path):
    result = _run(_COMPOSED, _composed_env(tmp_path, ARMS="arm1_sparc"))

    gen = _gen_lines(result.stdout, "composed_generate.py")[0]
    assert "--config cfg.yaml" in gen
    assert "--questions questions.jsonl" in gen
    assert "--lengths" not in gen, "the composed track has no length regime"

    report = _gen_lines(result.stdout, "composed_report.py")[0]
    assert "--direction-terms direction_terms.json" in report


@needs_bash
def test_the_composed_run_dirs_cannot_collide_with_the_description_ones(tmp_path: Path):
    composed = _run(_COMPOSED, _composed_env(tmp_path, ARMS="arm1_sparc"))
    description = _run(_MATRIX, _matrix_env(tmp_path, ARMS="arm1_sparc"))

    composed_name = _gen_lines(composed.stdout, "composed_generate.py")[0]
    description_name = _gen_lines(description.stdout, "phase3_generate.py")[0]
    assert "--run-name arm1_sparc_q" in composed_name
    assert "--run-name arm1_sparc " in description_name


# ---------------------------------------------------------------- smoke


@needs_bash
@pytest.mark.parametrize("script,env_fn,gen_script,limit_flag", [
    (_MATRIX, _matrix_env, "phase3_generate.py", "--limit 2"),
    (_COMPOSED, _composed_env, "composed_generate.py", "--limit 2"),
])
def test_smoke_shrinks_the_run_and_suffixes_the_dirs(
    tmp_path, script, env_fn, gen_script, limit_flag
):
    result = _run(script, env_fn(tmp_path, ARMS=_PILOT), "--smoke")

    gen = _gen_lines(result.stdout, gen_script)
    assert len(gen) == 3
    for line in gen:
        assert limit_flag in line
        assert "_smoke" in line


# ---------------------------------------------------------------- experiment


def _experiment_env(tmp_path: Path, **extra) -> dict:
    return {
        "DATASET": "demo",
        "CONFIG_PATTERN": "configs/run_smolvlm22_{length}.yaml",
        "COMPOSED_CONFIG": "configs/run_smolvlm22_long.yaml",
        "VOCABULARY": "vocab.json",
        "ANNOTATIONS": "annotations.jsonl",
        "QUESTIONS": "questions.jsonl",
        "DIRECTION_TERMS": "configs/direction_terms.json",
        "OUTPUT_ROOT": str(tmp_path / "runs").replace("\\", "/"),
        **extra,
    }


@needs_bash
def test_a_missing_required_path_stops_the_experiment(tmp_path: Path):
    env = _experiment_env(tmp_path)
    del env["QUESTIONS"]
    result = subprocess.run(
        [_bash, str(_EXPERIMENT)], capture_output=True, text=True,
        env={**os.environ, "PYTHON": "echo", **env, "QUESTIONS": ""},
    )

    assert result.returncode != 0
    assert "QUESTIONS" in result.stdout + result.stderr


@needs_bash
def test_an_unknown_stage_is_rejected(tmp_path: Path):
    result = _run(_EXPERIMENT, _experiment_env(tmp_path, STAGES="preflight bogus"))

    assert result.returncode != 0
    assert "bogus" in result.stdout + result.stderr


@needs_bash
def test_the_stage_selection_runs_only_what_was_asked(tmp_path: Path):
    """STAGES=description with a forced layer must not re-run the diagnostic."""
    result = _run(_EXPERIMENT, _experiment_env(
        tmp_path, STAGES="description", DERIVED_LAYER="9", ARMS="arm1_sparc",
    ))

    out = result.stdout
    assert "STAGE description" in out
    assert "STAGE diagnostic" not in out
    assert "STAGE preflight" not in out
    assert "generate_refs.py" not in out


@needs_bash
def test_the_description_stage_receives_the_forced_reference_layer(tmp_path: Path):
    result = _run(_EXPERIMENT, _experiment_env(
        tmp_path, STAGES="description", DERIVED_LAYER="9", ARMS="arm5_reflayer",
    ))

    gen = _gen_lines(result.stdout, "phase3_generate.py")
    assert gen and "--selected-layer 9" in gen[0]


@needs_bash
def test_the_later_stages_refuse_to_run_without_a_reference_layer(tmp_path: Path):
    result = _run(_EXPERIMENT, _experiment_env(
        tmp_path, STAGES="description", ARMS="arm1_sparc",
    ))

    assert result.returncode != 0
    assert "no reference layer" in result.stdout + result.stderr
    assert "phase3_generate.py" not in result.stdout


@needs_bash
def test_the_derived_layer_file_is_read_when_the_diagnostic_already_ran(tmp_path: Path):
    experiment_dir = tmp_path / "runs" / "experiment_demo"
    experiment_dir.mkdir(parents=True)
    (experiment_dir / "derived_layer.txt").write_text("17\n", encoding="utf-8")

    result = _run(_EXPERIMENT, _experiment_env(
        tmp_path, STAGES="description", ARMS="arm5_reflayer",
    ))

    assert "reference layer 17 read from" in result.stdout
    gen = _gen_lines(result.stdout, "phase3_generate.py")
    assert gen and "--selected-layer 17" in gen[0]


@needs_bash
def test_the_diagnostic_stage_chains_the_four_scripts_then_derives(tmp_path: Path):
    result = _run(_EXPERIMENT, _experiment_env(tmp_path, STAGES="diagnostic"))

    out = result.stdout
    for script in ("generate_refs.py", "collect_hidden_states.py",
                   "compute_metrics.py", "summarize.py"):
        assert script in out, script
    assert "select_reference_layer.py" in out
    assert "--out" in out, "the derived layer must be written to a file"


@needs_bash
def test_the_run_is_logged_to_a_timestamped_file(tmp_path: Path):
    _run(_EXPERIMENT, _experiment_env(
        tmp_path, STAGES="description", DERIVED_LAYER="9", ARMS="arm1_sparc",
    ))

    logs = list((tmp_path / "runs" / "experiment_demo").glob("run_*.log"))
    assert len(logs) == 1, "the run must leave a log to tail from outside tmux"
    assert "STAGE description" in logs[0].read_text(encoding="utf-8")


@needs_bash
def test_the_summary_names_the_stages_and_where_results_landed(tmp_path: Path):
    result = _run(_EXPERIMENT, _experiment_env(
        tmp_path, STAGES="description", DERIVED_LAYER="9", ARMS="arm1_sparc",
    ))

    out = result.stdout
    assert "EXPERIMENT SUMMARY" in out
    assert "description OK" in out
    assert "results under" in out
    assert "log file" in out


@needs_bash
def test_smoke_propagates_all_the_way_down(tmp_path: Path):
    result = _run(
        _EXPERIMENT,
        _experiment_env(tmp_path, STAGES="description", DERIVED_LAYER="9",
                        ARMS="arm1_sparc"),
        "--smoke",
    )

    gen = _gen_lines(result.stdout, "phase3_generate.py")
    assert gen and "--limit 2" in gen[0]
    assert "_smoke" in gen[0]


# ---------------------------------------------------------------- length regimes
#
# The short regime sat inside the noise in every condition of the previous
# matrix and was cut from scope; keeping it cost a third of the GPU time.


@needs_bash
def test_the_description_matrix_defaults_to_medium_and_long(tmp_path: Path):
    result = _run(_MATRIX, _matrix_env(tmp_path, ARMS="arm1_sparc"))

    line = _gen_lines(result.stdout, "phase3_generate.py")[0]
    assert "--lengths medium long " in line
    assert "short" not in line.split("--length-config-pattern")[0]


@needs_bash
def test_lengths_can_be_overridden(tmp_path: Path):
    result = _run(_MATRIX, _matrix_env(tmp_path, ARMS="arm1_sparc", LENGTHS="long"))

    line = _gen_lines(result.stdout, "phase3_generate.py")[0]
    assert "--lengths long " in line


@needs_bash
def test_an_unknown_length_regime_aborts(tmp_path: Path):
    result = _run(_MATRIX, _matrix_env(tmp_path, ARMS="arm1_sparc", LENGTHS="bogus"))

    assert result.returncode != 0
    assert "bogus" in result.stdout + result.stderr
    assert "phase3_generate.py" not in result.stdout


# ---------------------------------------------------------------- per-stage demand


def _adt_env(tmp_path: Path, **extra) -> dict:
    """ADT shape: objects, no questions."""
    return {
        "DATASET": "adt",
        "CONFIG_PATTERN": "configs/run_smolvlm22_{length}.yaml",
        "VOCABULARY": "vocab.json",
        "ANNOTATIONS": "annotations.jsonl",
        "OUTPUT_ROOT": str(tmp_path / "runs").replace("\\", "/"),
        **extra,
    }


def _odi_env(tmp_path: Path, **extra) -> dict:
    """ODI-Bench shape: questions, no per-image object list."""
    return {
        "DATASET": "odi",
        "COMPOSED_CONFIG": "configs/run_smolvlm22_long.yaml",
        "QUESTIONS": "questions.jsonl",
        "DIRECTION_TERMS": "configs/direction_terms.json",
        "OUTPUT_ROOT": str(tmp_path / "runs").replace("\\", "/"),
        **extra,
    }


@needs_bash
def test_an_object_only_dataset_runs_the_description_track(tmp_path: Path):
    result = _run(_EXPERIMENT, _adt_env(
        tmp_path, STAGES="preflight description", DERIVED_LAYER="9",
        ARMS="arm1_sparc",
    ))

    assert result.returncode == 0, result.stdout[-2000:]
    assert _gen_lines(result.stdout, "phase3_generate.py")


@needs_bash
def test_a_question_only_dataset_runs_the_composed_track(tmp_path: Path):
    result = _run(_EXPERIMENT, _odi_env(
        tmp_path, STAGES="preflight composed", DERIVED_LAYER="9",
        ARMS="arm1_sparc",
    ))

    assert result.returncode == 0, result.stdout[-2000:]
    assert _gen_lines(result.stdout, "composed_generate.py")


@needs_bash
def test_the_description_stage_still_demands_its_own_artefacts(tmp_path: Path):
    env = _adt_env(tmp_path, STAGES="description", DERIVED_LAYER="9")
    del env["VOCABULARY"]

    result = _run(_EXPERIMENT, {**env, "VOCABULARY": ""})

    assert result.returncode != 0
    assert "VOCABULARY" in result.stdout + result.stderr
    assert "description" in result.stdout + result.stderr


@needs_bash
def test_the_composed_stage_still_demands_its_own_artefacts(tmp_path: Path):
    result = _run(_EXPERIMENT, _odi_env(
        tmp_path, STAGES="composed", DERIVED_LAYER="9", QUESTIONS="",
    ))

    assert result.returncode != 0
    assert "QUESTIONS" in result.stdout + result.stderr


@needs_bash
def test_preflight_receives_only_the_artefacts_the_dataset_has(tmp_path: Path):
    adt = _run(_EXPERIMENT, _adt_env(
        tmp_path, STAGES="preflight description", DERIVED_LAYER="9", ARMS="arm1_sparc",
    ))
    odi = _run(_EXPERIMENT, _odi_env(
        tmp_path, STAGES="preflight composed", DERIVED_LAYER="9", ARMS="arm1_sparc",
    ))

    adt_line = _gen_lines(adt.stdout, "preflight.py")[0]
    assert "--annotations" in adt_line and "--vocabulary" in adt_line
    assert "--questions" not in adt_line and "--direction-terms" not in adt_line

    odi_line = _gen_lines(odi.stdout, "preflight.py")[0]
    assert "--questions" in odi_line and "--direction-terms" in odi_line
    assert "--annotations" not in odi_line and "--vocabulary" not in odi_line


@needs_bash
def test_preflight_only_demands_the_regimes_that_will_run(tmp_path: Path):
    result = _run(_EXPERIMENT, _adt_env(
        tmp_path, STAGES="preflight description", DERIVED_LAYER="9", ARMS="arm1_sparc",
    ))

    line = _gen_lines(result.stdout, "preflight.py")[0]
    assert "run_smolvlm22_medium.yaml" in line
    assert "run_smolvlm22_long.yaml" in line
    assert "run_smolvlm22_short.yaml" not in line, (
        "expanding a regime the run never opens would demand a file for nothing"
    )


@needs_bash
def test_the_diagnostic_falls_back_to_the_deepest_description_regime(tmp_path: Path):
    """An object-only dataset has no COMPOSED_CONFIG to borrow."""
    result = _run(_EXPERIMENT, _adt_env(tmp_path, STAGES="diagnostic"))

    assert result.returncode == 0, result.stdout[-2000:]
    assert "configs/run_smolvlm22_long.yaml" in result.stdout
