"""Tests for scripts/run_sparc_matrix.sh, the offline orchestrator of the
incremental SPARC matrix.

The generation itself needs a GPU, so these exercise only what is checkable
without one: static content of the script, ``bash -n`` syntax, the
DERIVED_LAYER abort, the fully-expanded per-arm commands (via ``PYTHON=echo``),
and shellcheck when it is installed. Bash-dependent tests skip when bash is
absent rather than fail.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parent.parent / "scripts" / "run_sparc_matrix.sh"
_ARMS = ("arm1_sparc", "arm2_adaptive", "arm3_qcond", "arm4_conserve", "arm5_reflayer")

_bash = shutil.which("bash")
_shellcheck = shutil.which("shellcheck")
needs_bash = pytest.mark.skipif(_bash is None, reason="bash not on PATH")


# ---------------------------------------------------------------- static content


def test_script_is_strict_bash():
    text = _SCRIPT.read_text(encoding="utf-8")
    assert text.startswith("#!/usr/bin/env bash")
    assert "set -euo pipefail" in text


def test_script_names_the_five_arms():
    text = _SCRIPT.read_text(encoding="utf-8")
    for arm in _ARMS:
        assert arm in text, arm


def test_script_carries_the_common_hparams_and_improvement_flags():
    text = _SCRIPT.read_text(encoding="utf-8")
    for token in ("ALPHA=1.05", "BETA=0.1", "TAU=3.0", "SELECTED_LAYER=15",
                  "REPETITION_PENALTY=1.2", "--repetition-penalty"):
        assert token in text, token
    for token in ("--adaptive", "--lam", "--ceiling", "--qcond", "--qtop-frac",
                  "--conserve", "--rho", "--sink-frac"):
        assert token in text, token


def test_script_requires_derived_layer_and_points_at_the_deriver():
    text = _SCRIPT.read_text(encoding="utf-8")
    assert "DERIVED_LAYER" in text
    assert "select_reference_layer" in text


# ---------------------------------------------------------------- bash-dependent


@needs_bash
def test_bash_syntax_is_valid():
    result = subprocess.run([_bash, "-n", str(_SCRIPT)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@needs_bash
def test_aborts_when_derived_layer_is_unset():
    env = {k: v for k, v in os.environ.items() if k != "DERIVED_LAYER"}
    result = subprocess.run(
        [_bash, str(_SCRIPT)], capture_output=True, text=True, env=env
    )
    assert result.returncode != 0
    assert "select_reference_layer" in (result.stdout + result.stderr)


def _expanded_commands(tmp_path, extra_env=None):
    out_root = str(tmp_path / "runs").replace("\\", "/")
    env = {
        **os.environ,
        "DERIVED_LAYER": "12",
        "PYTHON": "echo",  # print each command instead of running it
        "OUTPUT_ROOT": out_root,
    }
    if extra_env:
        env.update(extra_env)
    args = [_bash, str(_SCRIPT)]
    if extra_env and extra_env.get("_SMOKE"):
        args.append("--smoke")
        del env["_SMOKE"]
    result = subprocess.run(args, capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    return [ln for ln in result.stdout.splitlines() if "phase3_generate.py" in ln]


def _arm_line(lines: list[str], run_name: str) -> str:
    return next(ln for ln in lines if f"--run-name {run_name} " in ln)


@needs_bash
def test_expands_one_generation_command_per_arm(tmp_path):
    gen = _expanded_commands(tmp_path)
    assert len(gen) == 5


@needs_bash
def test_common_hparams_and_config_pattern_on_every_arm(tmp_path):
    gen = _expanded_commands(tmp_path)
    for ln in gen:
        assert "--alpha 1.05 --beta 0.1 --tau 3.0" in ln
        assert "--se-layers 0 24" in ln
        assert "--repetition-penalty 1.2" in ln
        # Regression: the {length} placeholder must survive intact.
        assert "--length-config-pattern configs/run_smolvlm22_{length}.yaml" in ln
        assert "--limit 100" in ln


@needs_bash
def test_each_arm_adds_exactly_its_improvement_flags(tmp_path):
    gen = _expanded_commands(tmp_path)

    arm1 = _arm_line(gen, "arm1_sparc")
    assert "--adaptive" not in arm1
    assert "--selected-layer 15" in arm1

    arm2 = _arm_line(gen, "arm2_adaptive")
    assert "--adaptive --lam 0.5 --ceiling 2.0" in arm2
    assert "--qcond" not in arm2

    arm3 = _arm_line(gen, "arm3_qcond")
    assert "--qcond --qtop-frac 0.05" in arm3
    assert "--conserve" not in arm3

    arm4 = _arm_line(gen, "arm4_conserve")
    assert "--conserve --rho 0.5 --sink-frac 0.05" in arm4
    assert "--selected-layer 15" in arm4


@needs_bash
def test_only_arm5_uses_the_derived_layer(tmp_path):
    gen = _expanded_commands(tmp_path)
    for name in ("arm1_sparc", "arm2_adaptive", "arm3_qcond", "arm4_conserve"):
        assert "--selected-layer 15" in _arm_line(gen, name)
    arm5 = _arm_line(gen, "arm5_reflayer")
    assert "--selected-layer 12" in arm5  # DERIVED_LAYER
    assert "--conserve --rho 0.5 --sink-frac 0.05" in arm5  # arm4's flags carried over


@needs_bash
def test_smoke_uses_two_images_and_suffixed_run_names(tmp_path):
    gen = _expanded_commands(tmp_path, extra_env={"_SMOKE": "1"})
    assert len(gen) == 5
    for ln in gen:
        assert "--limit 2" in ln
    for arm in _ARMS:
        assert any(f"--run-name {arm}_smoke " in ln for ln in gen)


@pytest.mark.skipif(_shellcheck is None, reason="shellcheck not on PATH")
def test_shellcheck_is_clean():
    result = subprocess.run([_shellcheck, str(_SCRIPT)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
