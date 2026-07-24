"""Tests for the phase3_generate CLI wiring: the argparse defaults and flags
that feed SparcHyperparams (adaptive / lam / ceiling) and the run_params
snapshot.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).parent.parent / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"_script_{name}", _SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def phase3():
    return _load_script("phase3_generate")


def test_phase3_defaults_keep_the_original_alpha_path(phase3):
    hp = phase3.sparc_hparams_from_args(phase3.build_parser().parse_args([]))
    assert hp.adaptive is False
    assert hp.lam == 0.0
    assert hp.ceiling == 2.0


def test_phase3_adaptive_flag_switches_the_path(phase3):
    hp = phase3.sparc_hparams_from_args(phase3.build_parser().parse_args(["--adaptive"]))
    assert hp.adaptive is True
    assert hp.lam == 0.0


def test_phase3_adaptive_accepts_lam_and_ceiling(phase3):
    args = phase3.build_parser().parse_args(
        ["--adaptive", "--lam", "0.7", "--ceiling", "1.5"]
    )
    hp = phase3.sparc_hparams_from_args(args)
    assert (hp.adaptive, hp.lam, hp.ceiling) == (True, 0.7, 1.5)


def test_phase3_adaptive_does_not_require_alpha_above_one(phase3):
    args = phase3.build_parser().parse_args(["--adaptive", "--alpha", "1.0"])
    assert phase3.sparc_hparams_from_args(args).adaptive is True


def test_phase3_without_adaptive_still_rejects_alpha_at_one(phase3):
    args = phase3.build_parser().parse_args(["--alpha", "1.0"])
    with pytest.raises(ValueError, match="alpha"):
        phase3.sparc_hparams_from_args(args)


def test_phase3_sparc_hparams_land_in_the_run_params_snapshot(phase3):
    """run_params.json is built by splatting as_dict(), so new hyperparameters
    reach the artefact without touching the snapshot code."""
    hp = phase3.sparc_hparams_from_args(
        phase3.build_parser().parse_args(["--adaptive", "--lam", "0.3"])
    )
    snapshot = {"run_name": "x", **hp.as_dict()}
    assert snapshot["adaptive"] is True
    assert snapshot["lam"] == 0.3
    assert snapshot["ceiling"] == 2.0
    json.dumps(snapshot)  # must stay serialisable
