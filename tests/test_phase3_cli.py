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


def test_phase3_conserve_defaults_are_off(phase3):
    args = phase3.build_parser().parse_args([])
    assert args.conserve is False
    assert args.rho == 0.5
    assert args.sink_frac == 0.05


def test_phase3_conserve_flags_wire_into_the_hparams(phase3):
    args = phase3.build_parser().parse_args(
        ["--adaptive", "--qcond", "--conserve", "--rho", "0.25", "--sink-frac", "0.1"]
    )
    hp = phase3.sparc_hparams_from_args(args)
    assert hp.conserve is True
    assert hp.rho == 0.25
    assert hp.sink_frac == 0.1


def test_phase3_conserve_lands_in_the_run_params_snapshot(phase3):
    hp = phase3.sparc_hparams_from_args(
        phase3.build_parser().parse_args(["--adaptive", "--qcond", "--conserve"])
    )
    snapshot = {"run_name": "x", **hp.as_dict()}
    assert snapshot["conserve"] is True
    assert snapshot["rho"] == 0.5
    assert snapshot["sink_frac"] == 0.05
    json.dumps(snapshot)  # must stay serialisable


# ---------------------------------------------------------------- per-entry sparc record


def test_sparc_snapshot_is_the_full_dict_for_on_and_none_for_off(phase3):
    hp = phase3.sparc_hparams_from_args(
        phase3.build_parser().parse_args(["--adaptive", "--lam", "0.3"])
    )
    assert phase3._sparc_snapshot(hp) == hp.as_dict()
    assert phase3._sparc_snapshot(None) is None


# ---------------------------------------------------------------- resume-arm guard


def _write_captions(path: Path, entries: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry) + "\n")


def _sparc_for(phase3, argv: list[str]) -> dict:
    return phase3.sparc_hparams_from_args(phase3.build_parser().parse_args(argv)).as_dict()


def test_resume_guard_accepts_the_same_arm(phase3, tmp_path):
    sparc = _sparc_for(phase3, ["--adaptive", "--lam", "0.5"])
    path = tmp_path / "captions.jsonl"
    _write_captions(path, [
        {"image_id": "a", "length": "short", "condition": "off", "sparc": None},
        {"image_id": "a", "length": "short", "condition": "on", "sparc": sparc},
    ])
    phase3.assert_resume_arm_matches(path, sparc)  # must not raise


def test_resume_guard_rejects_a_different_arm(phase3, tmp_path):
    written = _sparc_for(phase3, ["--adaptive", "--lam", "0.5"])
    current = _sparc_for(phase3, ["--adaptive", "--lam", "0.7"])
    path = tmp_path / "captions.jsonl"
    _write_captions(path, [{"image_id": "a", "length": "short", "condition": "on", "sparc": written}])
    with pytest.raises(ValueError, match="different SPARC arm"):
        phase3.assert_resume_arm_matches(path, current)


def test_resume_guard_rejects_a_legacy_entry_without_the_record(phase3, tmp_path):
    current = _sparc_for(phase3, ["--adaptive"])
    path = tmp_path / "captions.jsonl"
    _write_captions(path, [{"image_id": "a", "length": "short", "condition": "on", "alpha": 1.1}])
    with pytest.raises(ValueError, match="different SPARC arm"):
        phase3.assert_resume_arm_matches(path, current)


def test_resume_guard_ignores_off_entries(phase3, tmp_path):
    current = _sparc_for(phase3, ["--adaptive"])
    path = tmp_path / "captions.jsonl"
    _write_captions(path, [{"image_id": "a", "length": "short", "condition": "off", "sparc": None}])
    phase3.assert_resume_arm_matches(path, current)  # off is not an arm; no raise


def test_resume_guard_is_a_noop_when_the_file_is_absent(phase3, tmp_path):
    phase3.assert_resume_arm_matches(tmp_path / "missing.jsonl", {"adaptive": True})
