"""Tests for the pure logic of scripts/memvr_margin_diag.py.

The forward loop needs a GPU + checkpoint, so the script keeps its logic in
module-level functions and only those are exercised here: yes/no token
resolution, the margin, the directionality summary (directional vs generic-bias
cases), and the CLI flags.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_SCRIPTS = Path(__file__).parent.parent / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"_script_{name}", _SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def diag():
    return _load_script("memvr_margin_diag")


# ---------------------------------------------------------------- yes/no tokens


class _MockTokenizer:
    """Returns a fixed first-token id list per surface form."""

    def __init__(self, mapping: dict):
        self._mapping = mapping

    def encode(self, text, add_special_tokens=False):
        return self._mapping.get(text, [])


_MAPPING = {
    "Yes": [10], " Yes": [11], "yes": [12], " yes": [13],
    "No": [20], " No": [21], "no": [22], " no": [23],
}


def test_resolve_yes_no_tokens_picks_the_highest_baseline_variant(diag):
    logits = torch.full((30,), -10.0)
    logits[11] = 5.0  # " Yes" wins among the yes forms
    logits[10] = 1.0
    logits[21] = 4.0  # " No" wins among the no forms
    logits[20] = 0.0
    meta = diag.resolve_yes_no_tokens(_MockTokenizer(_MAPPING), logits)
    assert (meta["yes_id"], meta["yes_variant"]) == (11, " Yes")
    assert (meta["no_id"], meta["no_variant"]) == (21, " No")


def test_resolve_yes_no_tokens_can_choose_the_bare_form(diag):
    logits = torch.full((30,), -10.0)
    logits[10] = 9.0  # "Yes" (no leading space) wins
    logits[22] = 9.0  # "no" wins
    meta = diag.resolve_yes_no_tokens(_MockTokenizer(_MAPPING), logits)
    assert meta["yes_variant"] == "Yes"
    assert meta["no_variant"] == "no"


def test_resolve_yes_no_tokens_raises_when_nothing_tokenises(diag):
    with pytest.raises(RuntimeError, match="yes/no"):
        diag.resolve_yes_no_tokens(_MockTokenizer({}), torch.zeros(30))


def test_margin_from_logits_is_yes_minus_no(diag):
    logits = torch.zeros(30)
    logits[11] = 2.0
    logits[21] = 0.5
    assert diag.margin_from_logits(logits, 11, 21) == pytest.approx(1.5)


def test_margin_can_be_negative(diag):
    logits = torch.zeros(30)
    logits[11] = -1.0
    logits[21] = 1.0
    assert diag.margin_from_logits(logits, 11, 21) == pytest.approx(-2.0)


# ---------------------------------------------------------------- directionality verdict


def test_directional_verdict(diag):
    assert diag.classify_directionality(0.5, -0.5) == "directional"


def test_generic_bias_verdict_both_positive(diag):
    assert diag.classify_directionality(0.5, 0.5) == "generic_bias"


def test_generic_bias_verdict_both_negative(diag):
    assert diag.classify_directionality(-0.5, -0.5) == "generic_bias"


def test_anti_directional_is_inconclusive(diag):
    assert diag.classify_directionality(-0.5, 0.5) == "inconclusive"


def test_a_zero_group_mean_is_inconclusive(diag):
    assert diag.classify_directionality(0.0, -0.5) == "inconclusive"


def test_an_empty_group_is_undetermined(diag):
    assert diag.classify_directionality(float("nan"), -0.5) == "undetermined"


# ---------------------------------------------------------------- summary


def test_build_summary_detects_a_directional_shift(diag):
    per_question = [
        {"expected": "yes", "margins": {"baseline": 0.0, "memvr": 0.5}, "fired": {"memvr": True}},
        {"expected": "no", "margins": {"baseline": 0.0, "memvr": -0.5}, "fired": {"memvr": True}},
    ]
    s = diag.build_summary(per_question, ["baseline", "memvr"])["memvr"]
    assert s["verdict"] == "directional"
    assert s["delta_yes_mean"] == pytest.approx(0.5)
    assert s["delta_no_mean"] == pytest.approx(-0.5)
    assert s["n"] == 2
    assert s["n_yes"] == 1 and s["n_no"] == 1
    assert s["firing_rate"] == pytest.approx(1.0)


def test_build_summary_detects_a_generic_pro_yes_bias(diag):
    """The Point-2 artefact: the margin shifts toward yes regardless of truth."""
    per_question = [
        {"expected": "yes", "margins": {"baseline": 0.0, "memvr": 0.3}, "fired": {"memvr": True}},
        {"expected": "no", "margins": {"baseline": 0.0, "memvr": 0.2}, "fired": {"memvr": False}},
    ]
    s = diag.build_summary(per_question, ["baseline", "memvr"])["memvr"]
    assert s["verdict"] == "generic_bias"
    assert s["firing_rate"] == pytest.approx(0.5)
    assert s["delta_fired_mean"] == pytest.approx(0.3)
    assert s["delta_not_fired_mean"] == pytest.approx(0.2)


def test_build_summary_detects_a_generic_pro_no_bias(diag):
    per_question = [
        {"expected": "yes", "margins": {"baseline": 0.0, "memvr": -0.3}},
        {"expected": "no", "margins": {"baseline": 0.0, "memvr": -0.2}},
    ]
    s = diag.build_summary(per_question, ["baseline", "memvr"])["memvr"]
    assert s["verdict"] == "generic_bias"
    # No firing info supplied -> firing rate is NaN, not a silent 0.
    assert math.isnan(s["firing_rate"])


def test_build_summary_skips_baseline_and_covers_every_arm(diag):
    per_question = [
        {"expected": "yes", "margins": {"baseline": 0.0, "memvr": 0.1, "sparc_memvr": 0.2}},
    ]
    s = diag.build_summary(per_question, ["baseline", "memvr", "sparc_memvr"])
    assert set(s) == {"memvr", "sparc_memvr"}


def test_build_summary_ignores_a_question_missing_a_condition(diag):
    per_question = [
        {"expected": "yes", "margins": {"baseline": 0.0, "memvr": 0.5}},
        {"expected": "no", "margins": {"baseline": 0.0}},  # memvr forward failed
    ]
    s = diag.build_summary(per_question, ["baseline", "memvr"])["memvr"]
    assert s["n"] == 1
    assert s["n_no"] == 0


# ---------------------------------------------------------------- hparams mapping


def _args(**overrides) -> SimpleNamespace:
    base = dict(
        alpha=1.05, beta=0.1, tau=3.0, selected_layer=20, se_layers=(0, 31),
        memvr_gamma=0.75, memvr_alpha=0.12, memvr_window=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_baseline_condition_has_no_hparams(diag):
    assert diag.hparams_for_condition("baseline", _args()) == (None, None)


def test_memvr_condition_is_memvr_only(diag):
    sparc_hp, memvr_hp = diag.hparams_for_condition("memvr", _args())
    assert sparc_hp is None
    assert (memvr_hp.gamma, memvr_hp.alpha, memvr_hp.window) == (0.75, 0.12, None)


def test_sparc_memvr_uses_original_sparc_without_adaptive_or_qcond(diag):
    sparc_hp, memvr_hp = diag.hparams_for_condition("sparc_memvr", _args())
    assert sparc_hp is not None
    assert sparc_hp.adaptive is False
    assert sparc_hp.qcond is False
    assert sparc_hp.alpha == 1.05
    assert memvr_hp is not None


def test_memvr_window_override_flows_into_the_hparams(diag):
    _, memvr_hp = diag.hparams_for_condition("memvr", _args(memvr_window=[4, 12]))
    assert memvr_hp.window == (4, 12)


def test_unknown_condition_is_rejected(diag):
    with pytest.raises(ValueError, match="condition"):
        diag.hparams_for_condition("nonsense", _args())


# ---------------------------------------------------------------- CLI flags


def test_flag_defaults(diag):
    args = diag.build_parser().parse_args(["--config", "x.yaml"])
    assert args.limit == 300
    assert args.seed == 0
    assert args.entropy_profile == 20
    assert args.memvr_gamma == 0.75
    assert args.memvr_alpha == 0.12
    assert args.memvr_window is None
    assert args.conditions == list(diag.DIAG_CONDITIONS)


def test_conditions_flag_parses_a_subset(diag):
    args = diag.build_parser().parse_args(
        ["--config", "x.yaml", "--conditions", "baseline", "memvr"]
    )
    assert args.conditions == ["baseline", "memvr"]


def test_memvr_window_flag_parses_two_ints(diag):
    args = diag.build_parser().parse_args(["--config", "x.yaml", "--memvr-window", "4", "12"])
    assert args.memvr_window == [4, 12]


def test_entropy_profile_flag_parses(diag):
    args = diag.build_parser().parse_args(["--config", "x.yaml", "--entropy-profile", "5"])
    assert args.entropy_profile == 5
