"""Tests for the per-arm labelling and ordering of the CHAIR report
(scripts/chair_report.py): ``_condition_label`` derives a compact arm label from
the ``sparc`` record, and ``_condition_sort_key`` orders arms by complexity.
"""

from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import pytest

from vr_modality_bias.experiment.sparc import SparcHyperparams

_SCRIPTS = Path(__file__).parent.parent / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"_script_{name}", _SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def chair_report():
    return _load_script("chair_report")


# The five ON arms of the incremental matrix, as their SPARC records.
def _sparc(**kwargs) -> dict:
    return SparcHyperparams(**kwargs).as_dict()


ARM_SPARC = "α^c original"
ARM_SPARC_DICT = _sparc(alpha=1.1, selected_layer=20)
ARM_ADAPTIVE_DICT = _sparc(alpha=1.0, adaptive=True, lam=0.5, ceiling=2.0, selected_layer=20)
ARM_QCOND_DICT = _sparc(
    alpha=1.0, adaptive=True, qcond=True, qtop_frac=0.05, selected_layer=20
)
ARM_CONSERVE_DICT = _sparc(
    alpha=1.0, adaptive=True, qcond=True, conserve=True, rho=0.5, sink_frac=0.05,
    selected_layer=20,
)
ARM_CONSERVE_L15_DICT = _sparc(
    alpha=1.0, adaptive=True, qcond=True, conserve=True, rho=0.5, sink_frac=0.05,
    selected_layer=15,
)


def _on(sparc: dict | None) -> dict:
    return {"condition": "on", "alpha": 1.1, "sparc": sparc}


# ---------------------------------------------------------------- labels


def test_off_label(chair_report):
    assert chair_report._condition_label({"condition": "off", "sparc": None}) == "off"


def test_legacy_on_without_sparc_keeps_the_alpha_label(chair_report):
    assert chair_report._condition_label({"condition": "on", "alpha": 1.05}) == "on α=1.05"


def test_legacy_on_with_sparc_none_keeps_the_alpha_label(chair_report):
    assert chair_report._condition_label({"condition": "on", "alpha": 1.1, "sparc": None}) == "on α=1.1"


def test_label_alpha_c_arm(chair_report):
    assert chair_report._condition_label(_on(ARM_SPARC_DICT)) == "on sparc a=1.1 L20"


def test_label_adaptive_arm(chair_report):
    assert chair_report._condition_label(_on(ARM_ADAPTIVE_DICT)) == "on adaptive lam=0.5 ceil=2 L20"


def test_label_adaptive_qcond_arm(chair_report):
    assert chair_report._condition_label(_on(ARM_QCOND_DICT)) == "on adaptive+qcond q=0.05 L20"


def test_label_adaptive_qcond_conserve_arm(chair_report):
    assert (
        chair_report._condition_label(_on(ARM_CONSERVE_DICT))
        == "on adaptive+qcond+conserve rho=0.5 s=0.05 L20"
    )


def test_label_conserve_derived_layer_differs_only_in_the_layer(chair_report):
    assert (
        chair_report._condition_label(_on(ARM_CONSERVE_L15_DICT))
        == "on adaptive+qcond+conserve rho=0.5 s=0.05 L15"
    )


def test_every_arm_label_includes_the_reference_layer(chair_report):
    for sparc in (ARM_SPARC_DICT, ARM_ADAPTIVE_DICT, ARM_QCOND_DICT, ARM_CONSERVE_DICT):
        assert chair_report._condition_label(_on(sparc)).endswith("L20")


# ---------------------------------------------------------------- ordering


def test_condition_sort_key_orders_by_increasing_complexity(chair_report):
    labels = [
        chair_report._condition_label(_on(ARM_CONSERVE_DICT)),
        "on α=1.1",
        chair_report._condition_label(_on(ARM_QCOND_DICT)),
        "off",
        chair_report._condition_label(_on(ARM_CONSERVE_L15_DICT)),
        chair_report._condition_label(_on(ARM_ADAPTIVE_DICT)),
        chair_report._condition_label(_on(ARM_SPARC_DICT)),
    ]
    random.Random(0).shuffle(labels)
    ordered = sorted(labels, key=chair_report._condition_sort_key)
    assert ordered == [
        "off",
        "on α=1.1",
        "on sparc a=1.1 L20",
        "on adaptive lam=0.5 ceil=2 L20",
        "on adaptive+qcond q=0.05 L20",
        "on adaptive+qcond+conserve rho=0.5 s=0.05 L15",  # derived layer sorts by L
        "on adaptive+qcond+conserve rho=0.5 s=0.05 L20",
    ]


def test_off_sorts_first(chair_report):
    key = chair_report._condition_sort_key
    assert key("off") < key("on α=1.1")
    assert key("off") < key("on sparc a=1.1 L20")


def test_legacy_alpha_label_sorts_with_the_sparc_arm(chair_report):
    key = chair_report._condition_sort_key
    # Both at complexity 1, ahead of the adaptive family.
    assert key("on α=1.1")[0] == 1
    assert key("on sparc a=1.1 L20")[0] == 1
    assert key("on adaptive lam=0.5 ceil=2 L20")[0] == 2
