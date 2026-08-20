"""Tests for scripts/preflight.py.

The point of preflight is to fail in seconds instead of three hours in, and to
report EVERY problem in one pass rather than one per re-run. The id-crossing
check is the one that earns the script: if the manifest ids and the annotation
ids do not meet, generation completes and the report is empty.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

import yaml

from vr_modality_bias.data.annotations import (
    ObjectAnnotation,
    QuestionAnnotation,
    QuestionComponent,
    write_object_annotations,
    write_question_annotations,
)
from vr_modality_bias.data.manifests import ImageRecord, write_manifest

_SCRIPTS = Path(__file__).parent.parent / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"_script_{name}", _SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before executing: dataclasses resolves deferred annotations
    # through sys.modules[cls.__module__], and preflight defines one.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def preflight():
    return _load_script("preflight")


def _record(image_id: str) -> ImageRecord:
    return ImageRecord(image_id, f"{image_id}.png", 8, 8, "test")


def _stage(tmp_path: Path, ids: list[str], *, on_disk: list[str] | None = None) -> Path:
    images = tmp_path / "images"
    images.mkdir(parents=True, exist_ok=True)
    write_manifest([_record(i) for i in ids], tmp_path / "manifest.jsonl")
    for image_id in (ids if on_disk is None else on_disk):
        (images / f"{image_id}.png").write_bytes(b"x")

    cfg = {
        "run": {"name": "t", "seed_global": 42, "output_root": str(tmp_path / "runs")},
        "dataset": {
            "name": "t",
            "manifest_path": str(tmp_path / "manifest.jsonl"),
            "images_dir": str(images),
            "n_images": len(ids),
        },
        "model": {"key": "smolvlm-2.2b", "model_id": "x", "dtype": "float16"},
        "task": {"prompt_key": "caption_long"},
        "generation": {"max_new_tokens": 64, "do_sample": False,
                       "temperature": 1.0, "top_p": 1.0, "repetition_penalty": 1.0},
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


# ---------------------------------------------------------------- configs


def test_a_missing_config_is_a_problem(preflight, tmp_path: Path):
    findings, configs = preflight.check_configs([tmp_path / "nope.yaml"])

    assert not findings.ok
    assert "not found" in findings.problems[0]
    assert configs == []


def test_a_config_missing_a_block_is_a_problem(preflight, tmp_path: Path):
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump({"run": {}, "dataset": {}}), encoding="utf-8")

    findings, _ = preflight.check_configs([path])

    assert not findings.ok
    assert any("model" in p for p in findings.problems)


def test_a_good_config_passes(preflight, tmp_path: Path):
    findings, configs = preflight.check_configs([_stage(tmp_path, ["a"])])

    assert findings.ok
    assert len(configs) == 1


# ---------------------------------------------------------------- manifest


def test_a_manifest_smaller_than_the_limit_is_a_problem(preflight, tmp_path: Path):
    cfg_path = _stage(tmp_path, ["a", "b"])
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    findings, _ = preflight.check_manifest(cfg, 10)

    assert not findings.ok
    assert "asks for 10" in findings.problems[0]


def test_the_limit_truncates_the_records_returned(preflight, tmp_path: Path):
    cfg = yaml.safe_load(_stage(tmp_path, ["a", "b", "c"]).read_text(encoding="utf-8"))

    findings, records = preflight.check_manifest(cfg, 2)

    assert findings.ok
    assert [r.image_id for r in records] == ["a", "b"]


# ---------------------------------------------------------------- images


def test_a_missing_image_is_a_problem_naming_the_item(preflight, tmp_path: Path):
    cfg_path = _stage(tmp_path, ["a", "b"], on_disk=["a"])
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    _, records = preflight.check_manifest(cfg, 2)

    findings = preflight.check_images(cfg, records)

    assert not findings.ok
    assert "b" in findings.problems[0]


def test_all_images_present_passes(preflight, tmp_path: Path):
    cfg_path = _stage(tmp_path, ["a", "b"])
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    _, records = preflight.check_manifest(cfg, 2)

    assert preflight.check_images(cfg, records).ok


# ---------------------------------------------------------------- id crossing


def test_disjoint_id_spaces_are_the_loudest_problem(preflight):
    findings = preflight.check_ids_cross(
        {"adt_seq07_000123", "adt_seq07_000124"},
        {"000000000139", "000000000285"},
        what="annotations",
    )

    assert not findings.ok
    message = findings.problems[0]
    assert "NO id in the manifest" in message
    assert "adt_seq07_000123" in message, "must sample the manifest side"
    assert "000000000139" in message, "must sample the annotation side"


def test_a_full_match_reports_the_counts(preflight):
    findings = preflight.check_ids_cross({"a", "b"}, {"a", "b"}, what="annotations")

    assert findings.ok
    assert "2 matched" in findings.notes[0]


def test_a_manifest_item_without_annotation_is_a_problem(preflight):
    findings = preflight.check_ids_cross({"a", "b"}, {"a"}, what="annotations")

    assert not findings.ok
    assert "'b'" in findings.problems[0]


def test_an_annotation_without_a_manifest_item_is_only_a_note(preflight):
    findings = preflight.check_ids_cross({"a"}, {"a", "b"}, what="annotations")

    assert findings.ok, "extra ground truth is expected under a limit"
    assert any("no manifest item" in n for n in findings.notes)


# ---------------------------------------------------------------- output root


def test_a_writable_output_root_passes(preflight, tmp_path: Path):
    assert preflight.check_output_root(tmp_path / "runs", min_free_gb=0.0).ok


def test_an_impossible_free_space_floor_is_a_problem(preflight, tmp_path: Path):
    findings = preflight.check_output_root(tmp_path / "runs", min_free_gb=1e9)

    assert not findings.ok
    assert "free" in findings.problems[0]


# ---------------------------------------------------------------- arms


def test_an_unknown_arm_is_rejected(preflight):
    findings = preflight.check_arms(["arm1_sparc", "arm9_bogus"])

    assert not findings.ok
    assert "arm9_bogus" in findings.problems[0]


def test_the_pilot_arms_are_all_known(preflight):
    assert preflight.check_arms(["baseline", "arm1_sparc", "arm5_reflayer"]).ok


def test_baseline_is_a_known_arm(preflight):
    assert "baseline" in preflight.KNOWN_ARMS


# ---------------------------------------------------------------- whole run


def test_every_problem_is_reported_in_one_pass(preflight, tmp_path: Path):
    """The whole point: one run of preflight, the complete list of problems."""
    cfg_path = _stage(tmp_path, ["a", "b"], on_disk=["a"])
    write_object_annotations(
        [ObjectAnnotation(image_id="zzz_other", objects=("chair",))],
        tmp_path / "annotations.jsonl",
    )

    args = preflight.build_parser().parse_args([
        "--config", str(cfg_path),
        "--limit", "5",
        "--annotations", str(tmp_path / "annotations.jsonl"),
        "--output-root", str(tmp_path / "runs"),
        "--arms", "arm1_sparc", "arm9_bogus",
        "--skip-gpu-check",
    ])
    findings = preflight.run_checks(args)

    joined = "\n".join(findings.problems)
    assert "asks for 5" in joined, "manifest too small"
    assert "not on disk" in joined, "missing image file"
    assert "arm9_bogus" in joined, "unknown arm"
    assert len(findings.problems) >= 3, (
        "preflight must not stop at the first problem; each re-run costs a "
        "round trip to the remote box"
    )


def test_a_clean_setup_passes_end_to_end(preflight, tmp_path: Path):
    cfg_path = _stage(tmp_path, ["a", "b"])
    write_object_annotations(
        [ObjectAnnotation(image_id=i, objects=("chair",)) for i in ("a", "b")],
        tmp_path / "annotations.jsonl",
    )
    write_question_annotations(
        [
            QuestionAnnotation(
                image_id=i, question_id=f"{i}_q1", question_text="Are there chairs?",
                components=(QuestionComponent("existence", "Is there a chair?", "yes"),),
            )
            for i in ("a", "b")
        ],
        tmp_path / "questions.jsonl",
    )

    args = preflight.build_parser().parse_args([
        "--config", str(cfg_path),
        "--limit", "2",
        "--annotations", str(tmp_path / "annotations.jsonl"),
        "--questions", str(tmp_path / "questions.jsonl"),
        "--output-root", str(tmp_path / "runs"),
        "--min-free-gb", "0",
        "--arms", "baseline", "arm1_sparc", "arm5_reflayer",
        "--skip-gpu-check",
    ])
    findings = preflight.run_checks(args)

    assert findings.ok, findings.problems


def test_the_question_ids_are_crossed_too(preflight, tmp_path: Path):
    cfg_path = _stage(tmp_path, ["a"])
    write_object_annotations(
        [ObjectAnnotation(image_id="a", objects=("chair",))],
        tmp_path / "annotations.jsonl",
    )
    write_question_annotations(
        [QuestionAnnotation(
            image_id="wrong_format", question_id="q1", question_text="?",
            components=(QuestionComponent("existence", "Is there a chair?", "yes"),),
        )],
        tmp_path / "questions.jsonl",
    )

    args = preflight.build_parser().parse_args([
        "--config", str(cfg_path), "--limit", "1",
        "--annotations", str(tmp_path / "annotations.jsonl"),
        "--questions", str(tmp_path / "questions.jsonl"),
        "--output-root", str(tmp_path / "runs"),
        "--min-free-gb", "0", "--skip-gpu-check",
    ])
    findings = preflight.run_checks(args)

    assert any("questions" in p and "NO id" in p for p in findings.problems)


# ---------------------------------------------------------------- one-track datasets
#
# Neither real dataset carries all four artefacts: ADT has objects and no
# questions, ODI-Bench has questions and no per-image object list. Demanding
# both made each of them unrunnable without inventing a path.


def _adt_shaped(tmp_path: Path) -> list[str]:
    cfg_path = _stage(tmp_path, ["a", "b"])
    write_object_annotations(
        [ObjectAnnotation(image_id=i, objects=("chair",)) for i in ("a", "b")],
        tmp_path / "annotations.jsonl",
    )
    return [
        "--config", str(cfg_path), "--limit", "2",
        "--annotations", str(tmp_path / "annotations.jsonl"),
        "--output-root", str(tmp_path / "runs"),
        "--min-free-gb", "0", "--skip-gpu-check",
    ]


def _odi_shaped(tmp_path: Path) -> list[str]:
    cfg_path = _stage(tmp_path, ["a", "b"])
    write_question_annotations(
        [
            QuestionAnnotation(
                image_id=i, question_id=f"{i}_q1", question_text="Are there chairs?",
                components=(QuestionComponent("existence", "Is there a chair?", "yes"),),
            )
            for i in ("a", "b")
        ],
        tmp_path / "questions.jsonl",
    )
    return [
        "--config", str(cfg_path), "--limit", "2",
        "--questions", str(tmp_path / "questions.jsonl"),
        "--output-root", str(tmp_path / "runs"),
        "--min-free-gb", "0", "--skip-gpu-check",
    ]


def test_an_object_only_dataset_passes_without_questions(preflight, tmp_path: Path):
    args = preflight.build_parser().parse_args(_adt_shaped(tmp_path))

    findings = preflight.run_checks(args)

    assert findings.ok, findings.problems


def test_a_question_only_dataset_passes(preflight, tmp_path: Path):
    args = preflight.build_parser().parse_args(_odi_shaped(tmp_path))

    findings = preflight.run_checks(args)

    assert findings.ok, findings.problems


def test_neither_ground_truth_is_a_problem(preflight, tmp_path: Path):
    cfg_path = _stage(tmp_path, ["a"])
    args = preflight.build_parser().parse_args([
        "--config", str(cfg_path), "--limit", "1",
        "--output-root", str(tmp_path / "runs"),
        "--min-free-gb", "0", "--skip-gpu-check",
    ])

    findings = preflight.run_checks(args)

    assert not findings.ok
    assert any("neither --annotations nor --questions" in p for p in findings.problems)


# ---------------------------------------------------------------- no scoring
#
# A diagnostic-only run generates and measures attention; there is nothing for
# it to grade, so demanding a ground truth would make that run impossible.


def test_no_scoring_turns_the_missing_ground_truth_into_a_note(preflight, tmp_path: Path):
    cfg_path = _stage(tmp_path, ["a"])
    args = preflight.build_parser().parse_args([
        "--config", str(cfg_path), "--limit", "1",
        "--output-root", str(tmp_path / "runs"),
        "--min-free-gb", "0", "--skip-gpu-check", "--no-scoring",
    ])

    findings = preflight.run_checks(args)

    assert findings.ok, findings.problems
    assert any("no scoring stage selected" in n for n in findings.notes), (
        "the absence still has to be visible in the report, just not fatal"
    )


def test_no_scoring_does_not_silence_a_real_problem(preflight, tmp_path: Path):
    cfg_path = _stage(tmp_path, ["a", "b"], on_disk=["a"])
    args = preflight.build_parser().parse_args([
        "--config", str(cfg_path), "--limit", "2",
        "--output-root", str(tmp_path / "runs"),
        "--min-free-gb", "0", "--skip-gpu-check", "--no-scoring",
    ])

    findings = preflight.run_checks(args)

    assert not findings.ok
    assert any("not on disk" in p for p in findings.problems), (
        "--no-scoring waives the ground truth, nothing else"
    )


def test_the_id_crossing_still_runs_for_the_side_that_was_given(preflight, tmp_path: Path):
    cfg_path = _stage(tmp_path, ["a"])
    write_question_annotations(
        [QuestionAnnotation(
            image_id="wrong_format", question_id="q1", question_text="?",
            components=(QuestionComponent("existence", "Is there a chair?", "yes"),),
        )],
        tmp_path / "questions.jsonl",
    )
    args = preflight.build_parser().parse_args([
        "--config", str(cfg_path), "--limit", "1",
        "--questions", str(tmp_path / "questions.jsonl"),
        "--output-root", str(tmp_path / "runs"),
        "--min-free-gb", "0", "--skip-gpu-check",
    ])

    findings = preflight.run_checks(args)

    assert any("NO id in the manifest" in p and "questions" in p
               for p in findings.problems)
