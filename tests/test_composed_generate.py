"""Tests for scripts/composed_generate.py.

Generation needs a GPU and is validated on the DGX. What is checked here is the
plumbing around it: reading the question file, pairing it with the manifest, the
resume key, and the CLI. Plus one guard that is not plumbing at all — the prompt
must not ask for a long answer, or the whole measurement stops meaning anything.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from vr_modality_bias.data.annotations import (
    QuestionAnnotation,
    QuestionComponent,
    write_question_annotations,
)
from vr_modality_bias.data.manifests import ImageRecord, write_manifest

_SCRIPTS = Path(__file__).parent.parent / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"_script_{name}", _SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def composed():
    return _load_script("composed_generate")


_LAYERS = ["--selected-layer", "15", "--se-layers", "0", "23"]


def _question(image_id: str, question_id: str, text: str = "Are there chairs?") -> QuestionAnnotation:
    return QuestionAnnotation(
        image_id=image_id,
        question_id=question_id,
        question_text=text,
        components=(QuestionComponent("existence", "Is there a chair?", "yes"),),
    )


def _stage(tmp_path: Path, records: list[ImageRecord]) -> dict:
    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = tmp_path / "manifest.jsonl"
    write_manifest(records, manifest_path)
    for record in records:
        path = images_dir / record.file_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not a real image")
    return {
        "dataset": {
            "manifest_path": str(manifest_path),
            "images_dir": str(images_dir),
        }
    }


def _record(image_id: str) -> ImageRecord:
    return ImageRecord(image_id, f"{image_id}.png", 8, 8, "test")


# ---------------------------------------------------------------- the prompt


_BANNED = (
    "detail", "detailed", "long", "longer", "verbose", "thorough", "rich",
    "elaborate", "comprehensive", "paragraph", "sentences", "extensive",
    "in depth", "in-depth", "as much as", "exhaustive",
)


def test_the_prompt_never_asks_for_a_long_answer():
    """The length has to emerge from the question having several parts.

    If the prompt asks for detail, the experiment measures our instruction
    rather than the model's behaviour, and the result is indefensible.
    """
    from vr_modality_bias.data.prompts import get_prompt

    text = get_prompt("vqa_composed").lower()

    for word in _BANNED:
        assert word not in text, f"the composed prompt must not say {word!r}: {text!r}"


def test_the_prompt_carries_the_question_and_nothing_else(composed):
    prompt = composed.compose_prompt("Are there chairs? How many, and where?")

    assert "Are there chairs? How many, and where?" in prompt
    assert "{question}" not in prompt, "the template placeholder must be rendered"
    assert len(prompt.splitlines()[0]) < 60, (
        "the instruction line should stay minimal and neutral"
    )


# ---------------------------------------------------------------- questions


def test_questions_are_grouped_by_image_in_file_order(composed, tmp_path: Path):
    path = tmp_path / "questions.jsonl"
    write_question_annotations(
        [
            _question("img_b", "b_q1"),
            _question("img_a", "a_q1"),
            _question("img_b", "b_q2"),
        ],
        path,
    )

    grouped = composed.group_questions_by_image(path)

    assert list(grouped) == ["img_b", "img_a"], "file order, not sorted"
    assert [q.question_id for q in grouped["img_b"]] == ["b_q1", "b_q2"]


def test_every_question_of_a_selected_image_becomes_a_pair(composed, tmp_path: Path):
    cfg = _stage(tmp_path, [_record("img_a"), _record("img_b")])
    path = tmp_path / "questions.jsonl"
    write_question_annotations(
        [_question("img_a", "a_q1"), _question("img_a", "a_q2"), _question("img_b", "b_q1")],
        path,
    )

    items = composed.resolve_manifest_items(cfg, image_ids=None, limit=10)
    pairs, dropped = composed.pair_up(items, composed.group_questions_by_image(path))

    assert [(i, q.question_id) for i, _, q in pairs] == [
        ("img_a", "a_q1"), ("img_a", "a_q2"), ("img_b", "b_q1"),
    ]
    assert dropped == 0


def test_questions_outside_the_selection_are_counted_not_fatal(composed, tmp_path: Path):
    cfg = _stage(tmp_path, [_record("img_a"), _record("img_b")])
    path = tmp_path / "questions.jsonl"
    write_question_annotations(
        [_question("img_a", "a_q1"), _question("img_b", "b_q1"), _question("img_b", "b_q2")],
        path,
    )

    items = composed.resolve_manifest_items(cfg, image_ids=None, limit=1)
    pairs, dropped = composed.pair_up(items, composed.group_questions_by_image(path))

    assert [q.question_id for _, _, q in pairs] == ["a_q1"]
    assert dropped == 2, "--limit dropping questions is expected, not an error"


def test_a_selection_matching_no_question_is_fatal(composed, tmp_path: Path):
    cfg = _stage(tmp_path, [_record("adt_seq07_000123")])
    path = tmp_path / "questions.jsonl"
    write_question_annotations([_question("000000000139", "q1")], path)

    with pytest.raises(ValueError) as excinfo:
        composed.pair_up(
            composed.resolve_manifest_items(cfg, image_ids=None, limit=10),
            composed.group_questions_by_image(path),
        )

    message = str(excinfo.value)
    assert "adt_seq07_000123" in message, "must show the manifest-side id"
    assert "000000000139" in message, "must show the question-side id"


# ---------------------------------------------------------------- resume


def test_the_resume_key_is_image_question_condition(composed):
    assert composed.answer_key("img_a", "q1", "off") == ("img_a", "q1", "off")


def test_two_questions_on_one_image_are_distinct_cells(composed):
    assert composed.answer_key("img_a", "q1", "on") != composed.answer_key("img_a", "q2", "on")


def test_read_done_recovers_the_written_keys(composed, tmp_path: Path):
    path = tmp_path / "answers.jsonl"
    for entry in [
        {"image_id": "img_a", "question_id": "q1", "condition": "off", "answer": "x"},
        {"image_id": "img_a", "question_id": "q1", "condition": "on", "answer": "y"},
        {"image_id": "img_a", "question_id": "q2", "condition": "off", "answer": "z"},
    ]:
        composed._append(path, entry)

    done = composed.read_done(path)

    assert done == {
        ("img_a", "q1", "off"),
        ("img_a", "q1", "on"),
        ("img_a", "q2", "off"),
    }
    assert ("img_a", "q2", "on") not in done, "the unfinished cell must be regenerated"


def test_read_done_on_a_missing_file_is_empty(composed, tmp_path: Path):
    assert composed.read_done(tmp_path / "nothing.jsonl") == set()


def test_read_done_skips_malformed_lines(composed, tmp_path: Path):
    path = tmp_path / "answers.jsonl"
    path.write_text(
        '{"image_id": "a", "question_id": "q1", "condition": "off"}\n'
        "{ not json\n"
        '{"image_id": "a"}\n',
        encoding="utf-8",
    )

    assert composed.read_done(path) == {("a", "q1", "off")}


def test_resuming_into_a_different_arm_is_rejected(composed, tmp_path: Path):
    path = tmp_path / "answers.jsonl"
    written = composed.sparc_hparams_from_args(
        composed.build_parser().parse_args([
            "--config", "c.yaml", "--questions", "q.jsonl", *_LAYERS, "--lam", "0.5",
            "--adaptive",
        ])
    ).as_dict()
    current = composed.sparc_hparams_from_args(
        composed.build_parser().parse_args([
            "--config", "c.yaml", "--questions", "q.jsonl", *_LAYERS, "--lam", "0.7",
            "--adaptive",
        ])
    ).as_dict()
    composed._append(path, {
        "image_id": "a", "question_id": "q1", "condition": "on", "sparc": written,
    })

    with pytest.raises(ValueError, match="different SPARC arm"):
        composed.assert_resume_arm_matches(path, current)


# ---------------------------------------------------------------- argparse


def test_config_and_questions_are_required(composed):
    with pytest.raises(SystemExit):
        composed.build_parser().parse_args(_LAYERS)


def test_the_layer_arguments_are_required(composed):
    with pytest.raises(SystemExit):
        composed.build_parser().parse_args(["--config", "c.yaml", "--questions", "q.jsonl"])


def test_a_minimal_invocation_parses(composed):
    args = composed.build_parser().parse_args(
        ["--config", "c.yaml", "--questions", "q.jsonl", *_LAYERS]
    )

    assert args.selected_layer == 15
    assert args.se_layers == [0, 23]
    assert args.limit == 50
    assert args.adaptive is False


def test_the_arm_flags_reach_the_hyperparameters(composed):
    args = composed.build_parser().parse_args([
        "--config", "c.yaml", "--questions", "q.jsonl", *_LAYERS,
        "--adaptive", "--lam", "0.5", "--qcond", "--conserve", "--rho", "0.25",
    ])

    hp = composed.sparc_hparams_from_args(args)

    assert (hp.adaptive, hp.qcond, hp.conserve, hp.rho) == (True, True, True, 0.25)
    assert hp.selected_layer == 15
    assert hp.se_layers == (0, 23)


def test_the_sparc_record_lands_in_the_snapshot(composed):
    hp = composed.sparc_hparams_from_args(
        composed.build_parser().parse_args(
            ["--config", "c.yaml", "--questions", "q.jsonl", *_LAYERS, "--adaptive"]
        )
    )

    assert composed._sparc_snapshot(hp) == hp.as_dict()
    assert composed._sparc_snapshot(None) is None
    json.dumps(composed._sparc_snapshot(hp))
