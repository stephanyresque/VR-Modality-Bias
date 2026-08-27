"""Tests for :mod:`vr_modality_bias.data.annotations`.

Two ground-truth variants share one JSON Lines contract: objects per image for
the description track, composed questions decomposed into verifiable components
for the question track. The component type is validated at construction because
a typo there would silently produce a wrong per-type aggregation downstream.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vr_modality_bias.data.annotations import (
    COMPONENT_TYPES,
    ObjectAnnotation,
    QuestionAnnotation,
    QuestionComponent,
    read_object_annotations,
    read_question_annotations,
    write_object_annotations,
    write_question_annotations,
)


# ---------------------------------------------------------------- objects


def test_object_annotation_round_trip(tmp_path: Path):
    path = tmp_path / "objects.jsonl"
    records = [
        ObjectAnnotation(
            image_id="adt_seq07_000123",
            objects=("mug", "keyboard", "chair"),
        ),
        ObjectAnnotation(image_id="adt_seq07_000124", objects=("desk",)),
    ]

    assert write_object_annotations(records, path) == 2
    assert read_object_annotations(path) == records


def test_objects_are_normalised_to_a_tuple():
    record = ObjectAnnotation(image_id="x", objects=["lamp", "sofa"])

    assert record.objects == ("lamp", "sofa"), (
        "JSON gives back a list; without normalisation a written record and "
        "the record read back from it would not compare equal."
    )


def test_blank_lines_are_skipped(tmp_path: Path):
    path = tmp_path / "objects.jsonl"
    path.write_text(
        '{"image_id": "x", "objects": ["lamp"]}\n'
        "\n"
        '{"image_id": "y", "objects": ["sofa"]}\n',
        encoding="utf-8",
    )

    assert len(read_object_annotations(path)) == 2


def test_malformed_line_names_the_file_and_the_line(tmp_path: Path):
    path = tmp_path / "objects.jsonl"
    path.write_text(
        '{"image_id": "x", "objects": ["lamp"]}\n' "{ not json\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as excinfo:
        read_object_annotations(path)

    message = str(excinfo.value)
    assert str(path) in message
    assert "line 2" in message


def test_a_missing_required_field_names_the_file_and_the_line(tmp_path: Path):
    path = tmp_path / "objects.jsonl"
    path.write_text('{"image_id": "x"}\n', encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        read_object_annotations(path)

    message = str(excinfo.value)
    assert str(path) in message
    assert "line 1" in message


# ---------------------------------------------------------------- questions


def test_question_annotation_round_trip(tmp_path: Path):
    path = tmp_path / "questions.jsonl"
    records = [
        QuestionAnnotation(
            image_id="odi_0042",
            question_id="odi_0042_q1",
            question_text=(
                "Is there a sofa in the room, how many lamps are there, and is "
                "the window to the left of the door?"
            ),
            components=(
                QuestionComponent("existence", "Is there a sofa?", "yes"),
                QuestionComponent("count", "How many lamps are there?", 2),
                QuestionComponent(
                    "direction", "Is the window left of the door?", "left"
                ),
            ),
        ),
    ]

    assert write_question_annotations(records, path) == 1
    assert read_question_annotations(path) == records


def test_components_are_rebuilt_as_dataclasses_on_read(tmp_path: Path):
    path = tmp_path / "questions.jsonl"
    write_question_annotations(
        [
            QuestionAnnotation(
                image_id="x",
                question_id="x_q1",
                question_text="Is there a sofa?",
                components=(QuestionComponent("existence", "Is there a sofa?", "yes"),),
            )
        ],
        path,
    )

    component = read_question_annotations(path)[0].components[0]

    assert isinstance(component, QuestionComponent)
    assert component.component_type == "existence"
    assert component.answer == "yes"


def test_every_declared_component_type_is_accepted():
    # The legacy exp 1 types, the ten ODI types, and the six OmniCoT types.
    assert set(COMPONENT_TYPES) >= {"existence", "count", "direction"}
    assert set(COMPONENT_TYPES) >= {
        "ocr", "object_attribute", "human_attribute",
        "direction_ego", "direction_allo", "direction_rel",
        "scene_simulation", "odi_reasoning",
    }
    assert set(COMPONENT_TYPES) >= {"mot", "rac", "moi", "mdi", "ptm", "rtm"}
    for component_type in COMPONENT_TYPES:
        component = QuestionComponent(component_type, "Is there a sofa?", "yes")
        assert component.component_type == component_type


def test_a_component_type_outside_the_declared_set_is_rejected():
    with pytest.raises(ValueError, match="colour"):
        QuestionComponent("colour", "What colour is the sofa?", "red")


def test_an_invalid_component_type_in_a_file_names_the_file_and_the_line(
    tmp_path: Path,
):
    path = tmp_path / "questions.jsonl"
    path.write_text(
        '{"image_id": "x", "question_id": "x_q1", "question_text": "?", '
        '"components": [{"component_type": "existence", '
        '"question": "Is there a sofa?", "answer": "yes"}]}\n'
        '{"image_id": "y", "question_id": "y_q1", "question_text": "?", '
        '"components": [{"component_type": "colour", '
        '"question": "What colour?", "answer": "red"}]}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as excinfo:
        read_question_annotations(path)

    message = str(excinfo.value)
    assert str(path) in message
    assert "line 2" in message
    assert "colour" in message


def test_a_component_without_its_sub_question_is_rejected(tmp_path: Path):
    """A questions.jsonl written before the judge existed must fail loudly.

    The judge grades one sub-question at a time against its own reference, so a
    component with no statement would be judged against an empty prompt. Better
    to force a re-ingestion than to score nonsense.
    """
    path = tmp_path / "questions.jsonl"
    path.write_text(
        '{"image_id": "x", "question_id": "x_q1", "question_text": "?", '
        '"components": [{"component_type": "existence", "answer": "yes"}]}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as excinfo:
        read_question_annotations(path)

    message = str(excinfo.value)
    assert "line 1" in message
    assert "question" in message


def test_the_sub_question_survives_the_round_trip(tmp_path: Path):
    path = tmp_path / "questions.jsonl"
    write_question_annotations(
        [
            QuestionAnnotation(
                image_id="x", question_id="x_q1",
                question_text="Is there a sofa? How many lamps are there?",
                components=(
                    QuestionComponent("existence", "Is there a sofa?", "yes"),
                    QuestionComponent("count", "How many lamps are there?", 2),
                ),
            )
        ],
        path,
    )

    components = read_question_annotations(path)[0].components

    assert [c.question for c in components] == [
        "Is there a sofa?", "How many lamps are there?",
    ], (
        "question_text is the concatenation handed to the model and cannot be "
        "split back apart; the judge needs each statement on its own"
    )
