"""Tests for :mod:`vr_modality_bias.data.vocabulary`.

The one surviving consumer is the ODI-Bench ingestion, which loads
``configs/direction_terms.json`` through here to reject direction answers that
are not a single axis. The two load-time validations matter because that table
is hand-curated: a synonym listed under two categories would pick one silently.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vr_modality_bias.data.vocabulary import Vocabulary, load_vocabulary


def _write(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _valid() -> dict:
    return {
        "name": "tiny",
        "categories": ["cat", "dog", "dining table"],
        "synonyms": {
            "cat": ["cats", "kitten"],
            "dog": ["dogs", "puppy"],
            "dining table": ["table", "desk"],
        },
    }


# ---------------------------------------------------------------- loading


def test_loads_a_valid_vocabulary(tmp_path: Path):
    vocab = load_vocabulary(_write(tmp_path, _valid()))

    assert vocab.name == "tiny"
    assert vocab.categories == ("cat", "dog", "dining table")
    assert vocab.synonyms["cat"] == ("cats", "kitten")


def test_a_category_without_synonyms_is_allowed(tmp_path: Path):
    payload = _valid()
    payload["categories"].append("sofa")

    vocab = load_vocabulary(_write(tmp_path, payload))

    assert "sofa" in vocab.categories
    assert vocab.synonym_to_category["sofa"] == "sofa"


def test_a_missing_top_level_key_is_rejected(tmp_path: Path):
    payload = _valid()
    del payload["synonyms"]

    with pytest.raises(ValueError, match="synonyms"):
        load_vocabulary(_write(tmp_path, payload))


# ---------------------------------------------------------------- index


def test_the_index_maps_every_synonym_to_its_category(tmp_path: Path):
    vocab = load_vocabulary(_write(tmp_path, _valid()))

    assert vocab.synonym_to_category["kitten"] == "cat"
    assert vocab.synonym_to_category["puppy"] == "dog"
    assert vocab.synonym_to_category["desk"] == "dining table"


def test_the_index_maps_each_category_to_itself(tmp_path: Path):
    vocab = load_vocabulary(_write(tmp_path, _valid()))

    for category in vocab.categories:
        assert vocab.synonym_to_category[category] == category


def test_the_index_belongs_to_the_instance_not_to_the_module(tmp_path: Path):
    first = load_vocabulary(_write(tmp_path, _valid()))
    other = tmp_path / "other.json"
    other.write_text(
        json.dumps(
            {"name": "other", "categories": ["lamp"], "synonyms": {"lamp": ["lamps"]}}
        ),
        encoding="utf-8",
    )

    second = load_vocabulary(other)

    assert "kitten" in first.synonym_to_category
    assert "kitten" not in second.synonym_to_category, (
        "a second vocabulary must not see the first one's synonyms; that is "
        "the bug the module-level cache in chair.py would have caused."
    )


# ---------------------------------------------------------------- validation


def test_a_synonym_key_outside_the_categories_is_rejected(tmp_path: Path):
    payload = _valid()
    payload["synonyms"]["hamster"] = ["hamsters"]

    with pytest.raises(ValueError) as excinfo:
        load_vocabulary(_write(tmp_path, payload))

    message = str(excinfo.value)
    assert "hamster" in message
    assert "not declared categories" in message


def test_the_same_synonym_under_two_categories_is_rejected(tmp_path: Path):
    payload = _valid()
    payload["synonyms"]["cat"] = ["cats", "mesa"]
    payload["synonyms"]["dining table"] = ["table", "mesa"]

    with pytest.raises(ValueError) as excinfo:
        load_vocabulary(_write(tmp_path, payload))

    message = str(excinfo.value)
    assert "mesa" in message
    assert "cat" in message and "dining table" in message, (
        "the message must name both claimants, otherwise curating a large "
        "hand-written vocabulary means hunting for the collision by eye."
    )


def test_a_rejection_names_the_offending_file(tmp_path: Path):
    payload = _valid()
    payload["synonyms"]["hamster"] = ["hamsters"]
    path = _write(tmp_path, payload)

    with pytest.raises(ValueError, match=r"vocab\.json"):
        load_vocabulary(path)


def test_the_validations_run_on_direct_construction_too():
    with pytest.raises(ValueError, match="mesa"):
        Vocabulary(
            name="tiny",
            categories=("cat", "dog"),
            synonyms={"cat": ("mesa",), "dog": ("mesa",)},
        )
