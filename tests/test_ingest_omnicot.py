"""Tests for scripts/ingest_omnicot.py, on synthetic raw data.

The metadata structure (fields, type labels, random_objects shape) was read
off the HF viewer before the download; every assumption the script makes is
a loud failure, and these tests pin that contract: unmapped labels raise,
unrecognized anchor shapes raise, and the anchor preamble is versioned text.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

from vr_modality_bias.data.annotations import read_question_annotations
from vr_modality_bias.data.manifests import read_manifest

_SCRIPTS = Path(__file__).parent.parent / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"_script_{name}", _SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ingest():
    return _load_script("ingest_omnicot")


def _record(**overrides) -> dict:
    # Mirrors the real metadata measured on 27/08: the image is in `file_name`
    # (with the images/ prefix), cot is an empty list, and random_objects came
    # empty in the real subset. The anchor-bearing fixtures below override
    # random_objects explicitly to keep that machinery pinned.
    record = {
        "file_name": "images/balcony_0002.jpg",
        "scene_id": "balcony",
        "qa_id": "balcony_0002-1",
        "type": "multi_hop_object",
        "subtype": "",
        "question": "Starting at the plant, move two objects east. What do you reach?",
        "answer": "a wooden bench",
        "cot": [],
        "random_objects": [
            {"name": "plant", "position": [1.0, 2.0], "orientation": 90},
            {"name": "bench", "position": [3.5, 2.0], "orientation": 180},
        ],
    }
    record.update(overrides)
    return record


def _write_raw(tmp_path: Path, records: list[dict], images: list[str]) -> Path:
    raw = tmp_path / "raw"
    (raw / "real" / "images").mkdir(parents=True, exist_ok=True)
    with (raw / "real" / "metadata.jsonl").open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    for name in images:
        Image.new("RGB", (64, 32)).save(raw / "real" / "images" / name)
    return raw


def _run_ingest(ingest, raw: Path, out_dir: Path, *extra) -> int:
    args = ingest.build_parser().parse_args(
        ["--raw", str(raw), "--out", str(out_dir), *extra]
    )
    return ingest.run(args)


# ---------------------------------------------------------------- type mapping


def test_the_six_real_labels_map_to_the_paper_acronyms(ingest):
    """The six labels and their mapping were verified against the real
    metadata on 27/08; the per-label counts matched Table 12 exactly."""
    for label, acronym in [
        ("viewpoint_transform_identify", "mot"),
        ("viewpoint_transform_angle", "rac"),
        ("multi_hop_object", "moi"),
        ("multi_hop_direction", "mdi"),
        ("move_translation", "ptm"),
        ("move_turn_combined", "rtm"),
    ]:
        assert ingest.map_component_type(
            {"type": label, "subtype": ""}, source="f", lineno=1
        ) == acronym


def test_the_image_comes_from_file_name_with_the_images_prefix(ingest):
    """The real key is `file_name`, holding "images/<id>.jpg"."""
    record = {"file_name": "images/balcony_0002.jpg"}
    assert ingest.image_id_of(record, source="f", lineno=1) == "balcony_0002"


def test_a_legacy_image_key_still_resolves(ingest):
    assert ingest.image_id_of({"image": "x.jpg"}, source="f", lineno=1) == "x"


def test_a_record_without_any_image_key_raises_naming_the_keys(ingest):
    with pytest.raises(ValueError) as excinfo:
        ingest.image_id_of({"qa_id": "q1"}, source="f", lineno=9)
    message = str(excinfo.value)
    assert "f:9" in message and "file_name" in message and "qa_id" in message


def test_a_label_only_in_subtype_still_maps(ingest):
    record = {"type": "spatial", "subtype": "multi_hop_direction"}
    assert ingest.map_component_type(record, source="f", lineno=1) == "mdi"


def test_an_unmapped_label_raises_naming_both_fields(ingest):
    with pytest.raises(KeyError) as excinfo:
        ingest.map_component_type(
            {"type": "brand_new", "subtype": "also_new"}, source="f", lineno=7
        )
    message = str(excinfo.value)
    assert "brand_new" in message and "also_new" in message
    assert "TYPE_MAP" in message, "the message must say where the fix goes"


# ---------------------------------------------------------------- anchors


def test_the_anchor_preamble_renders_name_position_and_orientation(ingest):
    text = ingest.format_anchors(
        [{"name": "plant", "position": [1.0, 2.5], "orientation": 90}],
        source="f", lineno=1,
    )
    assert text.startswith(ingest.ANCHOR_HEADER)
    assert "- plant: position (1, 2.50), facing 90 degrees" in text


def test_a_dict_position_and_a_string_orientation_render(ingest):
    text = ingest.format_anchors(
        [{"name": "bench", "position": {"x": 3, "y": 4}, "orientation": "north"}],
        source="f", lineno=1,
    )
    assert "position (3, 4)" in text
    assert "facing north" in text


def test_no_anchors_means_no_preamble(ingest):
    assert ingest.format_anchors(None, source="f", lineno=1) == ""
    assert ingest.format_anchors([], source="f", lineno=1) == ""
    assert ingest.format_anchors("", source="f", lineno=1) == ""


def test_a_json_string_of_anchors_is_decoded(ingest):
    payload = json.dumps([{"name": "plant", "position": [1, 2], "orientation": 0}])
    text = ingest.format_anchors(payload, source="f", lineno=1)
    assert "- plant" in text


def test_an_anchor_without_name_raises_showing_the_record(ingest):
    with pytest.raises(ValueError) as excinfo:
        ingest.format_anchors([{"position": [1, 2]}], source="f", lineno=3)
    assert "f:3" in str(excinfo.value)


def test_an_anchor_with_only_a_name_raises_instead_of_rendering_nothing(ingest):
    with pytest.raises(ValueError):
        ingest.format_anchors([{"name": "plant"}], source="f", lineno=1)


def test_an_unexpected_container_shape_raises(ingest):
    with pytest.raises(ValueError):
        ingest.format_anchors(42, source="f", lineno=1)


def test_the_question_text_carries_the_preamble_then_the_question(ingest):
    record = _record()
    text = ingest.compose_question_text(record, source="f", lineno=1)
    assert text.startswith(ingest.ANCHOR_HEADER)
    assert text.endswith(record["question"])
    assert "\n\n" in text


# ---------------------------------------------------------------- items


def test_one_item_per_qa_with_a_single_component(ingest):
    records = [(1, _record()), (2, _record(qa_id="balcony_0002_q2",
                                           type="viewpoint_transform_angle",
                                           answer="270"))]
    items, tally = ingest.build_items(
        records, source="f", selected={"balcony_0002"}
    )

    assert len(items) == 2
    assert all(len(item.components) == 1 for item in items)
    assert [i.components[0].component_type for i in items] == ["moi", "rac"]
    assert tally["kept_by_type"] == {"moi": 1, "rac": 1}
    assert tally["with_anchors"] == 2


def test_the_component_question_is_the_bare_question_without_anchors(ingest):
    items, _ = ingest.build_items(
        [(1, _record())], source="f", selected={"balcony_0002"}
    )
    assert items[0].components[0].question == _record()["question"]
    assert items[0].question_text != items[0].components[0].question


def test_an_empty_answer_is_dropped_and_counted(ingest):
    items, tally = ingest.build_items(
        [(1, _record(answer="  "))], source="f", selected={"balcony_0002"}
    )
    assert items == []
    assert tally["dropped_empty_answer"] == 1


def test_a_duplicated_qa_id_is_disambiguated_not_clobbered(ingest):
    records = [(1, _record()), (2, _record())]
    items, _ = ingest.build_items(records, source="f", selected={"balcony_0002"})
    assert len({item.question_id for item in items}) == 2


def test_an_unselected_image_is_skipped(ingest):
    items, _ = ingest.build_items([(1, _record())], source="f", selected=set())
    assert items == []


# ---------------------------------------------------------------- end to end


def test_one_pass_produces_a_usable_dataset(ingest, tmp_path: Path):
    records = [
        _record(),
        _record(qa_id="balcony_0002_q2", type="multi_hop_direction", answer="east"),
        _record(file_name="images/office_0001.jpg", scene_id="office",
                qa_id="office_0001_q1", type="viewpoint_transform_identify",
                answer="a desk"),
    ]
    raw = _write_raw(tmp_path, records, ["balcony_0002.jpg", "office_0001.jpg"])
    out_dir = tmp_path / "processed"

    assert _run_ingest(ingest, raw, out_dir) == 0

    manifest = read_manifest(out_dir / "manifest.jsonl")
    questions = read_question_annotations(out_dir / "questions.jsonl")
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))

    assert [r.image_id for r in manifest] == ["balcony_0002", "office_0001"]
    assert all(r.dataset == "omnicot" for r in manifest)
    assert len(questions) == 3
    assert meta["items_written"] == 3
    assert meta["kept_by_type"] == {"moi": 1, "mdi": 1, "mot": 1}
    assert meta["anchor_header"] == ingest.ANCHOR_HEADER, (
        "the preamble is part of the method and must be recorded with the data"
    )
    assert (out_dir / "images" / "balcony_0002.jpg").is_file()
    assert (raw / "real" / "images" / "balcony_0002.jpg").is_file(), (
        "raw stays untouched"
    )


def test_a_metadata_image_without_a_file_is_fatal(ingest, tmp_path: Path):
    raw = _write_raw(tmp_path, [_record()], [])
    assert _run_ingest(ingest, raw, tmp_path / "processed") == 1


def test_the_image_limit_keeps_whole_images_with_all_their_qas(ingest, tmp_path: Path):
    records = [
        _record(),
        _record(qa_id="balcony_0002_q2", type="multi_hop_direction", answer="east"),
        _record(file_name="images/office_0001.jpg", scene_id="office",
                qa_id="office_0001_q1", type="viewpoint_transform_identify",
                answer="a desk"),
    ]
    raw = _write_raw(tmp_path, records, ["balcony_0002.jpg", "office_0001.jpg"])
    out_dir = tmp_path / "processed"

    assert _run_ingest(ingest, raw, out_dir, "--limit", "1", "--seed", "0") == 0

    questions = read_question_annotations(out_dir / "questions.jsonl")
    image_ids = {q.image_id for q in questions}
    assert len(image_ids) == 1, "the limit is in images, never in loose QAs"
    if "balcony_0002" in image_ids:
        assert len(questions) == 2


def test_a_dry_run_writes_nothing(ingest, tmp_path: Path):
    raw = _write_raw(tmp_path, [_record()], ["balcony_0002.jpg"])
    out_dir = tmp_path / "processed"

    assert _run_ingest(ingest, raw, out_dir, "--dry-run") == 0

    assert not out_dir.exists()


def test_inspect_prints_labels_and_writes_nothing(ingest, tmp_path: Path, capsys):
    raw = _write_raw(
        tmp_path,
        [_record(), _record(qa_id="q2", type="never_seen_label")],
        ["balcony_0002.jpg"],
    )
    out_dir = tmp_path / "processed"

    args = ingest.build_parser().parse_args(
        ["--raw", str(raw), "--out", str(out_dir), "--inspect", "1"]
    )
    assert ingest.run(args) == 0

    output = capsys.readouterr().out
    assert "never_seen_label" in output
    assert "UNMAPPED" in output, "inspect must point at the labels TYPE_MAP lacks"
    assert not out_dir.exists()
