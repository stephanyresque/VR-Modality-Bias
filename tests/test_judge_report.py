"""Tests for scripts/judge_report.py with an injected fake judge.

Nothing here downloads or loads a model. The script takes its judge from a
factory argument precisely so the whole pipeline -- task building, resume,
invalid handling, aggregation, the written artefacts -- is exercised on CPU.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from vr_modality_bias.data.annotations import (
    QuestionAnnotation,
    QuestionComponent,
    write_question_annotations,
)

_SCRIPTS = Path(__file__).parent.parent / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"_script_{name}", _SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def judge_report():
    return _load_script("judge_report")


# ---------------------------------------------------------------- fixtures


def _questions(path: Path) -> None:
    write_question_annotations(
        [
            QuestionAnnotation(
                image_id="indoor_001",
                question_id="indoor_001_q1",
                question_text="Is there a sofa? How many lamps are there?",
                components=(
                    QuestionComponent("existence", "Is there a sofa?", "yes"),
                    QuestionComponent("count", "How many lamps are there?", 2),
                ),
            ),
            QuestionAnnotation(
                image_id="indoor_002",
                question_id="indoor_002_q1",
                question_text="Is there a chair? Where is the window?",
                components=(
                    QuestionComponent("existence", "Is there a chair?", "no"),
                    QuestionComponent("direction", "Where is the window?", "left"),
                ),
            ),
        ],
        path,
    )


def _answer(image_id, condition, answer, sparc=None) -> dict:
    return {
        "image_id": image_id,
        "question_id": f"{image_id}_q1",
        "question_text": "Is there a sofa? How many lamps are there?",
        "condition": condition,
        "alpha": None if condition == "off" else 1.05,
        "sparc": sparc,
        "answer": answer,
        "model_id": "HuggingFaceTB/SmolVLM-Instruct",
    }


def _stage(tmp_path: Path, entries=None) -> tuple[Path, Path]:
    run_dir = tmp_path / "arm1_sparc_q"
    run_dir.mkdir(parents=True, exist_ok=True)
    questions_path = tmp_path / "questions.jsonl"
    _questions(questions_path)

    rows = entries if entries is not None else [
        _answer("indoor_001", "off", "There is a sofa and two lamps above it."),
        _answer("indoor_001", "on", "There is a sofa and two lamps above it."),
        _answer("indoor_002", "off", "I can see a window."),
        _answer("indoor_002", "on", "I can see a window."),
    ]
    with (run_dir / "answers.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return run_dir, questions_path


def _factory(*responses, record=None):
    """A judge factory that replays canned strings, cycling if it runs out."""
    def factory(**kwargs):
        if record is not None:
            record.update(kwargs)
        state = {"i": 0}

        def judge_fn(prompt: str) -> str:
            reply = responses[state["i"] % len(responses)]
            state["i"] += 1
            return reply

        return judge_fn
    return factory


_OK = '{"verdict": "correct", "evidence": "two lamps"}'


def _run(judge_report, run_dir, questions_path, *extra, factory=None):
    argv = ["--run-dir", str(run_dir), "--questions", str(questions_path), *extra]
    return judge_report.main(argv, judge_factory=factory or _factory(_OK))


def _verdicts(run_dir: Path) -> list[dict]:
    path = run_dir / "verdicts.jsonl"
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


# ---------------------------------------------------------------- task building


def test_one_task_per_sub_question(judge_report, tmp_path: Path):
    run_dir, questions_path = _stage(tmp_path)
    entries = judge_report._read_jsonl(run_dir / "answers.jsonl")
    questions = judge_report.index_questions(questions_path)

    tasks = judge_report.build_tasks(entries, questions)

    assert len(tasks) == 8, "4 answers x 2 sub-questions each"


def test_a_task_carries_its_own_sub_question_and_reference(judge_report, tmp_path: Path):
    run_dir, questions_path = _stage(tmp_path)
    entries = judge_report._read_jsonl(run_dir / "answers.jsonl")
    questions = judge_report.index_questions(questions_path)

    tasks = judge_report.build_tasks(entries, questions)
    first = tasks[0]

    assert first["sub_question"] == "Is there a sofa?"
    assert first["reference_answer"] == "yes"
    assert first["component_type"] == "existence"
    assert first["component_index"] == 0


def test_the_prompt_shows_the_composed_question_as_context(judge_report, tmp_path: Path):
    """One call per sub-question, but the judge still sees the whole question.

    Without it there is no way to tell which clause of a multi-part answer was
    meant for which sub-question.
    """
    run_dir, questions_path = _stage(tmp_path)
    entries = judge_report._read_jsonl(run_dir / "answers.jsonl")
    questions = judge_report.index_questions(questions_path)

    prompt = judge_report.build_tasks(entries, questions)[0]["prompt"]

    assert "Is there a sofa? How many lamps are there?" in prompt
    assert "There is a sofa and two lamps above it." in prompt


def test_the_prompt_of_one_sub_question_hides_the_other_reference(
    judge_report, tmp_path: Path
):
    """The verdicts have to be independent, so only this reference is shown."""
    run_dir, questions_path = _stage(tmp_path)
    entries = judge_report._read_jsonl(run_dir / "answers.jsonl")
    questions = judge_report.index_questions(questions_path)

    tasks = judge_report.build_tasks(entries, questions)
    count_task = next(t for t in tasks if t["component_type"] == "count")

    assert "HUMAN REFERENCE ANSWER" in count_task["prompt"]
    reference_block = count_task["prompt"].split("HUMAN REFERENCE ANSWER")[1]
    assert reference_block.split("\n")[1].strip() == "2"


# ---------------------------------------------------------------- resume


def test_the_resume_key_names_the_sub_question(judge_report):
    key = judge_report.verdict_key("indoor_001", "indoor_001_q1", "off", 1)

    assert key == ("indoor_001", "indoor_001_q1", "off", 1)


def test_two_sub_questions_of_one_answer_are_distinct_cells(judge_report):
    assert judge_report.verdict_key("i", "q", "off", 0) != \
        judge_report.verdict_key("i", "q", "off", 1)


def test_read_done_recovers_the_written_keys(judge_report, tmp_path: Path):
    run_dir, questions_path = _stage(tmp_path)
    _run(judge_report, run_dir, questions_path)

    done = judge_report.read_done(run_dir / "verdicts.jsonl")

    assert len(done) == 8
    assert ("indoor_001", "indoor_001_q1", "off", 0) in done


def test_read_done_on_a_missing_file_is_empty(judge_report, tmp_path: Path):
    assert judge_report.read_done(tmp_path / "nothing.jsonl") == set()


def test_a_second_run_judges_nothing_new(judge_report, tmp_path: Path, capsys):
    run_dir, questions_path = _stage(tmp_path)
    _run(judge_report, run_dir, questions_path)
    capsys.readouterr()

    _run(judge_report, run_dir, questions_path)
    out = capsys.readouterr().out

    assert "judged          : 0" in out
    assert "skipped (resume): 8" in out
    assert len(_verdicts(run_dir)) == 8, "the resumed run must not duplicate lines"


def test_a_partial_verdicts_file_is_completed_not_restarted(
    judge_report, tmp_path: Path
):
    run_dir, questions_path = _stage(tmp_path)
    tasks_done = 0

    def counting_factory(**kwargs):
        def judge_fn(prompt):
            nonlocal tasks_done
            tasks_done += 1
            return _OK
        return judge_fn

    _run(judge_report, run_dir, questions_path, "--limit", "1",
         factory=counting_factory)
    assert tasks_done == 4, "one image, two conditions, two sub-questions"

    _run(judge_report, run_dir, questions_path, factory=counting_factory)

    assert tasks_done == 8, "the second run judged only the four that were missing"
    assert len(_verdicts(run_dir)) == 8


# ---------------------------------------------------------------- invalid


def test_an_unparseable_reply_is_recorded_as_invalid(judge_report, tmp_path: Path):
    run_dir, questions_path = _stage(tmp_path)

    _run(judge_report, run_dir, questions_path, factory=_factory("I think so"))

    records = _verdicts(run_dir)
    assert {r["verdict"] for r in records} == {"invalid"}
    assert all(r["judge_raw"] == "I think so" for r in records)
    assert all("no JSON object" in r["invalid_reason"] for r in records)


def test_an_invalid_reply_never_becomes_a_verdict(judge_report, tmp_path: Path):
    run_dir, questions_path = _stage(tmp_path)

    _run(judge_report, run_dir, questions_path, factory=_factory("nope"))
    results = json.loads((run_dir / "judge_results.json").read_text(encoding="utf-8"))

    for row in results["rows"]:
        assert row["n_invalid"] == row["n_subquestions"]
        assert row["n_correct"] == 0
        assert row["n_incorrect"] == 0
        assert row["n_not_addressed"] == 0
        assert row["n_all_correct"] == 0


def test_the_invalid_count_is_surfaced_on_stdout(judge_report, tmp_path: Path, capsys):
    run_dir, questions_path = _stage(tmp_path)

    _run(judge_report, run_dir, questions_path, factory=_factory("nope"))

    assert "came back invalid" in capsys.readouterr().out


def test_a_raw_reply_is_kept_only_for_the_invalid_ones(judge_report, tmp_path: Path):
    run_dir, questions_path = _stage(tmp_path)

    _run(judge_report, run_dir, questions_path, factory=_factory(_OK, "garbage"))

    records = _verdicts(run_dir)
    for record in records:
        if record["verdict"] == "invalid":
            assert record["judge_raw"] == "garbage"
        else:
            assert record["judge_raw"] == ""


# ---------------------------------------------------------------- limit


def test_the_limit_keeps_every_condition_of_the_pairs_it_keeps(
    judge_report, tmp_path: Path
):
    """Dropping one condition of a pair would destroy the OFF/ON comparison."""
    run_dir, questions_path = _stage(tmp_path)
    entries = judge_report._read_jsonl(run_dir / "answers.jsonl")

    kept = judge_report.limit_entries(entries, 1)

    assert {e["image_id"] for e in kept} == {"indoor_001"}
    assert {e["condition"] for e in kept} == {"off", "on"}


def test_no_limit_keeps_everything(judge_report, tmp_path: Path):
    run_dir, questions_path = _stage(tmp_path)
    entries = judge_report._read_jsonl(run_dir / "answers.jsonl")

    assert judge_report.limit_entries(entries, None) == entries


# ---------------------------------------------------------------- dry run


def test_a_dry_run_loads_no_model_and_writes_nothing(judge_report, tmp_path: Path):
    run_dir, questions_path = _stage(tmp_path)

    def exploding_factory(**kwargs):
        raise AssertionError("--dry-run must never reach the judge factory")

    assert _run(judge_report, run_dir, questions_path, "--dry-run",
                factory=exploding_factory) == 0

    assert not (run_dir / "verdicts.jsonl").exists()
    assert not (run_dir / "judge_results.json").exists()


def test_a_dry_run_prints_the_prompts(judge_report, tmp_path: Path, capsys):
    run_dir, questions_path = _stage(tmp_path)

    _run(judge_report, run_dir, questions_path, "--dry-run", "--dry-run-prompts", "2",
         factory=lambda **kwargs: None)
    out = capsys.readouterr().out

    assert out.count("cannot see the photograph") == 2
    assert "8 prompt(s) would be sent" in out
    assert "Nothing was written" in out


# ---------------------------------------------------------------- end to end


def test_the_happy_path_writes_the_three_artefacts(judge_report, tmp_path: Path):
    run_dir, questions_path = _stage(tmp_path)

    assert _run(judge_report, run_dir, questions_path) == 0

    assert (run_dir / "verdicts.jsonl").is_file()
    assert (run_dir / "judge_results.json").is_file()
    assert (run_dir / "judge_results.csv").is_file()


def test_one_verdict_line_per_sub_question(judge_report, tmp_path: Path):
    run_dir, questions_path = _stage(tmp_path)

    _run(judge_report, run_dir, questions_path)
    records = _verdicts(run_dir)

    assert len(records) == 8
    assert {r["component_index"] for r in records} == {0, 1}
    assert {r["component_type"] for r in records} == {"existence", "count", "direction"}


def test_the_arms_are_separate_rows(judge_report, tmp_path: Path):
    run_dir, questions_path = _stage(tmp_path, entries=[
        _answer("indoor_001", "off", "There is a sofa and two lamps."),
        _answer("indoor_001", "on", "There is a sofa and two lamps.",
                sparc={"selected_layer": 15, "alpha": 1.05, "adaptive": False,
                       "qcond": False, "conserve": False}),
    ])

    _run(judge_report, run_dir, questions_path)
    results = json.loads((run_dir / "judge_results.json").read_text(encoding="utf-8"))

    labels = {row["condition_label"] for row in results["rows"]}
    assert labels == {"off", "on sparc a=1.05 L15"}


def test_the_judge_provenance_is_recorded(judge_report, tmp_path: Path):
    run_dir, questions_path = _stage(tmp_path)
    seen: dict = {}

    _run(judge_report, run_dir, questions_path,
         "--judge-model", "some/judge", "--judge-revision", "abc123", "--seed", "7",
         factory=_factory(_OK, record=seen))

    assert seen["model_id"] == "some/judge"
    assert seen["revision"] == "abc123"
    assert seen["seed"] == 7

    results = json.loads((run_dir / "judge_results.json").read_text(encoding="utf-8"))
    assert results["judge"] == {
        "judge_model": "some/judge", "judge_revision": "abc123", "judge_seed": 7,
    }
    assert all(r["judge_model"] == "some/judge" for r in _verdicts(run_dir))


def test_length_and_degeneration_ride_along_with_the_verdicts(
    judge_report, tmp_path: Path
):
    """The counterweight: an arm cannot win by answering shorter in silence."""
    run_dir, questions_path = _stage(tmp_path, entries=[
        _answer("indoor_001", "off", "There is a sofa and two lamps above it."),
        _answer("indoor_001", "on", "sofa sofa sofa sofa"),
    ])

    _run(judge_report, run_dir, questions_path)
    results = json.loads((run_dir / "judge_results.json").read_text(encoding="utf-8"))
    by_arm = {row["condition_label"]: row for row in results["rows"]}

    assert by_arm["off"]["mean_answer_words"] == 9
    assert by_arm["off"]["rate_degenerate"] == 0.0
    assert by_arm["on α=1.05"]["mean_answer_words"] == 4
    assert by_arm["on α=1.05"]["rate_degenerate"] == 1.0


def test_the_evaluated_model_is_carried_through(judge_report, tmp_path: Path):
    run_dir, questions_path = _stage(tmp_path)

    _run(judge_report, run_dir, questions_path)
    results = json.loads((run_dir / "judge_results.json").read_text(encoding="utf-8"))

    assert all(r["model_id"] == "HuggingFaceTB/SmolVLM-Instruct" for r in results["rows"])


# ---------------------------------------------------------------- guards


def test_an_answer_without_an_annotated_question_is_fatal(judge_report, tmp_path: Path):
    run_dir, questions_path = _stage(tmp_path, entries=[
        _answer("000000000139", "off", "some answer"),
    ])

    assert _run(judge_report, run_dir, questions_path) == 1


def test_the_grounding_failure_shows_both_id_formats(
    judge_report, tmp_path: Path, capsys
):
    run_dir, questions_path = _stage(tmp_path, entries=[
        _answer("000000000139", "off", "some answer"),
    ])

    _run(judge_report, run_dir, questions_path)
    err = capsys.readouterr().err

    assert "000000000139" in err
    assert "indoor_001" in err


def test_a_missing_answers_file_is_reported(judge_report, tmp_path: Path, capsys):
    run_dir = tmp_path / "empty_run"
    run_dir.mkdir()
    questions_path = tmp_path / "questions.jsonl"
    _questions(questions_path)

    assert _run(judge_report, run_dir, questions_path) == 1
    assert "answers.jsonl" in capsys.readouterr().err


# ---------------------------------------------------------------- think


def test_the_judge_grades_the_answer_after_think_and_reports_the_block(
    judge_report, tmp_path: Path, capsys
):
    run_dir, questions_path = _stage(tmp_path, entries=[
        {**_answer("indoor_001", "off", "There is a sofa and two lamps."),
         "answer_raw": "<think>lamps</think> There is a sofa and two lamps.",
         "think": "lamps", "think_well_formed": True},
        {**_answer("indoor_002", "off", "<think>I see a window and"),
         "answer_raw": "<think>I see a window and",
         "think": "", "think_well_formed": False},
    ])
    seen_prompts: list[str] = []

    def factory(**kwargs):
        def judge_fn(prompt: str) -> str:
            seen_prompts.append(prompt)
            return _OK
        return judge_fn

    _run(judge_report, run_dir, questions_path, factory=factory)
    out = capsys.readouterr().out
    results = json.loads((run_dir / "judge_results.json").read_text(encoding="utf-8"))

    assert all("<think>" not in p.split("THE FULL ANSWER THE MODEL GENERATED:")[1]
               for p in seen_prompts[:2])
    assert "%think_ok" in out and "think_words" in out
    row = results["rows"][0]
    assert row["n_think"] == 2
    assert row["rate_think_well_formed"] == 0.5


def test_plain_runs_show_no_think_columns(judge_report, tmp_path: Path, capsys):
    run_dir, questions_path = _stage(tmp_path)

    _run(judge_report, run_dir, questions_path)

    assert "%think_ok" not in capsys.readouterr().out


def test_the_limit_follows_the_generation_order_not_the_alphabet(judge_report, tmp_path: Path):
    run_dir, questions_path = _stage(tmp_path, entries=[
        _answer("indoor_002", "off", "I can see a window."),
        _answer("indoor_002", "on", "I can see a window."),
        _answer("indoor_001", "off", "There is a sofa and two lamps."),
        _answer("indoor_001", "on", "There is a sofa and two lamps."),
    ])
    entries = judge_report._read_jsonl(run_dir / "answers.jsonl")

    kept = judge_report.limit_entries(entries, 1)

    assert {e["image_id"] for e in kept} == {"indoor_002"}
