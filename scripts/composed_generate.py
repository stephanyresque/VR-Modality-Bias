#!/usr/bin/env python
"""Composed-question generation: one answer per (image, question) pair, baseline
(SPARC OFF) vs one SPARC arm (ON), written to answers.jsonl.

The prompt carries no length instruction of any kind. Answer length has to come
from the question having several parts, not from us asking for detail; an
instruction like "describe in detail" would make the measurement meaningless.

Run: python scripts/composed_generate.py --config configs/run_smolvlm22_long.yaml \
         --questions data/processed/<set>/questions.jsonl \
         --selected-layer L --se-layers LO HI
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
import traceback
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import torch
from loguru import logger
from PIL import Image
from pyprojroot import here

try:
    from vr_modality_bias.data.annotations import read_question_annotations
    from vr_modality_bias.data.manifests import iter_manifest
    from vr_modality_bias.data.prompts import get_prompt
    from vr_modality_bias.experiment.sparc import SparcHyperparams, enable_sparc
    from vr_modality_bias.models.registry import build_model
    from vr_modality_bias.utils.config import load_config
    from vr_modality_bias.utils.device import resolve_dtype, select_device
    from vr_modality_bias.utils.seeds import derive_image_seed
except ModuleNotFoundError:
    sys.path.insert(0, str(here()))

    from src.vr_modality_bias.data.annotations import read_question_annotations
    from src.vr_modality_bias.data.manifests import iter_manifest
    from src.vr_modality_bias.data.prompts import get_prompt
    from src.vr_modality_bias.experiment.sparc import SparcHyperparams, enable_sparc
    from src.vr_modality_bias.models.registry import build_model
    from src.vr_modality_bias.utils.config import load_config
    from src.vr_modality_bias.utils.device import resolve_dtype, select_device
    from src.vr_modality_bias.utils.seeds import derive_image_seed


PROMPT_KEY = "vqa_composed"


def _iso_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def _append(jsonl_path: Path, entry: dict) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _sparc_snapshot(hparams: SparcHyperparams | None) -> dict | None:
    return hparams.as_dict() if hparams is not None else None


def compose_prompt(question_text: str) -> str:
    return get_prompt(PROMPT_KEY).format(question=question_text)


def answer_key(image_id: str, question_id: str, condition: str) -> tuple[str, str, str]:
    """Resume key. There is no length regime here: the answer length emerges
    from the question, so (image, question, condition) identifies a cell."""
    return (str(image_id), str(question_id), str(condition))


def read_done(jsonl_path: Path) -> set[tuple[str, str, str]]:
    done: set[tuple[str, str, str]] = set()
    if not jsonl_path.exists():
        return done
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            try:
                done.add(
                    answer_key(
                        entry["image_id"], entry["question_id"], entry["condition"]
                    )
                )
            except KeyError:
                continue
    return done


def assert_resume_arm_matches(jsonl_path: Path, current_sparc: dict) -> None:
    """Abort resuming into an answers.jsonl written by a different SPARC arm.

    Same hazard as the description track: the resume key does not mention the
    arm, so a second arm pointed at a populated dir would skip every ON cell in
    silence.
    """
    if not jsonl_path.exists():
        return
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("condition") != "on":
                continue
            existing = entry.get("sparc")
            if existing != current_sparc:
                raise ValueError(
                    f"{jsonl_path} already holds ON answers from a different "
                    f"SPARC arm (found sparc={existing}, current sparc="
                    f"{current_sparc}). Use a different --run-name or --overwrite."
                )


def resolve_manifest_items(
    cfg: dict,
    *,
    image_ids: list[str] | None,
    limit: int,
) -> list[tuple[str, Path]]:
    """Return the ``(image_id, image_path)`` pairs to generate for.

    Duplicated from scripts/phase3_generate.py rather than shared, on purpose:
    the shared-module boundary is decided after both loops exist.
    """
    manifest_path = Path(cfg["dataset"]["manifest_path"])
    images_dir = Path(cfg["dataset"]["images_dir"])

    if image_ids:
        by_id = {record.image_id: record for record in iter_manifest(manifest_path)}
        unknown = [i for i in image_ids if i not in by_id]
        if unknown:
            raise ValueError(
                f"--image-ids not present in {manifest_path}: {unknown}. "
                f"The manifest holds {len(by_id)} item(s)."
            )
        records = [by_id[i] for i in image_ids]
    else:
        records = list(itertools.islice(iter_manifest(manifest_path), limit))

    if not records:
        raise ValueError(f"{manifest_path} yielded no items to generate for.")

    items: list[tuple[str, Path]] = []
    for record in records:
        image_path = images_dir / record.file_name
        if not image_path.is_file():
            raise FileNotFoundError(
                f"{manifest_path} lists image_id={record.image_id!r} with "
                f"file_name={record.file_name!r}, but {image_path} does not "
                f"exist. Fix the manifest or restage the images; this run "
                f"will not quietly drop the item."
            )
        items.append((record.image_id, image_path))
    return items


def group_questions_by_image(path: Path) -> OrderedDict[str, list]:
    """``{image_id: [QuestionAnnotation, ...]}`` in file order.

    File order, not sorted: a resumed run then walks the pairs the same way and
    the progress numbers stay comparable between runs.
    """
    grouped: OrderedDict[str, list] = OrderedDict()
    for record in read_question_annotations(path):
        grouped.setdefault(record.image_id, []).append(record)
    return grouped


def pair_up(
    items: list[tuple[str, Path]], questions_by_image: "OrderedDict[str, list]"
) -> tuple[list[tuple[str, Path, object]], int]:
    """Cross the selected images with their questions.

    Returns the ``(image_id, image_path, question)`` triples and how many
    questions were dropped because their image is not in the selection — which
    is expected under ``--limit`` and is reported, not fatal. A selection that
    matches no question at all IS fatal: that is the id-mismatch signature.
    """
    selected = {image_id for image_id, _ in items}
    dropped = sum(
        len(qs) for image_id, qs in questions_by_image.items() if image_id not in selected
    )

    pairs: list[tuple[str, Path, object]] = []
    for image_id, image_path in items:
        for question in questions_by_image.get(image_id, []):
            pairs.append((image_id, image_path, question))

    if not pairs:
        raise ValueError(
            f"none of the {len(items)} selected image(s) has a question. "
            f"Compare the two id formats:\n"
            f"  manifest image_id(s): {sorted(selected)[:5]}\n"
            f"  question image_id(s): {sorted(questions_by_image)[:5]}"
        )
    return pairs, dropped


def _probe_sparc_layout(model_wrapper, image, prompt):
    """Return ``(input_len, image_positions, question_positions)`` for one prefill.

    Must be re-run per PAIR, not per image: each composed question has its own
    token length, so ``input_len`` moves even though the image does not.

    ``question_positions`` is every prompt position after the last image
    placeholder. On the description track that span holds the captioning
    instruction; here it holds the user's actual question plus the trailing
    chat-template tokens. The mechanism is identical, the content is not.
    """
    processor = model_wrapper._processor  # noqa: SLF001
    messages = model_wrapper._build_messages(prompt, image)
    prefix_text = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )
    prefix_inputs = processor(text=[prefix_text], images=[image], return_tensors="pt")
    answer_start = int(prefix_inputs["input_ids"].shape[-1])
    image_token_id = int(model_wrapper._model.config.image_token_id)  # noqa: SLF001
    image_positions = (
        prefix_inputs["input_ids"][0] == image_token_id
    ).nonzero(as_tuple=True)[0]
    num_image_patches = int(image_positions.numel())
    question_positions = torch.arange(
        int(image_positions[-1]) + 1, answer_start, dtype=torch.long
    )
    return answer_start - num_image_patches, image_positions, question_positions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True,
        help="Family config supplying the model block, run.seed_global and "
             "generation.max_new_tokens. Its task.prompt_key is ignored: this "
             f"track always uses {PROMPT_KEY!r}.")
    parser.add_argument("--questions", type=Path, required=True,
        help="JSON Lines question-annotation file (image_id, question_id, "
             "question_text, components).")
    parser.add_argument("--run-name", type=str, default="composed",
        help="Output goes to <output-root>/<run-name>/.")
    parser.add_argument("--output-root", type=Path, default=Path("results/runs"),
        help="Parent dir for run directories (default: results/runs).")
    parser.add_argument("--limit", type=int, default=50,
        help="Max images taken from the manifest (all their questions run).")
    parser.add_argument("--image-ids", type=str, nargs="+", default=None,
        help="Specific manifest image_id(s) to use, in order. Overrides the "
             "auto-pick of the first --limit entries.")
    # SPARC arm. Identical surface to scripts/phase3_generate.py so the matrix
    # arms are spelled the same way on both tracks.
    parser.add_argument("--alpha", type=float, default=1.1,
        help="SPARC alpha.")
    parser.add_argument("--tau", type=float, default=1.5,
        help="SPARC tau.")
    parser.add_argument("--selected-layer", type=int, required=True,
        help="SPARC reference layer. No default on purpose: it is an "
             "experimental result for one model, not a pipeline constant.")
    parser.add_argument("--se-layers", type=int, nargs=2, required=True,
        help="SPARC se_layers (lo hi), inclusive. No default, same reason.")
    parser.add_argument("--beta", type=float, default=0.1,
        help="SPARC beta: smooths the reference attention used for selection.")
    parser.add_argument("--adaptive", action="store_true",
        help="Replace the accumulating alpha^c reinforcement with the deficit-driven "
             "target factor capped by --ceiling. alpha is unused in this mode.")
    parser.add_argument("--lam", type=float, default=0.0,
        help="SPARC lambda, the deficit sensitivity of --adaptive.")
    parser.add_argument("--ceiling", type=float, default=2.0,
        help="SPARC saturation ceiling for --adaptive.")
    parser.add_argument("--qcond", action="store_true",
        help="Select the visual tokens at the prefill by the attention the "
             "question pays them, and freeze that selection. Requires --adaptive.")
    parser.add_argument("--qtop-frac", type=float, default=0.05,
        help="Fraction of the visual tokens --qcond keeps.")
    parser.add_argument("--conserve", action="store_true",
        help="Reallocate attention mass from the visual sinks to the qcond "
             "selection at each decode step. Requires --qcond.")
    parser.add_argument("--rho", type=float, default=0.5,
        help="Fraction of each sink's attention mass reallocated per step.")
    parser.add_argument("--sink-frac", type=float, default=0.05,
        help="Top fraction by raw question attention that is a sink candidate.")
    # Decoding. Identical for OFF and ON — no generation parameter may differ
    # between the arms, or the comparison stops being about SPARC.
    parser.add_argument("--sampling", action="store_true",
        help="Use the sampling params from the config instead of greedy.")
    parser.add_argument("--repetition-penalty", type=float, default=None,
        help="Override repetition_penalty in gen_kwargs.")
    parser.add_argument("--no-repeat-ngram-size", type=int, default=None,
        help="If set, pass no_repeat_ngram_size to generate().")
    parser.add_argument("--print-answers", action="store_true",
        help="Print each answer to stdout in addition to the log.")
    parser.add_argument("--overwrite", action="store_true",
        help="Delete an existing answers.jsonl before starting.")
    return parser


def sparc_hparams_from_args(args) -> SparcHyperparams:
    return SparcHyperparams(
        alpha=args.alpha,
        tau=args.tau,
        selected_layer=args.selected_layer,
        se_layers=tuple(args.se_layers),
        beta=args.beta,
        adaptive=args.adaptive,
        lam=args.lam,
        ceiling=args.ceiling,
        qcond=args.qcond,
        qtop_frac=args.qtop_frac,
        conserve=args.conserve,
        rho=args.rho,
        sink_frac=args.sink_frac,
    )


def main() -> int:
    args = build_parser().parse_args()

    sparc_hparams = sparc_hparams_from_args(args)

    run_dir = args.output_root / args.run_name
    log_file = run_dir / "logs" / "composed.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_file), enqueue=True, level="INFO")

    cfg = load_config(args.config)
    seed_global = int(cfg["run"]["seed_global"])
    max_new_tokens = int(cfg["generation"]["max_new_tokens"])
    model_key = str(cfg["model"]["key"])
    model_id = str(cfg["model"]["model_id"])
    dtype_str = str(cfg["model"]["dtype"])

    if args.sampling:
        gen_kwargs = {
            "do_sample": bool(cfg["generation"]["do_sample"]),
            "temperature": float(cfg["generation"]["temperature"]),
            "top_p": float(cfg["generation"]["top_p"]),
            "repetition_penalty": float(cfg["generation"]["repetition_penalty"]),
        }
    else:
        gen_kwargs = {"do_sample": False, "num_beams": 1, "repetition_penalty": 1.0}
    if args.repetition_penalty is not None:
        gen_kwargs["repetition_penalty"] = float(args.repetition_penalty)
    if args.no_repeat_ngram_size is not None:
        gen_kwargs["no_repeat_ngram_size"] = int(args.no_repeat_ngram_size)

    logger.info("=" * 70)
    logger.info(f"Composed-question generation — run_name={args.run_name}")
    logger.info(f"config  : {args.config}  (model {model_id}, {dtype_str})")
    logger.info(f"questions: {args.questions}")
    logger.info(f"prompt_key: {PROMPT_KEY} — carries no length instruction")
    logger.info(
        f"SPARC arm: alpha={args.alpha} beta={args.beta} tau={args.tau} "
        f"selected_layer={args.selected_layer} se_layers={tuple(args.se_layers)} "
        f"adaptive={args.adaptive} qcond={args.qcond} conserve={args.conserve}"
    )
    logger.info(f"decoding : {gen_kwargs}  max_new_tokens={max_new_tokens}")
    logger.info(f"run dir  : {run_dir}")
    logger.info("=" * 70)

    snapshot = {
        "run_name": args.run_name,
        "config": str(args.config),
        "questions": str(args.questions),
        "prompt_key": PROMPT_KEY,
        **sparc_hparams.as_dict(),
        "limit": args.limit,
        "max_new_tokens": max_new_tokens,
        "gen_kwargs": gen_kwargs,
        "overwrite": args.overwrite,
        "timestamp_iso": _iso_now(),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_params.json").write_text(
        json.dumps(snapshot, indent=2) + "\n", encoding="utf-8",
    )

    jsonl_path = run_dir / "answers.jsonl"
    if args.overwrite and jsonl_path.exists():
        logger.info(f"--overwrite: removing existing {jsonl_path}")
        jsonl_path.unlink()
    done = read_done(jsonl_path)
    logger.info(f"Resume state: {len(done)} cells already in {jsonl_path}")

    sparc_dict = _sparc_snapshot(sparc_hparams)
    try:
        assert_resume_arm_matches(jsonl_path, sparc_dict)
    except ValueError as exc:
        logger.error(str(exc))
        return 1

    items = resolve_manifest_items(cfg, image_ids=args.image_ids, limit=args.limit)
    questions_by_image = group_questions_by_image(args.questions)
    pairs, dropped = pair_up(items, questions_by_image)
    logger.info(
        f"{len(pairs)} (image, question) pair(s) over {len(items)} image(s)."
    )
    if dropped:
        logger.info(
            f"{dropped} question(s) belong to images outside this selection "
            f"(expected under --limit; not an error)."
        )

    model_wrapper = build_model(model_key)
    model_wrapper.model_id = model_id
    dtype = resolve_dtype(dtype_str)
    device = select_device("cuda")
    if hasattr(model_wrapper, "_dtype"):
        model_wrapper._dtype = dtype  # noqa: SLF001
    logger.info(f"Loading {model_id} ({dtype_str}) on {device}...")
    model_wrapper.load(device)
    logger.info(f"Model loaded. n_layers={model_wrapper.n_layers}")

    total_planned = 2 * len(pairs)
    cells_done = cells_skipped = cells_failed = 0
    t_start = time.time()

    for image_id, image_path, question in pairs:
        prompt = compose_prompt(question.question_text)
        # One seed per pair, shared by OFF and ON so the two are directly
        # comparable; distinct across questions so two questions on one image
        # never share a sampling trajectory.
        seed = int(derive_image_seed(seed_global, f"{image_id}::{question.question_id}"))

        with Image.open(image_path) as raw:
            image = raw.convert("RGB")

        base_entry = {
            "image_id": image_id,
            "question_id": question.question_id,
            "question_text": question.question_text,
            "seed": seed,
            "prompt_key": PROMPT_KEY,
            "model_id": model_id,
            "dtype": dtype_str,
            "max_new_tokens": max_new_tokens,
        }

        # ---------------- OFF (baseline) ----------------
        key_off = answer_key(image_id, question.question_id, "off")
        if key_off in done:
            cells_skipped += 1
        else:
            t_cell = time.time()
            try:
                answer = model_wrapper.generate_caption(
                    image=image, prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    seed=seed,
                    generation_kwargs=gen_kwargs,
                )
                _append(jsonl_path, {
                    **base_entry,
                    "condition": "off",
                    "alpha": None,
                    "sparc": None,
                    "answer": answer,
                    "timestamp_iso": _iso_now(),
                })
                done.add(key_off)
                cells_done += 1
                logger.info(
                    f"[{image_id}|{question.question_id}|off] OK  "
                    f"words={len(answer.split())}  ({time.time() - t_cell:.1f}s)  "
                    f"progress {cells_done + cells_skipped}/{total_planned}"
                )
                if args.print_answers:
                    print(f"\n── [{image_id}|{question.question_id}|OFF] ─────────")
                    print(question.question_text)
                    print(answer)
                    print()
            except Exception as exc:
                cells_failed += 1
                logger.error(f"[{image_id}|{question.question_id}|off] FAILED: {exc}")
                logger.error(traceback.format_exc())

        # ---------------- ON (SPARC) ----------------
        # SPARC is reopened per PAIR, not per image: the buffer is sized from
        # input_len, and every question has a different token length.
        key_on = answer_key(image_id, question.question_id, "on")
        if key_on in done:
            cells_skipped += 1
            continue
        t_cell = time.time()
        try:
            input_len, image_positions, question_positions = _probe_sparc_layout(
                model_wrapper, image, prompt,
            )
            with enable_sparc(
                model_wrapper, hparams=sparc_hparams,
                probe_image=image, prompt=prompt,
            ) as buffer:
                buffer.reset()
                buffer.update_input_len(input_len)
                buffer.update_image_positions(image_positions)
                if sparc_hparams.qcond:
                    buffer.update_question_positions(question_positions)
                answer = model_wrapper.generate_caption(
                    image=image, prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    seed=seed,
                    generation_kwargs=gen_kwargs,
                )
            _append(jsonl_path, {
                **base_entry,
                "condition": "on",
                "alpha": float(args.alpha),
                "tau": float(args.tau),
                "selected_layer": int(args.selected_layer),
                "se_layers": list(args.se_layers),
                "beta": float(args.beta),
                "sparc": sparc_dict,
                "answer": answer,
                "timestamp_iso": _iso_now(),
            })
            done.add(key_on)
            cells_done += 1
            logger.info(
                f"[{image_id}|{question.question_id}|on] OK  "
                f"words={len(answer.split())}  ({time.time() - t_cell:.1f}s)  "
                f"progress {cells_done + cells_skipped}/{total_planned}"
            )
            if args.print_answers:
                print(f"\n── [{image_id}|{question.question_id}|ON] ─────────")
                print(question.question_text)
                print(answer)
                print()
        except Exception as exc:
            cells_failed += 1
            logger.error(f"[{image_id}|{question.question_id}|on] FAILED: {exc}")
            logger.error(traceback.format_exc())

    logger.info("=" * 70)
    logger.info(
        f"Composed-question generation DONE.  cells_done={cells_done}  "
        f"skipped={cells_skipped}  failed={cells_failed}  "
        f"elapsed={(time.time() - t_start) / 60:.1f}min"
    )
    logger.info(f"Answers : {jsonl_path}")
    logger.info("=" * 70)
    return 0 if cells_failed == 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover (operator-side script)
        logger.error(f"Top-level exception: {exc}")
        logger.error(traceback.format_exc())
        raise SystemExit(1)
