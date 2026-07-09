#!/usr/bin/env python
"""Answer the POPE questions under three conditions: baseline, SPARC (alpha^c)
and SPARC adaptive. Greedy, 10 new tokens. Resumable via pope_answers.jsonl.

Run: python scripts/pope_generate.py --config configs/run_smolvlm22_short.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger
from PIL import Image
from pyprojroot import here

try:
    from vr_modality_bias.experiment.sparc import SparcHyperparams, enable_sparc
    from vr_modality_bias.metrics.pope import normalize_answer
    from vr_modality_bias.models.registry import build_model
    from vr_modality_bias.utils.config import load_config
    from vr_modality_bias.utils.device import resolve_dtype, select_device
    from vr_modality_bias.utils.seeds import derive_image_seed
except ModuleNotFoundError:
    sys.path.insert(0, str(here()))

    from src.vr_modality_bias.experiment.sparc import SparcHyperparams, enable_sparc
    from src.vr_modality_bias.metrics.pope import normalize_answer
    from src.vr_modality_bias.models.registry import build_model
    from src.vr_modality_bias.utils.config import load_config
    from src.vr_modality_bias.utils.device import resolve_dtype, select_device
    from src.vr_modality_bias.utils.seeds import derive_image_seed


CONDITIONS = ("baseline", "sparc", "adaptive")

DEFAULT_QUESTIONS = Path("data/processed/mscoco_baseline/pope_questions.jsonl")

# POPE answers are one word. 10 tokens leaves room for "Yes, there is a ..."
# without paying for a caption, and greedy keeps the three conditions
# comparable (sampling noise would swamp the SPARC effect on a binary task).
MAX_NEW_TOKENS = 10
GEN_KWARGS = {"do_sample": False, "num_beams": 1, "repetition_penalty": 1.0}


def _iso_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def load_questions(path: Path) -> list[dict]:
    """Read pope_questions.jsonl, skipping blank and malformed lines."""
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def prompt_for(entry: dict) -> str:
    """The prompt is the pre-rendered question; build_pope.py owns the template."""
    question = entry.get("question")
    if not question:
        raise ValueError(f"question row has no 'question' field: {entry!r}")
    return str(question)


def answer_key(entry: dict, condition: str) -> tuple[str, str, str, str, str]:
    """Resume key. ``(image_id, strategy, object)`` already identifies a question:
    positives are the annotated objects, negatives exclude them, so an object
    never appears twice within one (image, strategy)."""
    return (
        str(entry["image_id"]),
        str(entry["strategy"]),
        str(entry["object"]),
        str(entry["expected"]),
        str(condition),
    )


def read_done(path: Path) -> set[tuple[str, str, str, str, str]]:
    """Keys already present in pope_answers.jsonl."""
    done: set[tuple[str, str, str, str, str]] = set()
    for entry in load_questions(path):
        try:
            done.add(answer_key(entry, entry["condition"]))
        except KeyError:
            continue
    return done


def append_answer(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def group_by_image(rows: list[dict]) -> OrderedDict[str, list[dict]]:
    """Preserve file order so a resumed run walks the images the same way."""
    grouped: OrderedDict[str, list[dict]] = OrderedDict()
    for row in rows:
        grouped.setdefault(str(row["image_id"]), []).append(row)
    return grouped


def hparams_for_condition(condition: str, args) -> SparcHyperparams | None:
    """``None`` for the baseline; otherwise the SPARC settings for that arm."""
    if condition == "baseline":
        return None
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}. Known: {CONDITIONS}.")
    return SparcHyperparams(
        alpha=args.alpha,
        tau=args.tau,
        selected_layer=args.selected_layer,
        se_layers=tuple(args.se_layers),
        beta=args.beta,
        adaptive=(condition == "adaptive"),
        lam=args.lam,
        ceiling=args.ceiling,
    )


def _probe_sparc_layout(model_wrapper, image, prompt):
    """Return ``(input_len, image_positions)`` for this (image, question) prefill.

    Must be re-run per QUESTION, not per image: POPE questions have different
    token lengths, so ``input_len`` (prompt length minus image placeholders)
    changes even though the image does not.
    """
    processor = model_wrapper._processor  # noqa: SLF001
    messages = model_wrapper._build_messages(prompt, image)  # noqa: SLF001
    prefix_text = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )
    prefix_inputs = processor(text=[prefix_text], images=[image], return_tensors="pt")
    caption_start = int(prefix_inputs["input_ids"].shape[-1])
    image_token_id = int(model_wrapper._model.config.image_token_id)  # noqa: SLF001
    image_positions = (
        prefix_inputs["input_ids"][0] == image_token_id
    ).nonzero(as_tuple=True)[0]
    num_image_patches = int(image_positions.numel())
    return caption_start - num_image_patches, image_positions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True,
        help="Family config; only the model block is used (generation is fixed "
             "to greedy / 10 tokens here).")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS,
        help=f"pope_questions.jsonl from build_pope.py (default: {DEFAULT_QUESTIONS}).")
    parser.add_argument("--run-name", type=str, default="pope",
        help="Output goes to <output-root>/<run-name>/pope_answers.jsonl.")
    parser.add_argument("--output-root", type=Path, default=Path("results/runs"),
        help="Parent dir for run directories (default: results/runs).")
    parser.add_argument("--images-dir", type=Path, default=None,
        help="Override the images dir; default comes from the config's dataset block.")
    parser.add_argument("--conditions", nargs="+", choices=list(CONDITIONS),
        default=list(CONDITIONS),
        help="Which arms to generate (default: all three).")
    parser.add_argument("--limit", type=int, default=0,
        help="Cap the number of questions (0 = all). Applies before grouping.")
    # SPARC alpha^c arm. Defaults are the paper recipe (run_all.py), NOT the
    # stale header of phase3_generate.py. selected_layer / se_layers are
    # depth-dependent, so pass the per-family values as run_all.py does.
    parser.add_argument("--alpha", type=float, default=1.05, help="SPARC alpha.")
    parser.add_argument("--beta", type=float, default=0.1, help="SPARC beta.")
    parser.add_argument("--tau", type=float, default=3.0, help="SPARC tau.")
    parser.add_argument("--selected-layer", type=int, default=20,
        help="Reference layer. Per family: 15 smolvlm, 20 llava, 18 qwen.")
    parser.add_argument("--se-layers", type=int, nargs=2, default=(0, 31),
        help="SPARC se_layers (lo hi).")
    # Adaptive arm.
    parser.add_argument("--lam", type=float, default=0.0,
        help="SPARC lambda for the adaptive arm. 0 makes it a no-op (neutrality gate).")
    parser.add_argument("--ceiling", type=float, default=2.0,
        help="Saturation ceiling for the adaptive arm.")
    parser.add_argument("--overwrite", action="store_true",
        help="Delete an existing pope_answers.jsonl before starting.")
    parser.add_argument("--print-answers", action="store_true",
        help="Echo each answer to stdout.")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    # Fail before the checkpoint load if an arm is misconfigured.
    hparams = {c: hparams_for_condition(c, args) for c in args.conditions}

    run_dir = args.output_root / args.run_name
    log_file = run_dir / "logs" / "pope_generate.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_file), enqueue=True, level="INFO")

    questions = load_questions(args.questions)
    if not questions:
        logger.error(f"No questions in {args.questions}. Run scripts/build_pope.py first.")
        return 1
    if args.limit:
        questions = questions[: args.limit]

    cfg = load_config(args.config)
    model_key = str(cfg["model"]["key"])
    model_id = str(cfg["model"]["model_id"])
    dtype_str = str(cfg["model"]["dtype"])
    seed_global = int(cfg["run"]["seed_global"])
    images_dir = Path(args.images_dir or cfg["dataset"]["images_dir"])

    logger.info("=" * 70)
    logger.info(f"POPE generation, run_name={args.run_name}")
    logger.info(f"questions : {len(questions)} from {args.questions}")
    logger.info(f"conditions: {args.conditions}")
    logger.info(f"model     : {model_id} ({dtype_str})")
    logger.info(f"decoding  : greedy, max_new_tokens={MAX_NEW_TOKENS}")
    for condition, hp in hparams.items():
        logger.info(f"  {condition:<9}: {hp.as_dict() if hp else 'no SPARC'}")
    logger.info(f"run dir   : {run_dir}")
    logger.info("=" * 70)

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_params.json").write_text(
        json.dumps({
            "run_name": args.run_name,
            "questions": str(args.questions),
            "conditions": list(args.conditions),
            "model_id": model_id,
            "dtype": dtype_str,
            "max_new_tokens": MAX_NEW_TOKENS,
            "gen_kwargs": GEN_KWARGS,
            "sparc": {c: (hp.as_dict() if hp else None) for c, hp in hparams.items()},
            "timestamp_iso": _iso_now(),
        }, indent=2) + "\n",
        encoding="utf-8",
    )

    answers_path = run_dir / "pope_answers.jsonl"
    if args.overwrite and answers_path.exists():
        logger.info(f"--overwrite: removing existing {answers_path}")
        answers_path.unlink()
    done = read_done(answers_path)
    logger.info(f"Resume state: {len(done)} answers already on disk")

    model_wrapper = build_model(model_key)
    model_wrapper.model_id = model_id
    if hasattr(model_wrapper, "_dtype"):
        model_wrapper._dtype = resolve_dtype(dtype_str)  # noqa: SLF001
    device = select_device("cuda")
    logger.info(f"Loading {model_id} on {device}...")
    model_wrapper.load(device)
    logger.info(f"Model loaded. n_layers={model_wrapper.n_layers}")

    grouped = group_by_image(questions)
    total_planned = len(questions) * len(args.conditions)
    n_done = n_skipped = n_failed = 0
    t_start = time.time()

    for image_id, rows in grouped.items():
        image_path = images_dir / f"{image_id}.jpg"
        if not image_path.exists():
            logger.error(f"{image_path} missing; skipping its {len(rows)} question(s).")
            n_failed += len(rows) * len(args.conditions)
            continue
        with Image.open(image_path) as raw:
            image = raw.convert("RGB")
        seed = int(derive_image_seed(seed_global, image_id))

        for row in rows:
            prompt = prompt_for(row)
            for condition in args.conditions:
                key = answer_key(row, condition)
                if key in done:
                    n_skipped += 1
                    continue
                try:
                    hp = hparams[condition]
                    if hp is None:
                        raw_answer = model_wrapper.generate_caption(
                            image=image, prompt=prompt,
                            max_new_tokens=MAX_NEW_TOKENS, seed=seed,
                            generation_kwargs=GEN_KWARGS,
                        )
                    else:
                        # Per QUESTION, not per image: input_len moves with the
                        # question length, and the adaptive registry is sized
                        # from image_positions at prefill.
                        input_len, image_positions = _probe_sparc_layout(
                            model_wrapper, image, prompt,
                        )
                        with enable_sparc(
                            model_wrapper, hparams=hp,
                            probe_image=image, prompt=prompt,
                        ) as buffer:
                            buffer.reset()
                            buffer.update_input_len(input_len)
                            buffer.update_image_positions(image_positions)
                            raw_answer = model_wrapper.generate_caption(
                                image=image, prompt=prompt,
                                max_new_tokens=MAX_NEW_TOKENS, seed=seed,
                                generation_kwargs=GEN_KWARGS,
                            )

                    entry = {
                        "image_id": image_id,
                        "strategy": row["strategy"],
                        "object": row["object"],
                        "question": prompt,
                        "expected": row["expected"],
                        "condition": condition,
                        "answer_raw": raw_answer,
                        "answer": normalize_answer(raw_answer),
                        "seed": seed,
                        "model_id": model_id,
                        "dtype": dtype_str,
                        "max_new_tokens": MAX_NEW_TOKENS,
                        "sparc": hp.as_dict() if hp else None,
                        "timestamp_iso": _iso_now(),
                    }
                    append_answer(answers_path, entry)
                    done.add(key)
                    n_done += 1
                    if args.print_answers:
                        print(f"[{image_id}|{row['strategy']}|{condition}] "
                              f"{prompt} -> {raw_answer!r} ({entry['answer']})")
                    if n_done % 50 == 0:
                        rate = n_done / max(time.time() - t_start, 0.1)
                        remaining = max(total_planned - n_done - n_skipped, 0)
                        logger.info(
                            f"progress {n_done + n_skipped}/{total_planned}  "
                            f"ETA {remaining / max(rate, 1e-6) / 60:.1f}min"
                        )
                except Exception as exc:
                    n_failed += 1
                    logger.error(f"[{image_id}|{row['strategy']}|{condition}] FAILED: {exc}")
                    logger.error(traceback.format_exc())

    logger.info("=" * 70)
    logger.info(f"POPE generation DONE. done={n_done} skipped={n_skipped} failed={n_failed}")
    logger.info(f"elapsed={(time.time() - t_start) / 60:.1f}min")
    logger.info(f"Answers: {answers_path}")
    logger.info("=" * 70)
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover (operator-side script)
        logger.error(f"Top-level exception: {exc}")
        logger.error(traceback.format_exc())
        raise SystemExit(1)
