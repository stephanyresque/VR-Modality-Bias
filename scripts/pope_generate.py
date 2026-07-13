#!/usr/bin/env python
"""Answer the POPE questions under several conditions (baseline, SPARC alpha^c,
adaptive, qcond, memvr, sparc_memvr). Greedy, 10 new tokens. Resumable via
pope_answers.jsonl.

Run: python scripts/pope_generate.py --config configs/run_smolvlm22_short.yaml
"""

from __future__ import annotations

import argparse
import contextlib
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
    from vr_modality_bias.experiment.memvr import (
        MemVRHyperparams,
        enable_memvr,
        resolve_effective_window,
    )
    from vr_modality_bias.experiment.sparc import SparcHyperparams, enable_sparc
    from vr_modality_bias.metrics.pope import normalize_answer
    from vr_modality_bias.models.registry import build_model
    from vr_modality_bias.utils.config import load_config
    from vr_modality_bias.utils.device import resolve_dtype, select_device
    from vr_modality_bias.utils.seeds import derive_image_seed
except ModuleNotFoundError:
    sys.path.insert(0, str(here()))

    from src.vr_modality_bias.experiment.memvr import (
        MemVRHyperparams,
        enable_memvr,
        resolve_effective_window,
    )
    from src.vr_modality_bias.experiment.sparc import SparcHyperparams, enable_sparc
    from src.vr_modality_bias.metrics.pope import normalize_answer
    from src.vr_modality_bias.models.registry import build_model
    from src.vr_modality_bias.utils.config import load_config
    from src.vr_modality_bias.utils.device import resolve_dtype, select_device
    from src.vr_modality_bias.utils.seeds import derive_image_seed


CONDITIONS = ("baseline", "sparc", "adaptive", "qcond", "memvr", "sparc_memvr")

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


def sparc_hparams_for_condition(condition: str, args) -> SparcHyperparams | None:
    """The SPARC arm of a condition, or ``None`` when it uses no attention patch.

    ``sparc_memvr`` uses the ORIGINAL alpha^c SPARC (no adaptive, no qcond); the
    ``adaptive`` / ``qcond`` arms keep Point 1 / Point 2 on. ``baseline`` and
    ``memvr`` carry no SPARC.
    """
    if condition in ("baseline", "memvr"):
        return None
    return SparcHyperparams(
        alpha=args.alpha,
        tau=args.tau,
        selected_layer=args.selected_layer,
        se_layers=tuple(args.se_layers),
        beta=args.beta,
        adaptive=condition in ("adaptive", "qcond"),
        lam=args.lam,
        ceiling=args.ceiling,
        qcond=(condition == "qcond"),
        qtop_frac=args.qtop_frac,
    )


def memvr_hparams_for_condition(condition: str, args) -> MemVRHyperparams | None:
    """The MemVR arm of a condition, or ``None`` when the condition has no MemVR."""
    if condition not in ("memvr", "sparc_memvr"):
        return None
    window = tuple(args.memvr_window) if args.memvr_window else None
    return MemVRHyperparams(
        gamma=args.memvr_gamma, alpha=args.memvr_alpha, window=window
    )


def hparams_for_condition(
    condition: str, args
) -> tuple[SparcHyperparams | None, MemVRHyperparams | None]:
    """Map a condition to its ``(sparc_hp, memvr_hp)`` pair.

    baseline -> (None, None); sparc / adaptive / qcond -> (SPARC, None);
    memvr -> (None, MemVR); sparc_memvr -> (original alpha^c SPARC, MemVR).
    """
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}. Known: {CONDITIONS}.")
    return (
        sparc_hparams_for_condition(condition, args),
        memvr_hparams_for_condition(condition, args),
    )


# The MemVR instrumentation columns, ordered; None on the arms without MemVR.
_MEMVR_COLUMNS = (
    "memvr_fired",
    "memvr_fired_in_prefill",
    "memvr_fire_layer",
    "memvr_fire_entropy",
    "memvr_n_fires",
)


def _memvr_columns(extra: dict) -> dict:
    """Flatten the per-question MemVR instrumentation, or all-``None`` if absent."""
    mv = extra.get("memvr")
    return {k: (mv[k] if mv else None) for k in _MEMVR_COLUMNS}


def _generate_answer(model_wrapper, image, prompt, seed, sparc_hp, memvr_hp):
    """Generate one POPE answer under the ``(sparc_hp, memvr_hp)`` pair.

    The generation body lives here exactly once. SPARC and MemVR contexts are
    entered only when their hyperparameters are present, nested SPARC-outer /
    MemVR-inner, and each buffer receives its per-question updates before the
    generate call. Returns ``(raw_answer, extra)``; ``extra`` carries the qcond
    prefill selection and the MemVR instrumentation, both read inside the context
    before it exits (as the old inline qcond path did).
    """
    extra: dict = {}
    if sparc_hp is None and memvr_hp is None:
        raw_answer = model_wrapper.generate_caption(
            image=image, prompt=prompt,
            max_new_tokens=MAX_NEW_TOKENS, seed=seed,
            generation_kwargs=GEN_KWARGS,
        )
        return raw_answer, extra

    # Pure tokenisation, safe to run before opening the contexts (it does not
    # touch the patched forwards). image_positions feeds both buffers.
    input_len, image_positions, question_positions = _probe_sparc_layout(
        model_wrapper, image, prompt
    )
    with contextlib.ExitStack() as stack:
        if sparc_hp is not None:
            sparc_buffer = stack.enter_context(
                enable_sparc(
                    model_wrapper, hparams=sparc_hp, probe_image=image, prompt=prompt
                )
            )
            sparc_buffer.reset()
            sparc_buffer.update_input_len(input_len)
            sparc_buffer.update_image_positions(image_positions)
            if sparc_hp.qcond:
                sparc_buffer.update_question_positions(question_positions)
        else:
            sparc_buffer = None

        if memvr_hp is not None:
            memvr_buffer = stack.enter_context(enable_memvr(model_wrapper, memvr_hp))
            memvr_buffer.update_image_positions(image_positions)
        else:
            memvr_buffer = None

        raw_answer = model_wrapper.generate_caption(
            image=image, prompt=prompt,
            max_new_tokens=MAX_NEW_TOKENS, seed=seed,
            generation_kwargs=GEN_KWARGS,
        )

        if sparc_buffer is not None and sparc_hp.qcond:
            extra["prefill_selected"] = (
                sparc_buffer.prefill_selected_local.tolist()
                if sparc_buffer.prefill_selected_local is not None
                else None
            )
        if memvr_buffer is not None:
            extra["memvr"] = {
                "memvr_fired": memvr_buffer.n_fires_total > 0,
                "memvr_fired_in_prefill": bool(memvr_buffer.fired_in_prefill),
                "memvr_fire_layer": memvr_buffer.fire_layer,
                "memvr_fire_entropy": memvr_buffer.fire_entropy,
                "memvr_n_fires": memvr_buffer.n_fires_total,
            }
    return raw_answer, extra


def _probe_sparc_layout(model_wrapper, image, prompt):
    """Return ``(input_len, image_positions, question_positions)`` for this prefill.

    Must be re-run per QUESTION, not per image: POPE questions have different
    token lengths, so ``input_len`` (prompt length minus image placeholders)
    changes even though the image does not.

    ``question_positions`` are every prompt position after the last image
    placeholder. That deliberately includes the trailing chat-template tokens:
    the question text is what must be inside the set, and averaging over the
    rows dilutes the template tokens, which attend the image diffusely.
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
    question_positions = torch.arange(
        int(image_positions[-1]) + 1, caption_start, dtype=torch.long
    )
    return caption_start - num_image_patches, image_positions, question_positions


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
    # Question-conditioned arm.
    parser.add_argument("--qtop-frac", type=float, default=0.05,
        help="Fraction of the visual tokens the qcond arm selects at the prefill.")
    # MemVR arm (conditions memvr, sparc_memvr).
    parser.add_argument("--memvr-gamma", type=float, default=0.75,
        help="MemVR entropy threshold (normalised); 1.0 never fires.")
    parser.add_argument("--memvr-alpha", type=float, default=0.12,
        help="MemVR retracing ratio (convex-mix weight); 0.0 is a no-op.")
    parser.add_argument("--memvr-window", type=int, nargs=2, default=None,
        metavar=("START", "END"),
        help="MemVR firing window (inclusive). Omitted = depth fraction "
             "(round(L/6), round(L/2)), capped at L-2.")
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
    for condition, (shp, mhp) in hparams.items():
        logger.info(
            f"  {condition:<12}: sparc={shp.as_dict() if shp else None} "
            f"memvr={mhp.as_dict() if mhp else None}"
        )
    logger.info(f"run dir   : {run_dir}")
    logger.info("=" * 70)

    run_dir.mkdir(parents=True, exist_ok=True)

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

    # Written after the load so the MemVR window can be resolved against the
    # real layer count (the depth-fraction default needs n_layers).
    memvr_effective = {
        c: {
            "gamma": mhp.gamma,
            "alpha": mhp.alpha,
            "top_k": mhp.top_k,
            "window_effective": list(
                resolve_effective_window(
                    model_wrapper.n_layers,
                    tuple(mhp.window) if mhp.window else None,
                )
            ),
        }
        for c, (shp, mhp) in hparams.items()
        if mhp is not None
    }
    (run_dir / "run_params.json").write_text(
        json.dumps({
            "run_name": args.run_name,
            "questions": str(args.questions),
            "conditions": list(args.conditions),
            "model_id": model_id,
            "dtype": dtype_str,
            "max_new_tokens": MAX_NEW_TOKENS,
            "gen_kwargs": GEN_KWARGS,
            "sparc": {c: (shp.as_dict() if shp else None) for c, (shp, mhp) in hparams.items()},
            "memvr": {c: (mhp.as_dict() if mhp else None) for c, (shp, mhp) in hparams.items()},
            "memvr_effective": memvr_effective,
            "timestamp_iso": _iso_now(),
        }, indent=2) + "\n",
        encoding="utf-8",
    )

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
                    # Per QUESTION, not per image: input_len moves with the
                    # question length, and the adaptive registry / MemVR Z are
                    # sized from image_positions at the prefill.
                    sparc_hp, memvr_hp = hparams[condition]
                    raw_answer, extra = _generate_answer(
                        model_wrapper, image, prompt, seed, sparc_hp, memvr_hp
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
                        "sparc": sparc_hp.as_dict() if sparc_hp else None,
                        "memvr": memvr_hp.as_dict() if memvr_hp else None,
                        "timestamp_iso": _iso_now(),
                        **_memvr_columns(extra),
                    }
                    if "prefill_selected" in extra:
                        entry["prefill_selected"] = extra["prefill_selected"]
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
