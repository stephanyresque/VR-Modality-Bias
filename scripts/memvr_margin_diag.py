#!/usr/bin/env python
"""Directionality go/no-go for MemVR: the prefill yes/no logit margin vs baseline
grouped by ground truth, plus a per-layer entropy profile for gamma tuning.

Run: python scripts/memvr_margin_diag.py --config configs/run_smolvlm22_short.yaml --limit 300
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import sys
import traceback
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
    from vr_modality_bias.models.registry import build_model
    from vr_modality_bias.utils.config import load_config
    from vr_modality_bias.utils.device import resolve_dtype, select_device
    from vr_modality_bias.utils.memvr import normalized_topk_entropy
except ModuleNotFoundError:
    sys.path.insert(0, str(here()))

    from src.vr_modality_bias.experiment.memvr import (
        MemVRHyperparams,
        enable_memvr,
        resolve_effective_window,
    )
    from src.vr_modality_bias.experiment.sparc import SparcHyperparams, enable_sparc
    from src.vr_modality_bias.models.registry import build_model
    from src.vr_modality_bias.utils.config import load_config
    from src.vr_modality_bias.utils.device import resolve_dtype, select_device
    from src.vr_modality_bias.utils.memvr import normalized_topk_entropy


# Only the three arms the go/no-go needs. sparc_memvr reuses the ORIGINAL alpha^c
# SPARC (no adaptive, no qcond), mirroring pope_generate's mapping; the
# adaptive/qcond arms are already known to be baseline-identical on POPE.
DIAG_CONDITIONS = ("baseline", "memvr", "sparc_memvr")

DEFAULT_QUESTIONS = Path("data/processed/mscoco_baseline/pope_questions.jsonl")

# The logit lens' top-k for the entropy profile, matching the MemVR trigger.
ENTROPY_TOP_K = 10

# Yes/No surface forms; the first token of the winning variant (by baseline
# logit) becomes the margin's yes/no id. Leading-space forms matter for BPE.
_YES_VARIANTS = ("Yes", " Yes", "yes", " yes")
_NO_VARIANTS = ("No", " No", "no", " no")


def _iso_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------- hyperparams


def hparams_for_condition(
    condition: str, args
) -> tuple[SparcHyperparams | None, MemVRHyperparams | None]:
    """``(sparc_hp, memvr_hp)`` for a diagnostic condition (mirrors Etapa 2).

    baseline -> (None, None); memvr -> (None, MemVR); sparc_memvr -> (original
    alpha^c SPARC, MemVR).
    """
    if condition not in DIAG_CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}. Known: {DIAG_CONDITIONS}.")
    sparc_hp = None
    if condition == "sparc_memvr":
        sparc_hp = SparcHyperparams(
            alpha=args.alpha,
            tau=args.tau,
            selected_layer=args.selected_layer,
            se_layers=tuple(args.se_layers),
            beta=args.beta,
        )
    memvr_hp = None
    if condition in ("memvr", "sparc_memvr"):
        window = tuple(args.memvr_window) if args.memvr_window else None
        memvr_hp = MemVRHyperparams(
            gamma=args.memvr_gamma, alpha=args.memvr_alpha, window=window
        )
    return sparc_hp, memvr_hp


# ---------------------------------------------------------------- yes/no tokens


def _first_token_ids(tokenizer, variants) -> dict:
    """Map each surface form to its first token id (no special tokens)."""
    ids = {}
    for variant in variants:
        toks = tokenizer.encode(variant, add_special_tokens=False)
        if toks:
            ids[variant] = int(toks[0])
    return ids


def resolve_yes_no_tokens(tokenizer, baseline_logits) -> dict:
    """Pick the yes/no variant whose first token is largest in ``baseline_logits``.

    Resolved once, from the first question's baseline logits, and frozen for the
    run. Returns ``{yes_id, no_id, yes_variant, no_variant}``.
    """
    yes_ids = _first_token_ids(tokenizer, _YES_VARIANTS)
    no_ids = _first_token_ids(tokenizer, _NO_VARIANTS)
    if not yes_ids or not no_ids:
        raise RuntimeError(
            "Could not tokenise any yes/no surface form; check the tokenizer."
        )
    yes_variant = max(yes_ids, key=lambda v: float(baseline_logits[yes_ids[v]]))
    no_variant = max(no_ids, key=lambda v: float(baseline_logits[no_ids[v]]))
    return {
        "yes_id": yes_ids[yes_variant],
        "no_id": no_ids[no_variant],
        "yes_variant": yes_variant,
        "no_variant": no_variant,
    }


def margin_from_logits(logits, yes_id: int, no_id: int) -> float:
    """The yes/no decision margin at a position: ``logit(yes) - logit(no)``."""
    return float(logits[yes_id]) - float(logits[no_id])


# ---------------------------------------------------------------- summary


def _mean(values) -> float:
    return sum(values) / len(values) if values else float("nan")


def classify_directionality(delta_yes_mean: float, delta_no_mean: float) -> str:
    """Verdict from the two grouped mean deltas.

    ``directional``: the margin moves TOWARD the ground truth (up when the answer
    is yes, down when it is no). ``generic_bias``: both deltas share a sign, the
    Point-2 artefact (a uniform pro-yes/pro-no shift regardless of the truth).
    ``inconclusive`` otherwise (anti-directional or a zero mean);
    ``undetermined`` if a group is empty.
    """
    if not (math.isfinite(delta_yes_mean) and math.isfinite(delta_no_mean)):
        return "undetermined"
    if delta_yes_mean > 0 and delta_no_mean < 0:
        return "directional"
    same_sign = (delta_yes_mean > 0 and delta_no_mean > 0) or (
        delta_yes_mean < 0 and delta_no_mean < 0
    )
    if same_sign:
        return "generic_bias"
    return "inconclusive"


def build_summary(per_question: list[dict], conditions) -> dict:
    """Aggregate per-question margins into the directionality verdict per arm.

    ``per_question`` items: ``{"expected": "yes"|"no", "margins": {cond: float},
    "fired": {cond: bool}}``. Deltas are ``margin[cond] - margin["baseline"]`` on
    the SAME question, so ``baseline`` must be among the conditions.
    """
    summary: dict = {}
    for cond in conditions:
        if cond == "baseline":
            continue
        deltas_yes, deltas_no = [], []
        deltas_fired, deltas_not_fired = [], []
        n_total = n_fired = n_fired_known = 0
        for item in per_question:
            margins = item["margins"]
            if "baseline" not in margins or cond not in margins:
                continue
            delta = margins[cond] - margins["baseline"]
            n_total += 1
            if item["expected"] == "yes":
                deltas_yes.append(delta)
            elif item["expected"] == "no":
                deltas_no.append(delta)
            fired = item.get("fired", {}).get(cond)
            if fired is not None:
                n_fired_known += 1
                if fired:
                    n_fired += 1
                    deltas_fired.append(delta)
                else:
                    deltas_not_fired.append(delta)
        delta_yes_mean = _mean(deltas_yes)
        delta_no_mean = _mean(deltas_no)
        summary[cond] = {
            "n": n_total,
            "n_yes": len(deltas_yes),
            "n_no": len(deltas_no),
            "delta_yes_mean": delta_yes_mean,
            "delta_no_mean": delta_no_mean,
            "verdict": classify_directionality(delta_yes_mean, delta_no_mean),
            "firing_rate": (n_fired / n_fired_known) if n_fired_known else float("nan"),
            "n_fired": n_fired,
            "delta_fired_mean": _mean(deltas_fired),
            "delta_not_fired_mean": _mean(deltas_not_fired),
        }
    return summary


# ---------------------------------------------------------------- IO helpers


def load_questions(path: Path) -> list[dict]:
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


def _append_jsonl(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _load_image(images_dir: Path, image_id: str):
    image_path = images_dir / f"{image_id}.jpg"
    if not image_path.exists():
        return None
    with Image.open(image_path) as raw:
        return raw.convert("RGB")


# ---------------------------------------------------------------- forward


def _prefill_inputs_and_layout(model_wrapper, image, prompt):
    """Build the model inputs for a prefill plus the SPARC/MemVR layout.

    Same tokenisation as pope_generate's ``_probe_sparc_layout`` (pure, safe
    before opening the contexts), but also returns the processor ``inputs`` so
    the caller can run a manual forward rather than ``generate``.
    """
    processor = model_wrapper._processor  # noqa: SLF001
    messages = model_wrapper._build_messages(prompt, image)  # noqa: SLF001
    prefix_text = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )
    inputs = processor(text=[prefix_text], images=[image], return_tensors="pt")
    input_ids = inputs["input_ids"][0]
    image_token_id = int(model_wrapper._model.config.image_token_id)  # noqa: SLF001
    image_positions = (input_ids == image_token_id).nonzero(as_tuple=True)[0]
    num_image_patches = int(image_positions.numel())
    caption_start = int(input_ids.shape[-1])
    question_positions = torch.arange(
        int(image_positions[-1]) + 1, caption_start, dtype=torch.long
    )
    return inputs, caption_start - num_image_patches, image_positions, question_positions


def _forward_last_logits(model, inputs, device):
    """One prefill forward; the last-position logits as a CPU float32 vector.

    ``use_cache=False``: we only read the prefill logits (which decide the first
    answer token), so no cache is needed, and the SPARC/MemVR patches all act on
    this single forward.
    """
    inputs = inputs.to(device)
    with torch.no_grad():
        outputs = model(**inputs, use_cache=False)
    return outputs.logits[0, -1, :].detach().float().cpu()


def _prefill_logits(model_wrapper, image, prompt, sparc_hp, memvr_hp, device):
    """Run one prefill under ``(sparc_hp, memvr_hp)``; return (logits, instr).

    Same nested-context pattern as Etapa 2 (SPARC outer, MemVR inner), each
    buffer updated before the forward; instrumentation read inside the context.
    """
    inputs, input_len, image_positions, question_positions = _prefill_inputs_and_layout(
        model_wrapper, image, prompt
    )
    if sparc_hp is None and memvr_hp is None:
        return _forward_last_logits(model_wrapper._model, inputs, device), None

    instrumentation = None
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
        if memvr_hp is not None:
            memvr_buffer = stack.enter_context(enable_memvr(model_wrapper, memvr_hp))
            memvr_buffer.update_image_positions(image_positions)
        else:
            memvr_buffer = None

        logits = _forward_last_logits(model_wrapper._model, inputs, device)

        if memvr_buffer is not None:
            instrumentation = {
                "memvr_fired_in_prefill": bool(memvr_buffer.fired_in_prefill),
                "memvr_fire_layer": memvr_buffer.fire_layer,
                "memvr_fire_entropy": memvr_buffer.fire_entropy,
            }
    return logits, instrumentation


def _entropy_profile(model_wrapper, questions, images_dir, device, n):
    """Per-layer normalised entropy of the last prefill position, baseline only.

    Reuses ``normalized_topk_entropy`` and the wrapper's logit lens
    (``get_final_norm`` + ``get_lm_head``); no MemVR is installed. Feeds the
    gamma choice when the prefill firing rate is low.
    """
    final_norm = model_wrapper.get_final_norm()
    lm_head = model_wrapper.get_lm_head()
    per_question_layers: list[list[float]] = []
    for q in questions[:n]:
        image = _load_image(images_dir, str(q["image_id"]))
        if image is None:
            continue
        inputs, *_ = _prefill_inputs_and_layout(model_wrapper, image, str(q["question"]))
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model_wrapper._model(
                **inputs, use_cache=False, output_hidden_states=True
            )
        # hidden_states[0] is the embedding output; [1:] are the per-layer
        # outputs the trigger would read.
        layer_entropies = []
        for hidden in outputs.hidden_states[1:]:
            last = hidden[0, -1, :]
            with torch.no_grad():
                logits = lm_head(final_norm(last))
            layer_entropies.append(
                float(normalized_topk_entropy(logits, ENTROPY_TOP_K).reshape(-1)[0])
            )
        per_question_layers.append(layer_entropies)
    return per_question_layers


# ---------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True,
        help="Family config; only the model + dataset blocks are used.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS,
        help=f"pope_questions.jsonl from build_pope.py (default: {DEFAULT_QUESTIONS}).")
    parser.add_argument("--run-name", type=str, default="memvr_margin_diag",
        help="Output goes to <output-root>/<run-name>/.")
    parser.add_argument("--output-root", type=Path, default=Path("results/runs"),
        help="Parent dir for run directories (default: results/runs).")
    parser.add_argument("--images-dir", type=Path, default=None,
        help="Override the images dir; default comes from the config's dataset block.")
    parser.add_argument("--conditions", nargs="+", choices=list(DIAG_CONDITIONS),
        default=list(DIAG_CONDITIONS),
        help="Arms to measure (default: all three; baseline is required).")
    parser.add_argument("--limit", type=int, default=300,
        help="Number of questions to sample (default 300).")
    parser.add_argument("--seed", type=int, default=0,
        help="Seed for the deterministic subsample of the manifest.")
    parser.add_argument("--entropy-profile", type=int, default=20,
        metavar="N",
        help="Profile the per-layer prefill entropy on the first N questions "
             "(baseline only). 0 disables it.")
    # SPARC arm of sparc_memvr (original alpha^c). Defaults match pope_generate.
    parser.add_argument("--alpha", type=float, default=1.05, help="SPARC alpha.")
    parser.add_argument("--beta", type=float, default=0.1, help="SPARC beta.")
    parser.add_argument("--tau", type=float, default=3.0, help="SPARC tau.")
    parser.add_argument("--selected-layer", type=int, default=20,
        help="Reference layer. Per family: 15 smolvlm, 20 llava, 18 qwen.")
    parser.add_argument("--se-layers", type=int, nargs=2, default=(0, 31),
        help="SPARC se_layers (lo hi).")
    # MemVR flags (identical to Etapa 2, for the later gamma sweep).
    parser.add_argument("--memvr-gamma", type=float, default=0.75,
        help="MemVR entropy threshold (normalised); 1.0 never fires.")
    parser.add_argument("--memvr-alpha", type=float, default=0.12,
        help="MemVR retracing ratio (convex-mix weight); 0.0 is a no-op.")
    parser.add_argument("--memvr-window", type=int, nargs=2, default=None,
        metavar=("START", "END"),
        help="MemVR firing window (inclusive). Omitted = depth fraction "
             "(round(L/6), round(L/2)), capped at L-2.")
    parser.add_argument("--overwrite", action="store_true",
        help="Delete existing diagnostic outputs before starting.")
    return parser


def _print_summary(summary: dict) -> None:
    for cond, s in summary.items():
        logger.info("-" * 70)
        logger.info(f"CONDITION {cond} vs baseline  (n={s['n']})")
        logger.info(
            f"  delta_margin gt=yes: {s['delta_yes_mean']:+.4f}  (n_yes={s['n_yes']})"
        )
        logger.info(
            f"  delta_margin gt=no : {s['delta_no_mean']:+.4f}  (n_no={s['n_no']})"
        )
        logger.info(f"  VERDICT: {s['verdict'].upper()}")
        logger.info(f"  prefill firing rate: {s['firing_rate'] * 100:.1f}%")
        logger.info(
            f"  delta_margin | fired: {s['delta_fired_mean']:+.4f}  "
            f"| not fired: {s['delta_not_fired_mean']:+.4f}"
        )


def main() -> int:
    args = build_parser().parse_args()

    if "baseline" not in args.conditions:
        logger.error(
            "baseline must be among --conditions: the margin deltas and the "
            "yes/no token resolution are both defined against it."
        )
        return 1

    # Fail before the checkpoint load if an arm is misconfigured.
    hparams = {c: hparams_for_condition(c, args) for c in args.conditions}
    # baseline first so the yes/no tokens are resolved from its logits.
    conditions = [c for c in DIAG_CONDITIONS if c in args.conditions]

    run_dir = args.output_root / args.run_name
    log_file = run_dir / "logs" / "memvr_margin_diag.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_file), enqueue=True, level="INFO")

    questions = load_questions(args.questions)
    if not questions:
        logger.error(f"No questions in {args.questions}. Run scripts/build_pope.py first.")
        return 1
    rng = random.Random(args.seed)
    rng.shuffle(questions)
    if args.limit:
        questions = questions[: args.limit]
    torch.manual_seed(args.seed)

    cfg = load_config(args.config)
    model_key = str(cfg["model"]["key"])
    model_id = str(cfg["model"]["model_id"])
    dtype_str = str(cfg["model"]["dtype"])
    images_dir = Path(args.images_dir or cfg["dataset"]["images_dir"])

    logger.info("=" * 70)
    logger.info(f"MemVR margin diagnostic, run_name={args.run_name}")
    logger.info(f"questions : {len(questions)} (seed {args.seed}) from {args.questions}")
    logger.info(f"conditions: {conditions}")
    logger.info(f"model     : {model_id} ({dtype_str})")
    logger.info("=" * 70)

    run_dir.mkdir(parents=True, exist_ok=True)
    records_path = run_dir / "margin_records.jsonl"
    summary_path = run_dir / "margin_summary.json"
    profile_path = run_dir / "entropy_profile.json"
    params_path = run_dir / "diag_params.json"
    if args.overwrite:
        for path in (records_path, summary_path, profile_path, params_path):
            if path.exists():
                path.unlink()

    model_wrapper = build_model(model_key)
    model_wrapper.model_id = model_id
    if hasattr(model_wrapper, "_dtype"):
        model_wrapper._dtype = resolve_dtype(dtype_str)  # noqa: SLF001
    device = select_device("cuda")
    logger.info(f"Loading {model_id} on {device}...")
    model_wrapper.load(device)
    logger.info(f"Model loaded. n_layers={model_wrapper.n_layers}")

    memvr_effective = {
        c: list(
            resolve_effective_window(
                model_wrapper.n_layers, tuple(mhp.window) if mhp.window else None
            )
        )
        for c, (shp, mhp) in hparams.items()
        if mhp is not None
    }

    tokenizer = model_wrapper._processor.tokenizer  # noqa: SLF001

    # ---- entropy profile (baseline only) ----
    if args.entropy_profile > 0:
        logger.info(f"Entropy profile on the first {args.entropy_profile} question(s)...")
        matrix = _entropy_profile(
            model_wrapper, questions, images_dir, device, args.entropy_profile
        )
        n_layers = len(matrix[0]) if matrix else 0
        per_layer_mean = [
            _mean([row[layer] for row in matrix]) for layer in range(n_layers)
        ]
        profile_path.write_text(
            json.dumps({
                "n_questions": len(matrix),
                "n_layers": n_layers,
                "top_k": ENTROPY_TOP_K,
                "per_layer_mean": per_layer_mean,
                "per_question_layers": matrix,
            }, indent=2) + "\n",
            encoding="utf-8",
        )
        logger.info(
            "Per-layer mean entropy: "
            + ", ".join(f"L{i}:{v:.3f}" for i, v in enumerate(per_layer_mean))
        )
        logger.info(f"Entropy profile: {profile_path}")

    # ---- margin diagnostic ----
    token_meta: dict | None = None
    per_question: list[dict] = []
    n_done = n_failed = 0

    for q in questions:
        image_id = str(q["image_id"])
        prompt = str(q["question"])
        expected = str(q["expected"])
        image = _load_image(images_dir, image_id)
        if image is None:
            logger.error(f"{images_dir / (image_id + '.jpg')} missing; skipping.")
            n_failed += 1
            continue

        item = {"image_id": image_id, "expected": expected, "margins": {}, "fired": {}}
        for condition in conditions:
            sparc_hp, memvr_hp = hparams[condition]
            try:
                logits, instr = _prefill_logits(
                    model_wrapper, image, prompt, sparc_hp, memvr_hp, device
                )
            except Exception as exc:
                n_failed += 1
                logger.error(f"[{image_id}|{condition}] FAILED: {exc}")
                logger.error(traceback.format_exc())
                continue

            if token_meta is None:
                token_meta = resolve_yes_no_tokens(tokenizer, logits)
                logger.info(
                    f"yes/no tokens: yes={token_meta['yes_variant']!r} "
                    f"(id {token_meta['yes_id']}), no={token_meta['no_variant']!r} "
                    f"(id {token_meta['no_id']})"
                )

            margin = margin_from_logits(logits, token_meta["yes_id"], token_meta["no_id"])
            item["margins"][condition] = margin
            if instr is not None:
                item["fired"][condition] = instr["memvr_fired_in_prefill"]

            record = {
                "image_id": image_id,
                "question": prompt,
                "expected": expected,
                "condition": condition,
                "margin": margin,
                "timestamp_iso": _iso_now(),
            }
            if instr is not None:
                record.update(instr)
            _append_jsonl(records_path, record)
            n_done += 1

        per_question.append(item)

    summary = build_summary(per_question, conditions)

    params_path.write_text(
        json.dumps({
            "run_name": args.run_name,
            "questions": str(args.questions),
            "n_questions": len(questions),
            "seed": args.seed,
            "conditions": conditions,
            "model_id": model_id,
            "dtype": dtype_str,
            "token_meta": token_meta,
            "memvr": {c: (mhp.as_dict() if mhp else None) for c, (shp, mhp) in hparams.items()},
            "memvr_effective": memvr_effective,
            "sparc": {c: (shp.as_dict() if shp else None) for c, (shp, mhp) in hparams.items()},
            "timestamp_iso": _iso_now(),
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    logger.info("=" * 70)
    logger.info(f"MARGIN DIAGNOSTIC SUMMARY (records: {n_done}, failed: {n_failed})")
    _print_summary(summary)
    logger.info("=" * 70)
    logger.info(f"Records : {records_path}")
    logger.info(f"Summary : {summary_path}")
    logger.info(f"Params  : {params_path}")
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
