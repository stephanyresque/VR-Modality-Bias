#!/usr/bin/env python
"""Generate ``ref_captions.jsonl`` for every manifest entry."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

from vr_modality_bias.data.manifests import iter_manifest
from vr_modality_bias.data.prompts import get_prompt
from vr_modality_bias.experiment.reference import generate_reference_captions
from vr_modality_bias.models.registry import build_model
from vr_modality_bias.utils.config import load_config, snapshot_config
from vr_modality_bias.utils.device import resolve_dtype, select_device
from vr_modality_bias.utils.logging import configure_logging, get_logger
from vr_modality_bias.utils.runs import make_run_dir
from vr_modality_bias.utils.seeds import set_global_seeds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    run_dir = make_run_dir(cfg["run"]["output_root"], cfg["run"]["name"])
    log_file = run_dir / "logs" / "03_generate_refs.log"
    configure_logging(log_file=log_file)
    log = get_logger(__name__)
    log.info("Run dir: %s", run_dir)

    snapshot_config(args.config, run_dir)

    seed_global = int(cfg["run"]["seed_global"])
    set_global_seeds(seed_global)

    prompt_key = str(cfg["task"]["prompt_key"])
    prompt = get_prompt(prompt_key)
    log.info("Prompt key: %s", prompt_key)

    model = build_model(cfg["model"]["key"])
    model.model_id = str(cfg["model"]["model_id"])
    dtype = resolve_dtype(str(cfg["model"]["dtype"]))
    device = select_device("cuda")
    log.info(
        "Loading model %s on %s (dtype=%s)…", model.model_id, device, dtype
    )
    # Some wrappers expose dtype as an init kw; SmolVLMWrapper does, so
    # update it after construction for any backend that supports it.
    if hasattr(model, "_dtype"):
        model._dtype = dtype  # noqa: SLF001
    model.load(device)
    log.info("Loaded. n_layers=%d", model.n_layers)

    manifest_path = Path(cfg["dataset"]["manifest_path"])
    images_dir = Path(cfg["dataset"]["images_dir"])
    manifest = iter_manifest(manifest_path)
    if args.limit:
        manifest = itertools.islice(manifest, args.limit)

    output_path = run_dir / "ref_captions.jsonl"
    gen_kwargs = {
        "do_sample": bool(cfg["generation"]["do_sample"]),
        "temperature": float(cfg["generation"]["temperature"]),
        "top_p": float(cfg["generation"]["top_p"]),
        "repetition_penalty": float(cfg["generation"]["repetition_penalty"]),
    }

    n = generate_reference_captions(
        model=model,
        manifest=manifest,
        images_dir=images_dir,
        output_path=output_path,
        prompt=prompt,
        prompt_key=prompt_key,
        seed_global=seed_global,
        max_new_tokens=int(cfg["generation"]["max_new_tokens"]),
        generation_kwargs=gen_kwargs,
        overwrite=args.overwrite,
    )
    log.info("Wrote %d caption(s) to %s.", n, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
