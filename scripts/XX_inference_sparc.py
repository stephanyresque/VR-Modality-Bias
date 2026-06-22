
from __future__ import annotations

import os
import sys
import glob 
from loguru import logger

import traceback
import argparse
from pathlib import Path
from pyprojroot import here
from PIL import Image
from loguru import logger

try:

    from vr_modality_bias.utils.attn import add_custom_attention_layers, SelectedIndexBuffer

    from vr_modality_bias.data.prompts import get_prompt
    from vr_modality_bias.io.results import write_metrics_table
    from vr_modality_bias.io.storage import load_hidden_states
    from vr_modality_bias.metrics.cosine import compute_cosine_distance_matrix
    from vr_modality_bias.metrics.kl import compute_kl_matrix
    from vr_modality_bias.metrics.residual import residual_drift_ratio
    from vr_modality_bias.models.registry import build_model
    from vr_modality_bias.utils.config import load_config
    from vr_modality_bias.utils.device import resolve_dtype, select_device
    from vr_modality_bias.utils.logging import configure_logging, get_logger
    from vr_modality_bias.utils.runs import current_run_dir
    from vr_modality_bias.utils.seeds import derive_image_seed

except ModuleNotFoundError:

    sys.path.insert(0, str(here()))

    from src.vr_modality_bias.utils.attn import add_custom_attention_layers, SelectedIndexBuffer

    from src.vr_modality_bias.data.prompts import get_prompt
    from src.vr_modality_bias.io.results import write_metrics_table
    from src.vr_modality_bias.io.storage import load_hidden_states
    from src.vr_modality_bias.metrics.cosine import compute_cosine_distance_matrix
    from src.vr_modality_bias.metrics.kl import compute_kl_matrix
    from src.vr_modality_bias.metrics.residual import residual_drift_ratio
    from src.vr_modality_bias.models.registry import build_model
    from src.vr_modality_bias.utils.config import load_config
    from src.vr_modality_bias.utils.device import resolve_dtype, select_device
    from src.vr_modality_bias.utils.logging import configure_logging, get_logger
    from src.vr_modality_bias.utils.runs import current_run_dir
    from src.vr_modality_bias.utils.seeds import derive_image_seed


def _discover_image_ids(hidden_states_dir: Path) -> list[str]:
    """Return image_ids that have BOTH ``*__A.h5`` and ``*__B.h5`` under ``dir``."""
    ids: dict[str, set[str]] = {}
    for path in hidden_states_dir.glob("*.h5"):
        stem = path.stem
        if "__" not in stem:
            continue
        image_id, condition = stem.rsplit("__", 1)
        if condition not in {"A", "B"}:
            continue
        ids.setdefault(image_id, set()).add(condition)
    complete = sorted(image_id for image_id, conds in ids.items() if conds == {"A", "B"})

    return complete


def _probe_image_tokens(model, image: Image.Image, prompt: str, image_token_id: int) -> tuple[int, int, int]:
    """Return ``(image_token_index, input_len, num_image_patches)`` for ``image``.

        ``image_token_index`` is the sequence position where image patch tokens
        begin; ``input_len`` is the prompt length excluding those tokens
        (matches ``SelectedIndexBuffer.update_patch_num``'s expectation).
    """
    messages = model._build_messages(prompt, image)
    prompt_text = model._processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    inputs = model._processor(text=[prompt_text], images=[image], return_tensors="pt")
    input_ids = inputs["input_ids"][0]

    image_positions = (input_ids == image_token_id).nonzero(as_tuple=True)[0]
    
    total_len = int(input_ids.shape[-1])
    num_image_patches = int(image_positions.numel())
    
    return int(image_positions[0]), total_len - num_image_patches, num_image_patches


def main() -> int:

    parser = argparse.ArgumentParser(description=__doc__)
    
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    
    args = parser.parse_args()

    cfg = load_config(args.config)

    work_dir = Path(cfg["run"]["output_root"], cfg["run"]["name"])
    os.makedirs(work_dir, exist_ok=True)

    # run_dir = current_run_dir()
    # log_file = run_dir / "logs" / "XX_inference.log"
    # configure_logging(log_file=log_file)

    ## -------------- load model
    model = build_model(cfg["model"]["key"])
    model.model_id = str(cfg["model"]["model_id"])

    dtype = resolve_dtype(str(cfg["model"]["dtype"]))
    device = select_device("cuda")
    logger.info(f"Loading model {model.model_id} on {device} (dtype={dtype})…")

    if hasattr(model, "_dtype"):
        model._dtype = dtype  # noqa: SLF001

    model.load(device)

    logger.info(f"Run dir: {work_dir}",)

    ## -------------- dataset
    images_dir = cfg["dataset"]["images_dir"]
    images_files = sorted(glob.glob(f"{images_dir}{os.sep}*.jpg"))
    if args.limit:
        images_files = images_files[: args.limit]

    ## -------------- inference
    prompt_key = str(cfg["task"]["prompt_key"])
    prompt = get_prompt(prompt_key)

    seed_global = int(cfg["run"]["seed_global"])
    max_new_tokens = 1024 # int(cfg["generation"]["max_new_tokens"])
    gen_kwargs = {
        "do_sample": True,
        "temperature": float(cfg["generation"]["temperature"]),
        "top_p": float(cfg["generation"]["top_p"]),
        "repetition_penalty": float(cfg["generation"]["repetition_penalty"]),
    }

    image_token_id = int(model._model.config.image_token_id)

    ## -------------- run on the first image, with and without SPARC
    image_path = images_files[0]
    image_id = Path(image_path).stem
    with Image.open(image_path) as raw:
        image = raw.convert("RGB")

    seed = derive_image_seed(seed_global, image_id)

    # 1) baseline: clean model, no SPARC interference (run before patching)
    caption_base = model.generate_caption(
        image=image,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        seed=seed,
        generation_kwargs=gen_kwargs,
    )
    logger.info(f"[{image_id}] WITHOUT sparc: {caption_base} {len(caption_base.split())}")

    gen_kwargs = {
        "do_sample": False,
        "repetition_penalty": 1.2,
        "no_repeat_ngram_size": 3,
    }

    # 2) with SPARC: monkey-patch the attention layers, then generate
    indices_buffer = SelectedIndexBuffer()
    indices_buffer.reset()

    image_token_index, input_len, _ = _probe_image_tokens(
        model, image, prompt, image_token_id
    )

    add_custom_attention_layers(
        model._model,
        indices_buffer=indices_buffer,
        image_token_index=image_token_index,
        alpha=1.05,
        beta=0.1, 
        tau=3.0,
        selected_layer=18, 
        se_layers=(0,29)
    )
    logger.info(f"sparc loaded. image_token_index={image_token_index}")

    indices_buffer.update_input_len(input_len)

    caption_sparc = model.generate_caption(
        image=image,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        seed=seed,
        generation_kwargs=gen_kwargs,
    )
    logger.info(f"[{image_id}] WITH sparc: {caption_sparc} {len(caption_sparc.split())}")

    return 0


if __name__ == "__main__": 

    try: 
        
        main()

    except Exception as e:

        logger.debug(f"exception: \n{e}")
        logger.error(traceback.format_exc())
