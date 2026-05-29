"""Paired teacher-forced execution: real image (A) and uniform-noise image (B)"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image

from vr_modality_bias.data.manifests import ImageRecord
from vr_modality_bias.data.perturbations import noise_image_uniform
from vr_modality_bias.io.storage import (
    hidden_states_filename,
    save_hidden_states,
)
from vr_modality_bias.models.base import ModelWrapper
from vr_modality_bias.utils.logging import get_logger
from vr_modality_bias.utils.seeds import derive_image_seed

__all__ = ["run_paired_for_image", "collect_paired_for_manifest"]


def _iso_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def run_paired_for_image(
    *,
    model: ModelWrapper,
    image_id: str,
    image: Image.Image,
    prompt: str,
    prompt_key: str,
    caption_ref: str,
    out_dir: Path,
    seed_global: int,
    noise_seed: int,
    compression: str | None = "gzip",
    compression_level: int = 4,
) -> tuple[Path, Path]:
    
    image_rgb = image.convert("RGB")
    noise_img = noise_image_uniform(image_rgb, seed=int(noise_seed))

    result_A = model.run_teacher_forcing(image_rgb, prompt, caption_ref)
    result_B = model.run_teacher_forcing(noise_img, prompt, caption_ref)

    if not torch.equal(result_A.input_ids, result_B.input_ids):
        raise RuntimeError(
            f"[{image_id}] input_ids differ between conditions A and B. "
            "Investigate the image processor before relaxing this check "
            "(EXPERIMENT.md §4.3, §12)."
        )

    timestamp = _iso_now()
    extra = {
        "image_id": image_id,
        "prompt_key": prompt_key,
        "seed_global": int(seed_global),
        "noise_seed": int(noise_seed),
        "caption_ref": caption_ref,
        "timestamp_iso": timestamp,
    }

    out_dir = Path(out_dir)
    path_A = out_dir / hidden_states_filename(image_id, "A")
    path_B = out_dir / hidden_states_filename(image_id, "B")

    save_hidden_states(
        path_A,
        result_A,
        condition="A",
        extra_attrs=extra,
        compression=compression,
        compression_level=compression_level,
    )
    save_hidden_states(
        path_B,
        result_B,
        condition="B",
        extra_attrs=extra,
        compression=compression,
        compression_level=compression_level,
    )
    return path_A, path_B


def collect_paired_for_manifest(
    *,
    model: ModelWrapper,
    manifest: Iterable[ImageRecord],
    images_dir: Path,
    out_dir: Path,
    captions: dict[str, dict],
    prompt: str,
    prompt_key: str,
    seed_global: int,
    compression: str | None = "gzip",
    compression_level: int = 4,
    overwrite: bool = False,
) -> int:
    
    log = get_logger(__name__)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_done = 0
    for record in manifest:
        path_A = out_dir / hidden_states_filename(record.image_id, "A")
        path_B = out_dir / hidden_states_filename(record.image_id, "B")
        if path_A.exists() and path_B.exists() and not overwrite:
            log.info("[%s] already present, skipping.", record.image_id)
            n_done += 1
            continue

        if record.image_id not in captions:
            raise KeyError(
                f"No reference caption for image_id={record.image_id!r}. "
                "Run scripts/03_generate_refs.py first."
            )
        caption_ref = str(captions[record.image_id]["caption_ref"])
        noise_seed = derive_image_seed(seed_global, record.image_id)

        image_path = Path(images_dir) / record.file_name
        with Image.open(image_path) as raw:
            image = raw.convert("RGB")

        run_paired_for_image(
            model=model,
            image_id=record.image_id,
            image=image,
            prompt=prompt,
            prompt_key=prompt_key,
            caption_ref=caption_ref,
            out_dir=out_dir,
            seed_global=seed_global,
            noise_seed=noise_seed,
            compression=compression,
            compression_level=compression_level,
        )
        n_done += 1
        log.info("[%s] paired hidden states saved.", record.image_id)

    return n_done
