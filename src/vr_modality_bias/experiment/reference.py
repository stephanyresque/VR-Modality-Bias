"""Free generation of ``caption_ref`` for every image in the manifest.

EXPERIMENT.md §4.1. The output is a JSON Lines file (one record per image)
that the teacher-forcing stage consumes as its forced target sequence.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from vr_modality_bias.data.manifests import ImageRecord
from vr_modality_bias.models.base import ModelWrapper
from vr_modality_bias.utils.logging import get_logger
from vr_modality_bias.utils.seeds import derive_image_seed

__all__ = ["RefCaptionRecord", "generate_reference_captions"]


class RefCaptionRecord(dict):
    """Dict subclass marking a single ``ref_captions.jsonl`` record."""


def _iso_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def generate_reference_captions(
    model: ModelWrapper,
    manifest: Iterable[ImageRecord],
    images_dir: Path,
    output_path: Path,
    *,
    prompt: str,
    prompt_key: str,
    seed_global: int,
    max_new_tokens: int,
    generation_kwargs: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> int:
    """Run free generation for every record in ``manifest``; write JSONL.

    Returns the number of records written. Skips records whose ``image_id``
    is already present in ``output_path`` unless ``overwrite`` is set.
    """
    log = get_logger(__name__)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    already_done: set[str] = set()
    if output_path.exists() and not overwrite:
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    already_done.add(json.loads(line)["image_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
        mode = "a"
        log.info(
            "Resuming — %d caption(s) already present in %s.",
            len(already_done),
            output_path,
        )
    else:
        mode = "w"

    n_written = 0
    with output_path.open(mode, encoding="utf-8") as f:
        for record in manifest:
            if record.image_id in already_done:
                continue
            image_path = Path(images_dir) / record.file_name
            with Image.open(image_path) as raw:
                image = raw.convert("RGB")

            noise_seed = derive_image_seed(seed_global, record.image_id)
            caption = model.generate_caption(
                image=image,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                seed=noise_seed,
                generation_kwargs=generation_kwargs,
            )

            row = RefCaptionRecord({
                "image_id": record.image_id,
                "caption_ref": caption,
                "model_id": model.model_id,
                "prompt_key": prompt_key,
                "noise_seed": int(noise_seed),
                "seed_global": int(seed_global),
                "timestamp": _iso_now(),
            })
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_written += 1
            log.info(
                "[%s] caption_ref: %s",
                record.image_id,
                (caption[:80] + "...") if len(caption) > 80 else caption,
            )

    return n_written


def read_reference_captions(path: Path) -> dict[str, dict[str, Any]]:
    """Load ``ref_captions.jsonl`` into ``{image_id: row}``."""
    out: dict[str, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[row["image_id"]] = row
    return out
