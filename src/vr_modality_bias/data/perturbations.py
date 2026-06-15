"""Image perturbations used as causal interventions."""

from __future__ import annotations

import sys

import numpy as np
from PIL import Image
from pyprojroot import here

try:
    from vr_modality_bias.utils.seeds import derive_image_seed
except ModuleNotFoundError:
    sys.path.insert(0, str(here()))

    from src.vr_modality_bias.utils.seeds import derive_image_seed

__all__ = ["derive_image_seed", "noise_image_uniform"]


def noise_image_uniform(image: Image.Image, seed: int) -> Image.Image:
    """Return a uniform-random-noise RGB image with the same ``(W, H)`` as ``image``."""
    image = image.convert("RGB")
    w, h = image.size
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")
