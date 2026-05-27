"""Image perturbations used as causal interventions.

The baseline contains a single intervention: a *brand-new* image whose pixels
are sampled i.i.d. uniformly in [0, 255]. This is **not** Gaussian noise
added to the original image. See EXPERIMENT.md §4.2 for the methodological
justification (probing causal contribution of the visual modality, not
simulating realistic VR-capture defects).
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from vr_modality_bias.utils.seeds import derive_image_seed

__all__ = ["derive_image_seed", "noise_image_uniform"]


def noise_image_uniform(image: Image.Image, seed: int) -> Image.Image:
    """Return a uniform-random-noise RGB image with the same ``(W, H)`` as ``image``.

    Args:
        image: The reference image — only its size and mode are read.
        seed: Seed for :func:`numpy.random.default_rng`. Use
            :func:`vr_modality_bias.utils.seeds.derive_image_seed` to derive
            a per-image seed from the global run seed.

    Returns:
        A new ``PIL.Image`` in RGB mode whose pixel values are i.i.d.
        uniform over ``[0, 255]``. The shape and mode of ``image`` are
        preserved so the visual encoder sees an input of valid form but null
        spatial/semantic structure.

    Note:
        The output is independent of ``image``'s pixel content — only its
        size matters. The input is forced to RGB before reading ``size`` so
        grayscale inputs still produce three-channel outputs.
    """
    image = image.convert("RGB")
    w, h = image.size
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")
