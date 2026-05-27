"""Deterministic seed utilities.

The single source of truth for both:

- :func:`derive_image_seed`, the per-(global_seed, image_id) seed derivation
  formalised in EXPERIMENT.md §4.2; and
- :func:`set_global_seeds`, a best-effort helper to seed Python's ``random``,
  NumPy and (when available) Torch in one call.
"""

from __future__ import annotations

import hashlib

__all__ = ["derive_image_seed", "set_global_seeds"]


def derive_image_seed(seed_global: int, image_id: str) -> int:
    """Return a deterministic 32-bit seed for the pair (``seed_global``, ``image_id``).

    Implements EXPERIMENT.md §4.2 verbatim::

        noise_seed = (seed_global + int(sha256(image_id)[:8], 16)) % 2**32

    The output is always a non-negative integer strictly below ``2**32``,
    suitable for use with :class:`numpy.random.Generator` and
    :func:`numpy.random.default_rng`.
    """
    h = hashlib.sha256(image_id.encode("utf-8")).hexdigest()
    return (seed_global + int(h[:8], 16)) % (2**32)


def set_global_seeds(seed: int) -> None:
    """Seed Python ``random``, NumPy, and (if installed) Torch for reproducibility.

    This is a best-effort convenience for top-level scripts; it does not
    enable deterministic CUDA kernels, which is left to the caller because
    that toggle has a non-trivial performance cost.
    """
    import random

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
