"""Deterministic seed utilities"""

from __future__ import annotations

import hashlib

__all__ = ["derive_image_seed", "set_global_seeds"]


def derive_image_seed(seed_global: int, image_id: str) -> int:
    
    h = hashlib.sha256(image_id.encode("utf-8")).hexdigest()
    return (seed_global + int(h[:8], 16)) % (2**32)


def set_global_seeds(seed: int) -> None:
    
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
