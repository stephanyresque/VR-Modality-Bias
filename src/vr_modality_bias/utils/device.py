"""Device and dtype helpers.

Defaults to CUDA when available (the canonical execution target — see
EXPERIMENT.md §3.1) and falls back to CPU otherwise.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

__all__ = ["select_device", "resolve_dtype"]


def select_device(prefer: str = "cuda") -> "torch.device":
    """Return the best torch device available, honouring ``prefer``.

    Args:
        prefer: ``"cuda"`` (default), ``"mps"``, or ``"cpu"``. If the
            preferred device is unavailable, falls back to CPU.

    Returns:
        A ``torch.device`` instance.
    """
    import torch

    prefer = prefer.lower()
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return torch.device("mps")
    return torch.device("cpu")


_DTYPE_ALIASES = {
    "float16": "float16",
    "fp16": "float16",
    "half": "float16",
    "float32": "float32",
    "fp32": "float32",
    "float": "float32",
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
}


def resolve_dtype(name: str) -> "torch.dtype":
    """Map a config string to a ``torch.dtype``."""
    import torch

    canonical = _DTYPE_ALIASES.get(name.lower())
    if canonical is None:
        raise ValueError(
            f"Unsupported dtype {name!r}. Known: {sorted(_DTYPE_ALIASES)}."
        )
    return getattr(torch, canonical)
