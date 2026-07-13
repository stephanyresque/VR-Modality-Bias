"""Thin facade over :mod:`vr_modality_bias.utils.memvr` that installs MemVR as a
context manager, mirroring ``experiment/sparc.enable_sparc`` so collection code
can flip it on/off per block and never leave the model hooked afterwards.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

from pyprojroot import here

try:
    from vr_modality_bias.utils.memvr import MemVRBuffer, install_memvr_hooks
except ModuleNotFoundError:
    sys.path.insert(0, str(here()))

    from src.vr_modality_bias.utils.memvr import MemVRBuffer, install_memvr_hooks

__all__ = [
    "MemVRHyperparams",
    "enable_memvr",
    "resolve_effective_window",
]


@dataclass
class MemVRHyperparams:
    """Container for MemVR hyperparameters.

    Defaults are the official README recipe (32-layer reference). ``window`` is
    ``None`` for the depth-fraction default resolved from the model's real layer
    count in :func:`enable_memvr`; pass an explicit ``(start, end)`` to override.
    """

    gamma: float = 0.75
    alpha: float = 0.12
    window: Optional[Tuple[int, int]] = None
    top_k: int = 10

    def __post_init__(self) -> None:
        # gamma is a NORMALIZED entropy threshold; 1.0 is the "never fires" gate.
        if self.gamma < 0.0:
            raise ValueError(f"gamma={self.gamma} must be >= 0.")
        # alpha is a convex-mix weight; 0.0 is the parametric-neutrality gate.
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError(f"alpha={self.alpha} must be in [0, 1].")
        if self.top_k < 1:
            raise ValueError(f"top_k={self.top_k} must be >= 1.")
        if self.window is not None:
            if len(self.window) != 2:
                raise ValueError(f"window={self.window} must be a (start, end) pair.")
            self.window = (int(self.window[0]), int(self.window[1]))

    def as_dict(self) -> dict:
        return {
            "gamma": self.gamma,
            "alpha": self.alpha,
            "window": list(self.window) if self.window is not None else None,
            "top_k": self.top_k,
        }


def resolve_effective_window(
    n_layers: int, window: Optional[Tuple[int, int]] = None
) -> Tuple[int, int]:
    """Resolve the (inclusive) firing window against the model's depth.

    Default is the depth fraction ``(round(L/6), round(L/2))`` (the paper's
    ``(5, 16)`` on 32 layers scaled by depth; ``(4, 12)`` on SmolVLM's 24). The
    end is hard-capped at ``L - 2`` because injection lands one layer above the
    firing layer (``l + 1``), which must stay a real layer.
    """
    if window is None:
        start = round(n_layers / 6)
        end = round(n_layers / 2)
    else:
        start, end = int(window[0]), int(window[1])
    start = max(start, 0)
    end = min(end, n_layers - 2)
    if start > end:
        raise ValueError(
            f"MemVR window is empty after the L-2 cap: n_layers={n_layers}, "
            f"resolved=({start}, {end}). Injection at l+1 needs a layer above "
            "the firing layer."
        )
    return (start, end)


@contextmanager
def enable_memvr(model_wrapper, hyperparams: MemVRHyperparams) -> Iterator[MemVRBuffer]:
    """Install MemVR on a loaded model for the duration of the with-block.

    Resolves the effective window from ``model_wrapper.n_layers``, installs the
    hooks + mlp wrappers, and restores everything on exit (including on an
    exception), so a later "MemVR OFF" run is never silently hooked.

    The caller MUST set ``buffer.update_image_positions(...)`` before each
    prefill: ``Z`` is read from the image positions and has no default location.

    Yields the :class:`MemVRBuffer` (shared state + instrumentation).
    """
    window = resolve_effective_window(model_wrapper.n_layers, hyperparams.window)

    buffer = MemVRBuffer()
    buffer.reset()
    buffer.gamma = float(hyperparams.gamma)
    buffer.alpha = float(hyperparams.alpha)
    buffer.top_k = int(hyperparams.top_k)
    buffer.window = window

    installation = install_memvr_hooks(model_wrapper, buffer)
    try:
        yield buffer
    finally:
        installation.remove()
        buffer.reset()
