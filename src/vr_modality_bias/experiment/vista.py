"""Thin facade over :mod:`vr_modality_bias.utils.vista` that installs VISTA as a
context manager, mirroring ``experiment/memvr.enable_memvr`` so collection code
can flip it on/off per block and never leave the model hooked afterwards.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

from pyprojroot import here

try:
    from vr_modality_bias.utils.vista import VistaBuffer, install_vista_hooks
except ModuleNotFoundError:
    sys.path.insert(0, str(here()))

    from src.vr_modality_bias.utils.vista import VistaBuffer, install_vista_hooks

__all__ = [
    "VistaHyperparams",
    "enable_vista",
    "resolve_sla_window",
    "resolve_vsv_window",
]


@dataclass
class VistaHyperparams:
    """Container for VISTA hyperparameters.

    Defaults are the official POPE recipe. ``window`` is ``None`` for "all
    layers" (the official VSV default); ``sla_window`` is ``None`` for the
    depth-proportional default ``(round(25L/32), round(30L/32))`` resolved from
    the model's real layer count in :func:`enable_vista`.
    """

    lam: float = 0.01
    window: Optional[Tuple[int, int]] = None
    sla: bool = False
    sla_alpha: float = 0.3
    sla_window: Optional[Tuple[int, int]] = None

    def __post_init__(self) -> None:
        # lam multiplies the steering vector directly; 0.0 is the parametric gate.
        if self.lam < 0.0:
            raise ValueError(f"lam={self.lam} must be >= 0.")
        # sla_alpha is a convex-mix weight over the logits.
        if not (0.0 <= self.sla_alpha <= 1.0):
            raise ValueError(f"sla_alpha={self.sla_alpha} must be in [0, 1].")
        if self.window is not None:
            if len(self.window) != 2:
                raise ValueError(f"window={self.window} must be a (start, end) pair.")
            self.window = (int(self.window[0]), int(self.window[1]))
        if self.sla_window is not None:
            if len(self.sla_window) != 2:
                raise ValueError(f"sla_window={self.sla_window} must be a (start, end) pair.")
            self.sla_window = (int(self.sla_window[0]), int(self.sla_window[1]))

    def as_dict(self) -> dict:
        return {
            "lam": self.lam,
            "window": list(self.window) if self.window is not None else None,
            "sla": self.sla,
            "sla_alpha": self.sla_alpha,
            "sla_window": list(self.sla_window) if self.sla_window is not None else None,
        }


def resolve_vsv_window(
    n_layers: int, window: Optional[Tuple[int, int]] = None
) -> Optional[Tuple[int, int]]:
    """Resolve the (inclusive) VSV steering window; ``None`` means all layers.

    An explicit window is clamped to ``[0, L-1]``. Unlike MemVR there is no L-2
    cap: VISTA steers a layer's own output, it does not reach one layer up.
    """
    if window is None:
        return None
    start, end = int(window[0]), int(window[1])
    start = max(start, 0)
    end = min(end, n_layers - 1)
    if start > end:
        raise ValueError(
            f"VISTA window is empty: n_layers={n_layers}, resolved=({start}, {end})."
        )
    return (start, end)


def resolve_sla_window(
    n_layers: int, window: Optional[Tuple[int, int]] = None
) -> Tuple[int, int]:
    """Resolve the (inclusive) SLA layer window.

    Default is the depth-proportional ``(round(25L/32), round(30L/32))`` (the
    paper's ``(25, 30)`` on 32 layers scaled by depth; ``(19, 22)`` on SmolVLM's
    24). An explicit window is clamped to ``[0, L-1]``.
    """
    if window is None:
        start = round(25 * n_layers / 32)
        end = round(30 * n_layers / 32)
    else:
        start, end = int(window[0]), int(window[1])
    start = max(start, 0)
    end = min(end, n_layers - 1)
    if start > end:
        raise ValueError(
            f"SLA window is empty: n_layers={n_layers}, resolved=({start}, {end})."
        )
    return (start, end)


@contextmanager
def enable_vista(model_wrapper, hyperparams: VistaHyperparams) -> Iterator[VistaBuffer]:
    """Install VISTA on a loaded model for the duration of the with-block.

    Resolves the VSV and SLA windows from ``model_wrapper.n_layers``, installs
    the mlp wrapper, and restores everything on exit (including on an exception),
    so a later "VISTA OFF" run is never silently hooked.

    The VSV is NOT computed here: :func:`compute_vsv` must run on the clean model
    BEFORE this context opens (as the official code computes the vector before
    ``add_vsv_layers``), and the caller arms the buffer with
    ``buffer.set_vsv(vsv)`` after entering. Until then every layer is a no-op.

    Yields the :class:`VistaBuffer` (shared state + instrumentation).
    """
    vsv_window = resolve_vsv_window(model_wrapper.n_layers, hyperparams.window)

    buffer = VistaBuffer()
    buffer.reset()
    buffer.lam = float(hyperparams.lam)
    buffer.window = vsv_window
    buffer.sla = bool(hyperparams.sla)
    buffer.sla_alpha = float(hyperparams.sla_alpha)
    buffer.sla_window = (
        resolve_sla_window(model_wrapper.n_layers, hyperparams.sla_window)
        if hyperparams.sla
        else None
    )

    installation = install_vista_hooks(model_wrapper, buffer)
    try:
        yield buffer
    finally:
        installation.remove()
        buffer.reset()
