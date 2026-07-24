"""Reference-layer selection (Ponto 4): derive SPARC's reference layer from the
diagnostic's per-layer visual-influence (KL) curve as the first layer whose
normalized value crosses a threshold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from loguru import logger

__all__ = ["ReferenceLayerResult", "reference_layer_from_curve"]


@dataclass(frozen=True)
class ReferenceLayerResult:
    recommended_layer: int
    argmax_layer: int
    deep_block_start: int
    normalized_curve: list[float]


def reference_layer_from_curve(kl_per_layer, theta: float = 0.5) -> ReferenceLayerResult:
    """Return the reference layer implied by a per-layer KL curve.

    The recommended layer is the first ``l`` with ``k[l] / max(k) >= theta``:
    the entry to the integration zone, where visual influence stops being
    marginal. ``argmax`` (peak) and ``deep_block_start`` (``floor(2L/3)``, the
    last third) come along for sanity when reading the result.

    Normalization is invariant to positive rescaling of the curve, so the value
    is comparable across model families. Non-finite entries are excluded from
    the max and can never be the reference layer; they stay ``nan`` in the
    returned normalized curve.
    """
    if not (0.0 < theta <= 1.0):
        raise ValueError(f"theta={theta} must be in (0, 1].")

    curve = [float(v) for v in kl_per_layer]
    if len(curve) == 0:
        raise ValueError("empty KL curve.")

    finite_idx = [i for i, v in enumerate(curve) if math.isfinite(v)]
    n_nonfinite = len(curve) - len(finite_idx)
    if n_nonfinite:
        dropped = [i for i, v in enumerate(curve) if not math.isfinite(v)]
        logger.warning(
            f"reference_layer_from_curve: ignoring {n_nonfinite} non-finite "
            f"value(s) at layer(s) {dropped} during normalization."
        )
    if not finite_idx:
        raise ValueError("KL curve has no finite values.")

    max_finite = max(curve[i] for i in finite_idx)
    if max_finite <= 0.0:
        raise ValueError("KL curve is all zero; nothing to normalize.")

    normalized = [
        (v / max_finite) if math.isfinite(v) else float("nan") for v in curve
    ]
    # The argmax normalizes to 1.0 >= theta, so a crossing always exists.
    recommended = next(
        i for i in range(len(curve))
        if math.isfinite(normalized[i]) and normalized[i] >= theta
    )
    argmax = max(finite_idx, key=lambda i: curve[i])
    deep_block_start = (2 * len(curve)) // 3

    return ReferenceLayerResult(
        recommended_layer=recommended,
        argmax_layer=argmax,
        deep_block_start=deep_block_start,
        normalized_curve=normalized,
    )
