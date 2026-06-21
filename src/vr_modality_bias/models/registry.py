"""Simple key→wrapper-factory registry for VLMs."""

from __future__ import annotations

import sys
from collections.abc import Callable

from pyprojroot import here

try:
    from vr_modality_bias.models.base import ModelWrapper
except ModuleNotFoundError:
    sys.path.insert(0, str(here()))

    from src.vr_modality_bias.models.base import ModelWrapper

__all__ = ["build_model", "list_models", "register_model"]

_FACTORIES: dict[str, Callable[[], ModelWrapper]] = {}


def register_model(key: str, factory: Callable[[], ModelWrapper]) -> None:
    """Register ``factory`` under ``key``. Re-registration is allowed."""
    _FACTORIES[key] = factory


def build_model(key: str) -> ModelWrapper:
    """Instantiate (but do **not** load) the wrapper registered under ``key``."""
    try:
        factory = _FACTORIES[key]
    except KeyError as exc:
        raise KeyError(
            f"Unknown model key {key!r}. Registered: {sorted(_FACTORIES)}."
        ) from exc
    return factory()


def list_models() -> list[str]:
    """Return the sorted list of registered model keys."""
    return sorted(_FACTORIES)


def _register_builtin() -> None:
    """Eagerly register baseline wrappers without importing heavy deps."""

    def _llava_1_5_7b() -> ModelWrapper:
        # The single supported family post-migration. Wrapper at
        # models/llava.py; SPARC infra (utils/attn.py) dispatches LLaVA
        # to the llama forward via detect_model_family.
        try:
            from vr_modality_bias.models.llava import LlavaWrapper
        except ModuleNotFoundError:
            from src.vr_modality_bias.models.llava import LlavaWrapper

        return LlavaWrapper(model_id="llava-hf/llava-1.5-7b-hf")

    register_model("llava-1.5-7b", _llava_1_5_7b)


_register_builtin()
