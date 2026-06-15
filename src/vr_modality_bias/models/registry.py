"""Simple key→wrapper-factory registry for VLMs"""

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
    """Eagerly register baseline wrappers without importing heavy deps"""

    def _smolvlm_256m() -> ModelWrapper:
        try:
            from vr_modality_bias.models.smolvlm import SmolVLMWrapper
        except ModuleNotFoundError:
            from src.vr_modality_bias.models.smolvlm import SmolVLMWrapper

        return SmolVLMWrapper(model_id="HuggingFaceTB/SmolVLM-256M-Instruct")

    def _smolvlm_2_2b() -> ModelWrapper:
        try:
            from vr_modality_bias.models.smolvlm import SmolVLMWrapper
        except ModuleNotFoundError:
            from src.vr_modality_bias.models.smolvlm import SmolVLMWrapper

        return SmolVLMWrapper(model_id="HuggingFaceTB/SmolVLM-Instruct")

    def _qwen2_5_vl_7b() -> ModelWrapper:
        try:
            from vr_modality_bias.models.qwen_vl import QwenVLWrapper
        except ModuleNotFoundError:
            from src.vr_modality_bias.models.qwen_vl import QwenVLWrapper

        return QwenVLWrapper(model_id="Qwen/Qwen2.5-VL-7B-Instruct")

    register_model("smolvlm-256m", _smolvlm_256m)
    register_model("smolvlm-2.2b", _smolvlm_2_2b)
    register_model("qwen2.5-vl-7b", _qwen2_5_vl_7b)


_register_builtin()
