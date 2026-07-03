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

    def _qwen2_5_vl_3b() -> ModelWrapper:
        # Smallest variant in the Qwen2.5-VL family — same forward / mRoPE /
        # mm_token_type_ids code path as 7B. Used by scripts/13 for the fp32
        # architectural-exactness gate (Phase 1, item 1).
        try:
            from vr_modality_bias.models.qwen_vl import QwenVLWrapper
        except ModuleNotFoundError:
            from src.vr_modality_bias.models.qwen_vl import QwenVLWrapper

        return QwenVLWrapper(model_id="Qwen/Qwen2.5-VL-3B-Instruct")

    def _llava_1_5_7b() -> ModelWrapper:
        # New target family for the study — LLaVA-1.5-7B is the canonical
        # baseline of the SPARC paper. Wrapper at models/llava.py; the
        # other families (smolvlm-*, qwen2.5-vl-*) stay registered until
        # the migration block-2 retires them.
        try:
            from vr_modality_bias.models.llava import LlavaWrapper
        except ModuleNotFoundError:
            from src.vr_modality_bias.models.llava import LlavaWrapper

        return LlavaWrapper(model_id="llava-hf/llava-1.5-7b-hf")

    def _internvl2_8b() -> ModelWrapper:
        # Fourth family added for the SPARC + CHAIR evaluation ONLY.
        # Diagnostic stage (hidden states / share_tail) stays on SmolVLM.
        # Wrapper at models/internvl.py; SPARC forward at
        # utils/attn.py::forward_internlm2. Both need Step 0
        # (scripts/25_internvl_inspect.py) + Step 6.3
        # (scripts/26_internvl_exactness_gate.py) to have passed before
        # the results can be trusted.
        try:
            from vr_modality_bias.models.internvl import InternVLWrapper
        except ModuleNotFoundError:
            from src.vr_modality_bias.models.internvl import InternVLWrapper

        return InternVLWrapper(model_id="OpenGVLab/InternVL2-8B")

    register_model("smolvlm-256m", _smolvlm_256m)
    register_model("smolvlm-2.2b", _smolvlm_2_2b)
    register_model("qwen2.5-vl-3b", _qwen2_5_vl_3b)
    register_model("qwen2.5-vl-7b", _qwen2_5_vl_7b)
    register_model("llava-1.5-7b", _llava_1_5_7b)
    register_model("internvl2-8b", _internvl2_8b)


_register_builtin()
