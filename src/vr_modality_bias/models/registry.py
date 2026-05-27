"""Simple key→wrapper-factory registry for VLMs.

Allows ``experiment/`` scripts to ask for ``"smolvlm-256m"`` and receive a
fresh, unloaded :class:`ModelWrapper`. Add new model families by registering
their factories here (deferred per EXPERIMENT.md §9).
"""

from __future__ import annotations

from collections.abc import Callable

from vr_modality_bias.models.base import ModelWrapper

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
    """Eagerly register baseline wrappers without importing heavy deps.

    The factory closures defer the ``from vr_modality_bias.models.smolvlm
    import SmolVLMWrapper`` until ``build_model`` is actually called, so
    importing ``registry`` does not pull in ``transformers``.
    """

    def _smolvlm_256m() -> ModelWrapper:
        from vr_modality_bias.models.smolvlm import SmolVLMWrapper

        return SmolVLMWrapper(model_id="HuggingFaceTB/SmolVLM-256M-Instruct")

    register_model("smolvlm-256m", _smolvlm_256m)


_register_builtin()
