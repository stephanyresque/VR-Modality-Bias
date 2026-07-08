"""Thin facade over :mod:`vr_modality_bias.utils.attn` that installs SPARC as
a context manager, so collection code can flip it on/off per block and never
leave a monkey-patched model behind (which would contaminate later runs).
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from types import MethodType
from typing import Iterator

from PIL import Image
from pyprojroot import here

try:
    from vr_modality_bias.utils.attn import (
        SelectedIndexBuffer,
        add_custom_attention_layers,
        decoder_of,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(here()))

    from src.vr_modality_bias.utils.attn import (
        SelectedIndexBuffer,
        add_custom_attention_layers,
        decoder_of,
    )

__all__ = [
    "SparcHyperparams",
    "enable_sparc",
    "probe_image_token_index",
]


class SparcHyperparams:
    """Plain container for SPARC hyperparameters."""

    def __init__(
        self,
        *,
        alpha: float = 1.3,
        tau: float = 2.0,
        selected_layer: int = 15,
        se_layers: tuple[int, int] = (0, 31),
        beta: float = 0.0,
    ) -> None:
        if alpha <= 1.0:
            raise ValueError(
                f"alpha={alpha} would make SPARC a no-op. Pass alpha > 1."
            )
        self.alpha = float(alpha)
        self.tau = float(tau)
        self.selected_layer = int(selected_layer)
        self.se_layers = (int(se_layers[0]), int(se_layers[1]))
        self.beta = float(beta)

    def as_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "tau": self.tau,
            "selected_layer": self.selected_layer,
            "se_layers": list(self.se_layers),
            "beta": self.beta,
        }


def probe_image_token_index(
    model_wrapper,
    image: Image.Image,
    prompt: str,
) -> tuple[int, int, int]:
    """Return ``(image_token_index, input_len, num_image_patches)``.

    Same helper that ``scripts/inference_sparc.py`` uses, lifted here so
    callers don't have to import a script. ``input_len`` is the prompt
    length excluding image patches, which is what
    :meth:`SelectedIndexBuffer.update_input_len` expects.
    """
    image_token_id = int(model_wrapper._model.config.image_token_id)
    messages = model_wrapper._build_messages(prompt, image)
    prompt_text = model_wrapper._processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    inputs = model_wrapper._processor(
        text=[prompt_text], images=[image], return_tensors="pt"
    )
    input_ids = inputs["input_ids"][0]
    image_positions = (input_ids == image_token_id).nonzero(as_tuple=True)[0]
    total_len = int(input_ids.shape[-1])
    num_image_patches = int(image_positions.numel())

    return int(image_positions[0]), total_len - num_image_patches, num_image_patches


def _decoder_of(model) -> object:
    """Locate the decoder module (with ``.layers``) for Qwen and Idefics3.

    Delegates to ``utils.attn.decoder_of``, which knows both the Qwen
    (``model.model.language_model``) and SmolVLM / Idefics3
    (``model.model.text_model``) paths.
    """
    return decoder_of(model)


@contextmanager
def enable_sparc(
    model_wrapper,
    *,
    hparams: SparcHyperparams,
    probe_image: Image.Image,
    prompt: str,
) -> Iterator[SelectedIndexBuffer]:
    """Install SPARC on a loaded model for the duration of the with-block.

    Why a context manager: ``add_custom_attention_layers`` monkey-patches
    every decoder layer's ``self_attn.forward``. If we forget to undo it,
    later "baseline" measurements will silently be SPARC measurements. The
    enter/exit cycle here snapshots and restores the originals.

    Args:
        model_wrapper: Loaded :class:`ModelWrapper`.
        hparams: :class:`SparcHyperparams` with the calibration coefficient
            and the rest of the SPARC settings.
        probe_image: Any image from the manifest — used to discover
            ``image_token_index`` once. The collection function still has
            to reset the buffer and refresh ``input_len`` **per image**
            (different images can have different patch counts).
        prompt: Same prompt the collection will use; needed so the chat
            template renders the same tokens during the probe.

    Yields:
        The shared :class:`SelectedIndexBuffer` (passed to
        ``collect_forced_decoding`` via its ``sparc_buffer`` argument).
    """
    model = model_wrapper._model
    decoder = _decoder_of(model)

    # Snapshot the original forwards so we can restore them after the block.
    originals = [layer.self_attn.forward for layer in decoder.layers]

    image_token_index, input_len, _ = probe_image_token_index(
        model_wrapper, probe_image.convert("RGB"), prompt
    )

    buffer = SelectedIndexBuffer()
    buffer.reset()

    buffer.update_input_len(input_len) 

    add_custom_attention_layers(
        model,
        alpha=hparams.alpha,
        beta=hparams.beta,
        tau=hparams.tau,
        selected_layer=hparams.selected_layer,
        se_layers=hparams.se_layers,
        image_token_index=image_token_index,
        indices_buffer=buffer,
    )

    try:
        yield buffer
    finally:
        # Restore originals — important even on exceptions, otherwise the
        # next "SPARC OFF" run would still be running through the patched
        # forwards.
        for layer, original in zip(decoder.layers, originals):
            # ``add_custom_attention_layers`` stored a partial bound via
            # ``MethodType``; we just replace it back with the original.
            layer.self_attn.forward = original  # type: ignore[assignment]
        # Reset buffer so a re-entry starts clean.
        buffer.reset()


# Re-export for convenience: callers usually import both from here.
__all__.append("SelectedIndexBuffer")
