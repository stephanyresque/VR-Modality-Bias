"""Concrete wrapper for ``HuggingFaceTB/SmolVLM-256M-Instruct``.

Implements :class:`vr_modality_bias.models.base.ModelWrapper` for the
Idefics3-based SmolVLM-256M architecture. Heavy imports (``transformers``,
``accelerate``) are deferred to :meth:`SmolVLMWrapper.load` so the module
remains importable without those installed.

References:
    EXPERIMENT.md §6.2 (implementation notes), §4.3 (TF protocol), §12
    (known fragility around ``lm_head`` discovery).
"""

from __future__ import annotations

from typing import Any

import torch
from PIL import Image

from vr_modality_bias.models.base import HiddenStatesResult, ModelWrapper

__all__ = ["SmolVLMWrapper"]

# Candidate attribute paths for the language-modelling head, ordered most→least
# likely for Idefics3-derived architectures. Probed in order; first match wins.
_LM_HEAD_CANDIDATES: tuple[str, ...] = (
    "lm_head",
    "language_model.lm_head",
    "model.lm_head",
    "model.language_model.lm_head",
    "text_model.lm_head",
    "model.text_model.lm_head",
    "model.embed_out",
)

# Candidate paths for the number of language-decoder layers. The Idefics3
# config nests text_config; some wrappers expose num_hidden_layers directly.
_N_LAYERS_CANDIDATES: tuple[str, ...] = (
    "config.text_config.num_hidden_layers",
    "config.num_hidden_layers",
    "config.text_config.n_layer",
)


def _resolve_attr(root: Any, dotted: str) -> Any:
    """Return ``root.a.b.c`` for ``dotted="a.b.c"`` or raise :class:`AttributeError`."""
    obj = root
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


class SmolVLMWrapper(ModelWrapper):
    """SmolVLM-256M-Instruct wrapper (Idefics3 family)."""

    def __init__(
        self,
        model_id: str = "HuggingFaceTB/SmolVLM-256M-Instruct",
        *,
        dtype: torch.dtype = torch.float16,
        attn_implementation: str = "eager",
    ) -> None:
        self.model_id = model_id
        self._dtype = dtype
        self._attn_implementation = attn_implementation
        self._device: torch.device | None = None
        self._model = None  # transformers.PreTrainedModel after load()
        self._processor = None  # transformers.ProcessorMixin
        self._lm_head: torch.nn.Module | None = None
        self._n_layers: int | None = None

    # ------------------------------------------------------------------ load

    def load(self, device: torch.device) -> None:
        """Load processor + model onto ``device`` and resolve ``lm_head``/``n_layers``."""
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self._device = device
        self._processor = AutoProcessor.from_pretrained(self.model_id)

        # bf16/fp16 are the canonical inference dtypes; on CPU we fall back to fp32
        # because half-precision matmul on CPU is dramatically slower and rarely
        # representative of the target hardware.
        effective_dtype = self._dtype if device.type == "cuda" else torch.float32

        model_kwargs: dict[str, Any] = {"torch_dtype": effective_dtype}
        if device.type == "cuda":
            model_kwargs["_attn_implementation"] = self._attn_implementation

        self._model = AutoModelForVision2Seq.from_pretrained(
            self.model_id, **model_kwargs
        ).to(device)
        self._model.eval()

        self._lm_head = self._discover_lm_head()
        self._n_layers = self._discover_n_layers()

    # ----------------------------------------------------------- introspect

    def _discover_lm_head(self) -> torch.nn.Module:
        for path in _LM_HEAD_CANDIDATES:
            try:
                head = _resolve_attr(self._model, path)
            except AttributeError:
                continue
            if isinstance(head, torch.nn.Module):
                return head
        raise RuntimeError(
            f"Could not locate lm_head on {type(self._model).__name__}. "
            f"Tried: {list(_LM_HEAD_CANDIDATES)}. "
            "See EXPERIMENT.md §12 — investigate before relaxing this check."
        )

    def _discover_n_layers(self) -> int:
        for path in _N_LAYERS_CANDIDATES:
            try:
                value = _resolve_attr(self._model, path)
            except AttributeError:
                continue
            if isinstance(value, int) and value > 0:
                return value
        raise RuntimeError(
            f"Could not determine n_layers on {type(self._model).__name__}. "
            f"Tried: {list(_N_LAYERS_CANDIDATES)}."
        )

    @property
    def n_layers(self) -> int:
        if self._n_layers is None:
            raise RuntimeError("Model not loaded — call .load() first.")
        return self._n_layers

    def get_lm_head(self) -> torch.nn.Module:
        if self._lm_head is None:
            raise RuntimeError("Model not loaded — call .load() first.")
        return self._lm_head

    # ------------------------------------------------------------ generate

    def generate_caption(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int,
        seed: int,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> str:
        """Free generation. Seeded per-call for per-image determinism."""
        if self._model is None or self._processor is None:
            raise RuntimeError("Model not loaded — call .load() first.")

        messages = self._build_messages(prompt)
        prompt_text = self._processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = self._processor(
            text=prompt_text, images=[image.convert("RGB")], return_tensors="pt"
        ).to(self._device)

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": True,
            "temperature": 0.8,
            "top_p": 0.9,
            "repetition_penalty": 1.0,
        }
        if generation_kwargs:
            gen_kwargs.update(generation_kwargs)

        torch.manual_seed(int(seed))
        if self._device is not None and self._device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))

        with torch.no_grad():
            generated = self._model.generate(**inputs, **gen_kwargs)

        prefix_len = inputs["input_ids"].shape[-1]
        new_tokens = generated[0, prefix_len:]
        text = self._processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text.strip()

    # ----------------------------------------------------- teacher forcing

    def run_teacher_forcing(
        self,
        image: Image.Image,
        prompt: str,
        caption_ref: str,
    ) -> HiddenStatesResult:
        """Forward pass with TF; returns stacked layer hidden states (fp16 CPU)."""
        if self._model is None or self._processor is None or self._device is None:
            raise RuntimeError("Model not loaded — call .load() first.")

        image_rgb = image.convert("RGB")
        messages = self._build_messages(prompt)
        prefix_text = self._processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        # Length of the prefix once the image tokens have been inserted by the
        # processor. The caption begins at this index in the full sequence.
        prefix_inputs = self._processor(
            text=prefix_text, images=[image_rgb], return_tensors="pt"
        )
        caption_start = int(prefix_inputs["input_ids"].shape[-1])

        full_text = prefix_text + caption_ref.strip()
        full_inputs = self._processor(
            text=full_text, images=[image_rgb], return_tensors="pt"
        )

        # Sanity check: the prefix must remain a prefix of the full input.
        full_ids = full_inputs["input_ids"][0]
        prefix_ids = prefix_inputs["input_ids"][0]
        if not torch.equal(full_ids[:caption_start], prefix_ids):
            raise RuntimeError(
                "Tokenisation of prefix changed when caption was appended. "
                "Cannot derive caption_start safely. "
                "Investigate before relaxing — see EXPERIMENT.md §12."
            )
        caption_len = int(full_ids.shape[0] - caption_start)
        if caption_len <= 0:
            raise RuntimeError(
                f"caption_len <= 0 ({caption_len}); caption_ref tokenised to nothing. "
                f"caption_ref={caption_ref!r}"
            )

        full_inputs = full_inputs.to(self._device)
        with torch.no_grad():
            outputs = self._model(
                **full_inputs,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

        # outputs.hidden_states is a tuple of length (n_layers + 1):
        # index 0 is the embedding output; indices 1..L are the layer outputs.
        # Per EXPERIMENT.md §4.4 the comparison runs over l ∈ [1, L], so drop 0.
        layer_states = outputs.hidden_states[1:]
        if len(layer_states) != self.n_layers:
            raise RuntimeError(
                f"Unexpected hidden_states count: got {len(layer_states)} "
                f"layer states, expected n_layers={self.n_layers}."
            )
        hidden = torch.stack(layer_states, dim=0).squeeze(1)  # (L, T, D)
        hidden = hidden.to(dtype=torch.float16, device="cpu", copy=True).contiguous()

        input_ids_cpu = full_ids.to(device="cpu", dtype=torch.int64).contiguous()
        attention_mask = full_inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = (
                attention_mask[0].to(device="cpu", dtype=torch.int8).contiguous()
            )

        return HiddenStatesResult(
            hidden_states=hidden,
            input_ids=input_ids_cpu,
            caption_start=caption_start,
            caption_len=caption_len,
            attention_mask=attention_mask,
            metadata={
                "model_id": self.model_id,
                "hidden_dim": int(hidden.shape[-1]),
                "n_layers": int(hidden.shape[0]),
            },
        )

    # --------------------------------------------------------------- helpers

    @staticmethod
    def _build_messages(prompt: str) -> list[dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
