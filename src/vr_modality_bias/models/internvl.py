"""Wrapper for ``OpenGVLab/InternVL3-8B-hf`` — the NATIVE HF checkpoint (the
legacy remote-code ``InternVL2-8B`` breaks on transformers v5). Used only in
the SPARC + CHAIR evaluation stage: ``run_teacher_forcing`` intentionally
raises. Loads in bf16 with eager attention (SPARC patches the forward).
"""

from __future__ import annotations

import sys
from typing import Any

import torch
from PIL import Image
from pyprojroot import here

try:
    from vr_modality_bias.models.base import HiddenStatesResult, ModelWrapper
except ModuleNotFoundError:
    sys.path.insert(0, str(here()))

    from src.vr_modality_bias.models.base import HiddenStatesResult, ModelWrapper

__all__ = ["InternVLWrapper"]


_LM_HEAD_CANDIDATES: tuple[str, ...] = (
    "lm_head",
    "language_model.lm_head",
    "model.lm_head",
    "model.language_model.lm_head",
    "language_model.model.lm_head",
)


_N_LAYERS_CANDIDATES: tuple[str, ...] = (
    "config.text_config.num_hidden_layers",
    "config.llm_config.num_hidden_layers",
    "config.num_hidden_layers",
)


def _resolve_attr(root: Any, dotted: str) -> Any:
    obj = root
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


class InternVLWrapper(ModelWrapper):
    """InternVL3-8B-hf wrapper (evaluation / SPARC-CHAIR only)."""

    def __init__(
        self,
        model_id: str = "OpenGVLab/InternVL3-8B-hf",
        *,
        dtype: torch.dtype = torch.bfloat16,     # bf16, NOT fp16
        attn_implementation: str = "eager",      # required by SPARC
    ) -> None:
        self.model_id = model_id
        self._dtype = dtype
        self._attn_implementation = attn_implementation
        self._device: torch.device | None = None
        self._model = None
        self._processor = None
        self._lm_head: torch.nn.Module | None = None
        self._n_layers: int | None = None

    def load(self, device: torch.device) -> None:
        """Load the model + processor via the native HF interface.

        No ``trust_remote_code`` -- the ``-hf`` checkpoint ships its
        modeling code inside ``transformers``.
        """
        from transformers import AutoProcessor

        self._device = device
        self._processor = AutoProcessor.from_pretrained(self.model_id)

        effective_dtype = self._dtype if device.type == "cuda" else torch.float32

        model_kwargs: dict[str, Any] = {
            "dtype": effective_dtype,
            "low_cpu_mem_usage": True,
        }
        if device.type == "cuda":
            model_kwargs["attn_implementation"] = self._attn_implementation

        self._model = self._load_model(self.model_id, model_kwargs).to(device)
        self._model.eval()

        self._lm_head = self._discover_lm_head()
        self._n_layers = self._discover_n_layers()

    @staticmethod
    def _load_model(model_id: str, model_kwargs: dict[str, Any]):
        """Try the explicit class first, then the generic auto class."""

        import transformers

        candidate_classes = (
            "InternVLForConditionalGeneration",
            "AutoModelForImageTextToText",
        )
        last_exc: Exception | None = None
        for name in candidate_classes:
            cls = getattr(transformers, name, None)
            if cls is None:
                continue
            try:
                return cls.from_pretrained(model_id, **model_kwargs)
            except Exception as exc:
                last_exc = exc
                continue
        raise RuntimeError(
            f"Could not load {model_id} via any of "
            f"{candidate_classes}. Last error: {last_exc!r}"
        )

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
            "Re-run Step 0 (scripts/internvl_inspect.py) and extend "
            "_LM_HEAD_CANDIDATES here if needed."
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
            raise RuntimeError("Model not loaded -- call .load() first.")
        return self._n_layers

    def get_lm_head(self) -> torch.nn.Module:
        if self._lm_head is None:
            raise RuntimeError("Model not loaded -- call .load() first.")
        return self._lm_head

    def generate_caption(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int,
        seed: int,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> str:
        """Free generation via ``processor.apply_chat_template`` + ``model.generate``.

        Identical shape to ``QwenVLWrapper.generate_caption`` -- the
        native InternVL processor exposes the same interface. Any SPARC
        monkey-patch on ``self_attn.forward`` runs inside ``model.generate``
        as usual.
        """
        if self._model is None or self._processor is None:
            raise RuntimeError("Model not loaded -- call .load() first.")

        image_rgb = image.convert("RGB")
        messages = self._build_messages(prompt, image_rgb)
        prompt_text = self._processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        inputs = self._processor(
            text=[prompt_text],
            images=[image_rgb],
            return_tensors="pt",
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

        text = self._processor.tokenizer.decode(
            new_tokens, skip_special_tokens=True
        )
        return text.strip()

    def run_teacher_forcing(
        self,
        image: Image.Image,
        prompt: str,
        caption_ref: str,
    ) -> HiddenStatesResult:
        """Not implemented -- diagnostic stage stays on SmolVLM only.

        The InternVL family is added exclusively for the SPARC + CHAIR
        evaluation. Hidden-state collection (Section IV of the paper)
        is not run on InternVL, so this method intentionally raises.
        """
        raise NotImplementedError(
            "InternVLWrapper.run_teacher_forcing is intentionally not "
            "implemented: the InternVL family is used only for the SPARC "
            "+ CHAIR evaluation, not the hidden-state diagnostic. See the "
            "wrapper docstring."
        )

    @staticmethod
    def _build_messages(prompt: str, image: Image.Image) -> list[dict[str, Any]]:
        """Standard v5 VLM messages format (same as Qwen2.5-VL)."""
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
