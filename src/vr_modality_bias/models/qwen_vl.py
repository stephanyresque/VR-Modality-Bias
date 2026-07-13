"""Concrete wrapper for ``Qwen/Qwen2.5-VL-7B-Instruct``."""

from __future__ import annotations

import sys
from typing import Any

import torch
from loguru import logger
from PIL import Image
from pyprojroot import here

try:
    from vr_modality_bias.models.base import HiddenStatesResult, ModelWrapper
except ModuleNotFoundError:
    sys.path.insert(0, str(here()))

    from src.vr_modality_bias.models.base import HiddenStatesResult, ModelWrapper

__all__ = ["QwenVLWrapper"]

_LM_HEAD_CANDIDATES: tuple[str, ...] = (
    "lm_head",
    "language_model.lm_head",
    "model.lm_head",
    "model.language_model.lm_head",
    "text_model.lm_head",
    "model.text_model.lm_head",
    "model.embed_out",
)

_N_LAYERS_CANDIDATES: tuple[str, ...] = (
    "config.text_config.num_hidden_layers",
    "config.num_hidden_layers",
    "config.text_config.n_layer",
    "config.llm_config.num_hidden_layers",
)

# Final decoder norm (applied before lm_head). Ordered decoder-specific first so
# a wrong ``.norm`` on an outer multimodal wrapper is never picked. Qwen2.5-VL
# keeps it at ``model.language_model.norm``.
_FINAL_NORM_CANDIDATES: tuple[str, ...] = (
    "model.text_model.norm",
    "model.language_model.norm",
    "model.language_model.model.norm",
    "language_model.model.norm",
    "language_model.norm",
    "text_model.norm",
    "model.norm",
    "norm",
)


def _resolve_attr(root: Any, dotted: str) -> Any:
    obj = root
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


class QwenVLWrapper(ModelWrapper):
    """Qwen2.5-VL-7B-Instruct wrapper."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        *,
        dtype: torch.dtype = torch.float16,
        attn_implementation: str = "eager",
    ) -> None:

        self.model_id = model_id
        self._dtype = dtype
        self._attn_implementation = attn_implementation
        self._device: torch.device | None = None
        self._model = None
        self._processor = None
        self._lm_head: torch.nn.Module | None = None
        self._final_norm: torch.nn.Module | None = None
        self._n_layers: int | None = None


    def load(self, device: torch.device) -> None:
        from transformers import AutoProcessor

        self._device = device
        self._processor = AutoProcessor.from_pretrained(self.model_id)

        effective_dtype = self._dtype if device.type == "cuda" else torch.float32

        model_kwargs: dict[str, Any] = {"torch_dtype": effective_dtype}
        if device.type == "cuda":
            model_kwargs["_attn_implementation"] = self._attn_implementation

        self._model = self._load_model(self.model_id, model_kwargs).to(device)
        self._model.eval()

        self._lm_head = self._discover_lm_head()
        self._final_norm = self._discover_final_norm()
        self._n_layers = self._discover_n_layers()

    @staticmethod
    def _load_model(model_id: str, model_kwargs: dict[str, Any]):

        import transformers

        candidate_classes = (
            "Qwen2_5_VLForConditionalGeneration",
            "AutoModelForImageTextToText",
            "AutoModelForVision2Seq",
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
            "Extend _LM_HEAD_CANDIDATES in qwen_vl.py if needed."
        )

    def _discover_final_norm(self) -> torch.nn.Module:
        for path in _FINAL_NORM_CANDIDATES:
            try:
                norm = _resolve_attr(self._model, path)
            except AttributeError:
                continue
            if isinstance(norm, torch.nn.Module):
                return norm
        raise RuntimeError(
            f"Could not locate the final norm on {type(self._model).__name__}. "
            f"Tried: {list(_FINAL_NORM_CANDIDATES)}. "
            "Extend _FINAL_NORM_CANDIDATES in qwen_vl.py if needed."
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

    def get_final_norm(self) -> torch.nn.Module:
        if self._final_norm is None:
            raise RuntimeError("Model not loaded -- call .load() first.")
        return self._final_norm

    def generate_caption(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int,
        seed: int,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> str:
        if self._model is None or self._processor is None:
            raise RuntimeError("Model not loaded — call .load() first.")

        messages = self._build_messages(prompt, image)
        prompt_text = self._processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        inputs = self._processor(
            text=[prompt_text],
            images=[image.convert("RGB")],
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

        # for k, v in inputs.items(): 
        #     logger.debug(f"k: {k} -- {v.device, v.dtype}")

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
        if self._model is None or self._processor is None or self._device is None:
            raise RuntimeError("Model not loaded — call .load() first.")

        image_rgb = image.convert("RGB")
        messages = self._build_messages(prompt, image_rgb)
        prefix_text = self._processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        prefix_inputs = self._processor(
            text=[prefix_text], images=[image_rgb], return_tensors="pt"
        )
        caption_start = int(prefix_inputs["input_ids"].shape[-1])

        full_text = prefix_text + caption_ref.strip()
        full_inputs = self._processor(
            text=[full_text], images=[image_rgb], return_tensors="pt"
        )

        full_ids = full_inputs["input_ids"][0]
        prefix_ids = prefix_inputs["input_ids"][0]
        if not torch.equal(full_ids[:caption_start], prefix_ids):
            raise RuntimeError(
                "Tokenisation of prefix changed when caption was appended. "
                "Cannot derive caption_start safely (Qwen path)."
            )
        caption_len = int(full_ids.shape[0] - caption_start)
        if caption_len <= 0:
            raise RuntimeError(
                f"caption_len <= 0 ({caption_len}); caption_ref empty. "
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

        layer_states = outputs.hidden_states[1:]
        if len(layer_states) != self.n_layers:
            raise RuntimeError(
                f"Unexpected hidden_states count: got {len(layer_states)} "
                f"layer states, expected n_layers={self.n_layers}."
            )
        hidden = torch.stack(layer_states, dim=0).squeeze(1)
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


    @staticmethod
    def _build_messages(prompt: str, image: Image.Image) -> list[dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
