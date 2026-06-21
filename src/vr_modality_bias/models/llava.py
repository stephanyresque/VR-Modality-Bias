"""Concrete wrapper for ``llava-hf/llava-1.5-7b-hf``.

LLaVA-1.5 is the canonical baseline of the SPARC paper (their reference
results are on the same backbone family). This wrapper exists so the
study can pivot to LLaVA-1.5-7B without rewriting any of the experiment
orchestration — same :class:`ModelWrapper` contract as
:class:`SmolVLMWrapper` and :class:`QwenVLWrapper`.

Family-specific notes
---------------------
* Prompt format: LLaVA-1.5 uses ``"USER: <image>\\n{prompt} ASSISTANT:"``.
  Modern (5.x) transformers ships a chat template on the processor that
  renders the canonical list-of-content-dicts format into that exact
  string, so we prefer ``processor.apply_chat_template`` when it's
  registered. The manual fallback below produces the same string for
  environments where the template isn't installed.
* Image token: the processor inserts ``<image>`` (id depends on the
  vocab) and the LLaVA model expands it internally to 576 visual feature
  tokens. We never touch this manually — same convention as the other
  wrappers.
* Decoder: 32 transformer layers (LlamaModel, ``config.text_config.
  num_hidden_layers``). ``lm_head`` resolves at the top level.
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

__all__ = ["LlavaWrapper"]


# Same candidate lists as SmolVLMWrapper — the resolution order is what
# eventually picks up ``lm_head`` (top-level) and
# ``config.text_config.num_hidden_layers`` (= 32 for 7B) on LLaVA.
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
)


def _resolve_attr(root: Any, dotted: str) -> Any:
    """Return ``root.a.b.c`` for ``dotted="a.b.c"`` or raise :class:`AttributeError`."""
    obj = root
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


class LlavaWrapper(ModelWrapper):
    """LLaVA-1.5-7B wrapper (LlamaForCausalLM backbone + CLIP vision tower)."""

    def __init__(
        self,
        model_id: str = "llava-hf/llava-1.5-7b-hf",
        *,
        dtype: torch.dtype = torch.float16,
        attn_implementation: str = "eager",
    ) -> None:
        self.model_id = model_id
        self._dtype = dtype
        self._attn_implementation = attn_implementation
        self._device: torch.device | None = None
        self._model = None  # transformers.PreTrainedModel after load()
        self._processor = None  # transformers.LlavaProcessor
        self._lm_head: torch.nn.Module | None = None
        self._n_layers: int | None = None

    # ------------------------------------------------------------ load

    def load(self, device: torch.device) -> None:
        """Load processor + model onto ``device`` and resolve ``lm_head``/``n_layers``."""
        # ``AutoModelForImageTextToText`` is the modern (5.x) entry point
        # and dispatches to ``LlavaForConditionalGeneration`` for LLaVA-1.5.
        # The except branch keeps a direct fallback in case the env still
        # lacks the auto class (e.g. older 4.x) or the dispatcher fails to
        # match the checkpoint.
        from transformers import AutoProcessor

        self._device = device
        self._processor = AutoProcessor.from_pretrained(self.model_id)

        effective_dtype = self._dtype if device.type == "cuda" else torch.float32

        model_kwargs: dict[str, Any] = {"torch_dtype": effective_dtype}
        if device.type == "cuda":
            model_kwargs["_attn_implementation"] = self._attn_implementation

        try:
            from transformers import AutoModelForImageTextToText as _AutoModel
            self._model = _AutoModel.from_pretrained(self.model_id, **model_kwargs)
        except (ImportError, Exception):  # pragma: no cover (fallback path)
            from transformers import LlavaForConditionalGeneration
            self._model = LlavaForConditionalGeneration.from_pretrained(
                self.model_id, **model_kwargs,
            )
        self._model = self._model.to(device)
        self._model.eval()

        self._lm_head = self._discover_lm_head()
        self._n_layers = self._discover_n_layers()

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
            f"Tried: {list(_LM_HEAD_CANDIDATES)}. Inspect the loaded model "
            "and extend _LM_HEAD_CANDIDATES in llava.py if needed."
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

    # ------------------------------------------------------------ prompt

    def _format_prompt(self, prompt: str) -> str:
        """Render ``prompt`` into the LLaVA-1.5 chat string.

        Prefer the processor's chat template (the canonical path on
        transformers 5.x). Fall back to the manual ``USER: <image>\\n
        {prompt} ASSISTANT:`` format if the template isn't registered —
        same string, just produced by hand. Either way, the processor
        expands ``<image>`` to its 576 patch positions when called with
        ``images=[...]``.
        """
        if self._processor is None:
            raise RuntimeError("Model not loaded — call .load() first.")

        chat_template = getattr(self._processor, "chat_template", None)
        if chat_template:
            messages = self._build_messages(prompt, None)
            return self._processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False,
            )
        return f"USER: <image>\n{prompt} ASSISTANT:"

    # ------------------------------------------------------------ generation

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

        prompt_text = self._format_prompt(prompt)
        inputs = self._processor(
            text=prompt_text, images=[image.convert("RGB")], return_tensors="pt",
        ).to(self._device)

        # Greedy by default — matches the official SPARC COCO setup and
        # is required for SPARC stability on long captions (amplification
        # + sampling = bola-de-neve de repetição). Callers may override
        # any of these via ``generation_kwargs`` (e.g. ``do_sample=True``
        # + ``temperature``/``top_p`` for diagnostic non-deterministic
        # runs). The pre-Block-5 default was sampling, which warned about
        # unused temperature/top_p whenever the caller passed do_sample=False.
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": False,
            "num_beams": 1,
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

    # ------------------------------------------------------------ TF

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
        prefix_text = self._format_prompt(prompt)

        # Length of the prefix once <image> has been expanded by the
        # processor (576 visual tokens for LLaVA-1.5). The forced caption
        # begins at this absolute index in the full sequence.
        prefix_inputs = self._processor(
            text=prefix_text, images=[image_rgb], return_tensors="pt",
        )
        caption_start = int(prefix_inputs["input_ids"].shape[-1])

        full_text = prefix_text + caption_ref.strip()
        full_inputs = self._processor(
            text=full_text, images=[image_rgb], return_tensors="pt",
        )

        # Sanity check: the prefix must remain a byte-identical prefix of
        # the full input. If this fires, tokenisation drifted when the
        # caption was appended — same invariant as the other wrappers.
        full_ids = full_inputs["input_ids"][0]
        prefix_ids = prefix_inputs["input_ids"][0]
        if not torch.equal(full_ids[:caption_start], prefix_ids):
            raise RuntimeError(
                "Tokenisation of prefix changed when caption was appended "
                "(LLaVA path). Cannot derive caption_start safely. "
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

    @staticmethod
    def _build_messages(prompt: str, image: Image.Image | None = None) -> list[dict[str, Any]]:
        """Polymorphic with the other wrappers' ``_build_messages``.

        ``image`` is unused (the LLaVA chat template only needs a
        ``{"type": "image"}`` placeholder; the pixel data rides through
        the processor's ``images=`` kwarg). Kept in the signature so
        cross-family call sites in scripts/18 etc. don't have to branch.
        """
        del image  # explicitly unused
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
