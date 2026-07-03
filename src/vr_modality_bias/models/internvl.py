"""Concrete wrapper for ``OpenGVLab/InternVL2-8B`` (evaluation only).

Only Passos 1-2 of the InternVL bloco are wired here (load + free
generation). ``run_teacher_forcing`` intentionally raises: the InternVL
family is used exclusively in the intervention-evaluation stage
(SPARC + CHAIR); the diagnostic (Section IV: hidden states, share_tail,
heatmaps) stays on SmolVLM-2.2B and is NOT ported here.

Architecture notes (see Passo 0 -- ``scripts/25_internvl_inspect.py``):

* Top class      : ``InternVLChatModel``  (matched by _INTERNLM2_MARKERS
                   in utils/attn.py). Verify with the Step-0 dump.
* LM backbone    : ``model.language_model`` -> ``InternLM2ForCausalLM``.
* Decoder layers : ``model.language_model.model.layers``. decoder_of
                   already covers this via the extra-nesting branch.
* Attention attr : ``layer.attention`` (NOT ``layer.self_attn``). The
                   ``_attention_module_of`` helper in utils/attn.py
                   dispatches transparently.
* Visual token   : ``<IMG_CONTEXT>`` -- id comes from the tokenizer,
                   NOT ``config.image_token_id`` (which InternVL doesn't
                   populate). Consumers should use ``tokenizer.
                   convert_tokens_to_ids('<IMG_CONTEXT>')`` -- exposed
                   here as :attr:`image_token_id`.
* dtype          : bf16 (InternVL was trained in bf16; do NOT use fp16).
* Attention impl : eager (SPARC patches the forward -- flash/sdpa
                   attention paths hide attn_weights).

Preprocessing:
    InternVL uses a dynamic 448x448 tiling (multi-crop) that the
    ``AutoProcessor`` doesn't currently expose in a stable way. If the
    generation API of choice is ``model.chat(tokenizer, pixel_values,
    question, generation_config)`` (which the official model card uses),
    we lean on it: it embeds the tiling / prompt-format / image-token
    injection internally and hands us clean text back.
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


# Candidate paths for the LM head. Reused from qwen_vl / smolvlm patterns,
# with the InternVL-specific ``language_model.output`` added first
# (some InternLM2 remote code names its head ``output`` instead of ``lm_head``).
_LM_HEAD_CANDIDATES: tuple[str, ...] = (
    "language_model.output",              # InternLM2 remote-code convention
    "language_model.lm_head",             # standard HF convention
    "language_model.model.output",
    "language_model.model.lm_head",
    "lm_head",
    "output",
    "model.language_model.lm_head",
    "model.language_model.model.lm_head",
)

# Depth candidates. The InternVL config nests LM config under llm_config.
_N_LAYERS_CANDIDATES: tuple[str, ...] = (
    "config.llm_config.num_hidden_layers",
    "config.text_config.num_hidden_layers",
    "config.num_hidden_layers",
)


def _resolve_attr(root: Any, dotted: str) -> Any:
    obj = root
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


class InternVLWrapper(ModelWrapper):
    """InternVL2-8B wrapper (evaluation / SPARC-CHAIR only)."""

    def __init__(
        self,
        model_id: str = "OpenGVLab/InternVL2-8B",
        *,
        dtype: torch.dtype = torch.bfloat16,     # bf16, NOT fp16
        attn_implementation: str = "eager",      # required by SPARC
    ) -> None:
        self.model_id = model_id
        self._dtype = dtype
        self._attn_implementation = attn_implementation
        self._device: torch.device | None = None
        self._model = None
        self._tokenizer = None       # InternVL uses a tokenizer directly (chat API)
        self._image_token_id: int | None = None
        self._lm_head: torch.nn.Module | None = None
        self._n_layers: int | None = None

    # ------------------------------------------------------------ load

    def load(self, device: torch.device) -> None:
        """Load model + tokenizer with ``trust_remote_code=True``.

        InternVL2 requires trust_remote_code because its modeling code
        (image tiling, chat template, cache convention) lives on the
        Hub, not in ``transformers``.
        """
        from transformers import AutoModel, AutoTokenizer

        self._device = device
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True,
        )

        # eager attention is required so SPARC can see attn_weights.
        effective_dtype = self._dtype if device.type == "cuda" else torch.float32
        model_kwargs: dict[str, Any] = {
            "torch_dtype": effective_dtype,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if device.type == "cuda":
            model_kwargs["attn_implementation"] = self._attn_implementation

        self._model = AutoModel.from_pretrained(self.model_id, **model_kwargs).to(device)
        self._model.eval()

        # Resolve <IMG_CONTEXT> id -- InternVL's visual token marker.
        # ``probe_image_token_index`` and the SPARC per-id mask will use
        # this instead of ``config.image_token_id`` (which InternVL
        # doesn't populate reliably).
        try:
            self._image_token_id = int(
                self._tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
            )
        except Exception:  # pragma: no cover -- Step 0 should catch this
            self._image_token_id = None

        self._lm_head = self._discover_lm_head()
        self._n_layers = self._discover_n_layers()

    # ------------------------------------------------------------ introspection

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
            "Re-run Step 0 (scripts/25_internvl_inspect.py) and extend the "
            "candidate list here."
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

    @property
    def image_token_id(self) -> int:
        """<IMG_CONTEXT> token id. Used by SPARC's per-id mask."""
        if self._image_token_id is None:
            raise RuntimeError(
                "<IMG_CONTEXT> token not resolved -- was the tokenizer "
                "loaded with trust_remote_code=True? Check Step 0 output."
            )
        return self._image_token_id

    # ------------------------------------------------------------ generation

    def generate_caption(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int,
        seed: int,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> str:
        """Free generation via the InternVL ``model.chat`` API.

        The chat method internally handles the dynamic tiling, prompt
        formatting, and image-token injection. SPARC patches
        ``layer.attention.forward`` -- ``model.chat`` funnels through
        the same forward path via ``.generate`` internally, so the
        patch takes effect. If ``chat`` bypasses that path in the
        version we get, the fallback is to build ``pixel_values`` +
        input_ids by hand and call ``model.generate`` -- Passo 6.1
        confirms which flow the deployed remote code uses.
        """
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not loaded -- call .load() first.")

        # InternVL's official image transform: dynamic tiling to 448x448 tiles.
        # Exposed by the remote code as ``load_image`` in the model card.
        pixel_values = self._preprocess_image(image)

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": True,
            "temperature": 0.8,
            "top_p": 0.9,
            "repetition_penalty": 1.0,
        }
        if generation_kwargs:
            gen_kwargs.update(generation_kwargs)

        # Per-call seed so generations are reproducible per image_id.
        torch.manual_seed(int(seed))
        if self._device is not None and self._device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))

        with torch.no_grad():
            response = self._model.chat(
                self._tokenizer,
                pixel_values,
                prompt,
                gen_kwargs,
                history=None,
                return_history=False,
            )
        return str(response).strip()

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """InternVL dynamic-tile transform (448x448, up to N tiles).

        This mirrors ``load_image`` from the InternVL2 model card. Passo
        6.1's smoke (1 image, base generation) confirms it produces
        coherent captions -- if the transform is wrong, captions come
        out garbled even in the baseline.
        """
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode

        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)

        def _build_transform(input_size: int) -> T.Compose:
            return T.Compose([
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize(
                    (input_size, input_size),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])

        # Single tile at 448 -- the simplest transform matching the InternVL
        # official ``load_image(image, max_num=1)`` case. Multi-crop (max_num=6
        # or 12) is the paper default; enabling it here requires porting the
        # ``dynamic_preprocess`` logic from the model card. Passo 6.1 will
        # tell us whether single-tile captions are already coherent; if not,
        # port the full dynamic preprocess.
        transform = _build_transform(input_size=448)
        pixel_values = transform(image).unsqueeze(0).to(dtype=self._dtype)
        if self._device is not None:
            pixel_values = pixel_values.to(self._device)
        return pixel_values

    # ------------------------------------------------------------ TF (unused)

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

    # ------------------------------------------------------------ chat template

    @staticmethod
    def _build_messages(prompt: str, image: Image.Image | None = None) -> list[dict[str, Any]]:
        """Signature-polymorphic with the other wrappers.

        InternVL's ``model.chat`` doesn't take the list-of-dicts format;
        the raw prompt string is enough. Kept for cross-family call
        sites so they don't have to branch on family type.
        """
        del image
        return [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]}]
