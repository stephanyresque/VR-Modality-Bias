"""Abstract interface for VLM wrappers.

EXPERIMENT.md §6.1. Subclasses implement the concrete (load, generate,
teacher-forced forward, lm_head/n_layers introspection) for a specific
model. ``SmolVLMWrapper`` is the only concrete class active in the
baseline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
from PIL import Image

__all__ = ["HiddenStatesResult", "ModelWrapper"]


@dataclass
class HiddenStatesResult:
    """Output of a single teacher-forced forward pass.

    Attributes:
        hidden_states: Stacked layer outputs with shape
            ``(n_layers, seq_len, hidden_dim)`` (the embedding output at
            index 0 of ``outputs.hidden_states`` is **dropped**, per
            EXPERIMENT.md §4.4 which indexes layers in ``[1, L]``).
            Typically fp16 on CPU by the time it is returned.
        input_ids: 1-D ``int64`` token ids, length ``seq_len``.
        caption_start: Index of the first caption token in ``input_ids``.
            All comparisons in §4.4 read positions ``t - 1`` for
            ``t in [caption_start, caption_start + caption_len)``.
        caption_len: Number of caption tokens
            (``len(input_ids) - caption_start``).
        metadata: Free-form per-run metadata (``model_id``, ``prompt_key``,
            seeds, etc.). Persisted to HDF5 attrs by ``io.storage``.
        attention_mask: Optional 1-D attention mask matching ``input_ids``.
    """

    hidden_states: torch.Tensor
    input_ids: torch.Tensor
    caption_start: int
    caption_len: int
    metadata: dict[str, Any] = field(default_factory=dict)
    attention_mask: torch.Tensor | None = None


class ModelWrapper(ABC):
    """Abstract VLM wrapper consumed by ``experiment/`` orchestration."""

    model_id: str

    @abstractmethod
    def load(self, device: torch.device) -> None:
        """Load the underlying model and processor onto ``device``."""

    @abstractmethod
    def generate_caption(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int,
        seed: int,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> str:
        """Free generation. Produces the ``caption_ref`` used as TF target."""

    @abstractmethod
    def run_teacher_forcing(
        self,
        image: Image.Image,
        prompt: str,
        caption_ref: str,
    ) -> HiddenStatesResult:
        """Forward pass with teacher forcing; returns stacked hidden states."""

    @abstractmethod
    def get_lm_head(self) -> torch.nn.Module:
        """Return the language-modelling head (the ``lm_head`` linear).

        Required by ``metrics.kl`` to project hidden states into the
        vocabulary logit space.
        """

    @property
    @abstractmethod
    def n_layers(self) -> int:
        """Number of transformer layers in the language decoder."""
