"""Abstract interface for VLM wrappers"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
from PIL import Image

__all__ = ["HiddenStatesResult", "ModelWrapper"]


@dataclass
class HiddenStatesResult:
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
