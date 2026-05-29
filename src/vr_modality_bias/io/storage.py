"""HDF5 persistence for per-(image, condition) hidden states"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from vr_modality_bias.models.base import HiddenStatesResult

__all__ = ["save_hidden_states", "load_hidden_states", "hidden_states_filename"]

_ATTR_KEYS: tuple[str, ...] = (
    "image_id",
    "condition",
    "model_id",
    "prompt_key",
    "caption_start",
    "caption_len",
    "seed_global",
    "noise_seed",
    "caption_ref",
    "timestamp_iso",
    "hidden_dim",
)


def hidden_states_filename(image_id: str, condition: str) -> str:
    """Return the canonical filename for an ``(image_id, condition)`` pair."""
    if condition not in {"A", "B"}:
        raise ValueError(f"condition must be 'A' or 'B', got {condition!r}.")
    return f"{image_id}__{condition}.h5"


def save_hidden_states(
    path: Path,
    result: HiddenStatesResult,
    condition: str,
    *,
    extra_attrs: dict[str, Any] | None = None,
    compression: str | None = "gzip",
    compression_level: int = 4,
) -> Path:
   
    if condition not in {"A", "B"}:
        raise ValueError(f"condition must be 'A' or 'B', got {condition!r}.")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    hs = result.hidden_states.detach().to(dtype=torch.float16, device="cpu").contiguous()
    input_ids = result.input_ids.detach().to(dtype=torch.int64, device="cpu").contiguous()

    compression_opts = (
        {"compression": compression, "compression_opts": compression_level}
        if compression
        else {}
    )

    with h5py.File(path, "w") as f:
        f.create_dataset(
            "hidden_states", data=hs.numpy(), dtype="float16", **compression_opts
        )
        f.create_dataset("input_ids", data=input_ids.numpy(), dtype="int64")

        if result.attention_mask is not None:
            attn = (
                result.attention_mask.detach()
                .to(dtype=torch.int8, device="cpu")
                .contiguous()
                .numpy()
            )
            f.create_dataset("attention_mask", data=attn, dtype="int8")

        attrs = dict(result.metadata)
        if extra_attrs:
            attrs.update(extra_attrs)
        attrs.setdefault("condition", condition)
        attrs.setdefault("caption_start", int(result.caption_start))
        attrs.setdefault("caption_len", int(result.caption_len))
        attrs.setdefault("hidden_dim", int(hs.shape[-1]))

        for key in _ATTR_KEYS:
            if key in attrs and attrs[key] is not None:
                f.attrs[key] = _coerce_attr_value(attrs[key])

        for key, value in attrs.items():
            if key not in _ATTR_KEYS and value is not None and key not in f.attrs:
                f.attrs[key] = _coerce_attr_value(value)

    return path


def load_hidden_states(path: Path) -> HiddenStatesResult:
    """Load an HDF5 file produced by :func:`save_hidden_states`."""
    path = Path(path)
    with h5py.File(path, "r") as f:
        hidden_np = f["hidden_states"][:]
        input_ids_np = f["input_ids"][:]
        attention_mask_np: np.ndarray | None = None
        if "attention_mask" in f:
            attention_mask_np = f["attention_mask"][:]

        metadata = {k: _decode_attr_value(v) for k, v in f.attrs.items()}

    hidden = torch.from_numpy(np.asarray(hidden_np))
    input_ids = torch.from_numpy(np.asarray(input_ids_np)).to(torch.int64)
    attention_mask = (
        torch.from_numpy(np.asarray(attention_mask_np)).to(torch.int8)
        if attention_mask_np is not None
        else None
    )

    caption_start = int(metadata.get("caption_start", 0))
    caption_len = int(metadata.get("caption_len", input_ids.shape[0] - caption_start))

    return HiddenStatesResult(
        hidden_states=hidden,
        input_ids=input_ids,
        caption_start=caption_start,
        caption_len=caption_len,
        metadata=metadata,
        attention_mask=attention_mask,
    )


def _coerce_attr_value(value: Any) -> Any:
    
    if isinstance(value, (str, bytes, int, float, bool, np.generic)):
        return value
    return str(value)


def _decode_attr_value(value: Any) -> Any:
    """Decode an attr value, turning bytes back into ``str`` for convenience."""
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value
    if isinstance(value, np.generic):
        return value.item()
    return value
