from __future__ import annotations

import math

import torch
import torch.nn as nn
from PIL import Image

__all__ = [
    "SpheRoPETextRotary",
    "SpheRoPEVisionRotary",
    "circular_pad_image",
    "pad_columns",
    "sfc_width_half_angles",
    "width_freq_slots",
]


def sfc_width_half_angles(
    rows: torch.Tensor,
    cols: torch.Tensor,
    n_rows: int,
    n_cols: int,
    inv_freq: torch.Tensor,
    pad_cols: int = 0,
    max_error: float = 0.06,
) -> torch.Tensor:
    rows = rows.to(torch.float64)
    cols = cols.to(torch.float64) - float(pad_cols)
    width = float(n_cols - 2 * pad_cols)
    freqs = inv_freq.detach().to(torch.float64)

    lat = (rows / max(n_rows - 1, 1)) * math.pi - (math.pi / 2.0)
    lon = (cols / width) * 2.0 * math.pi - math.pi
    radius = width / (2.0 * math.pi)
    x = torch.cos(lat) * torch.cos(lon) * radius
    y = torch.cos(lat) * torch.sin(lon) * radius

    fundamental = (2.0 * math.pi) / width
    k = freqs / fundamental
    rounded = torch.round(k)
    error = torch.abs(k - rounded) / (k + 1e-8)
    valid = (k >= 1.0) & (error <= max_error)
    invalid = torch.where(~valid)[0]
    split = int(invalid[0]) if invalid.numel() > 0 else int(freqs.numel())
    harmonic_mask = torch.arange(freqs.numel(), device=freqs.device) < split

    linear = cols.unsqueeze(1) * (rounded * fundamental).unsqueeze(0)
    spherical = torch.empty((rows.numel(), freqs.numel()), dtype=torch.float64, device=freqs.device)
    spherical[:, 0::2] = x.unsqueeze(1) * freqs[0::2].unsqueeze(0)
    spherical[:, 1::2] = y.unsqueeze(1) * freqs[1::2].unsqueeze(0)
    return torch.where(harmonic_mask.unsqueeze(0), linear, spherical).to(torch.float32)


class SpheRoPEVisionRotary(nn.Module):
    def __init__(self, original: nn.Module, pad_cols: int = 0, max_error: float = 0.06) -> None:
        super().__init__()
        self.original = original
        self.pad_cols = int(pad_cols)
        self.max_error = float(max_error)
        self.calls = 0

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        base = self.original(position_ids)
        inv_freq = self.original.inv_freq
        n_freq = int(inv_freq.numel())
        rows = position_ids[:, 0]
        cols = position_ids[:, 1]
        n_rows = int(rows.max()) + 1
        n_cols = int(cols.max()) + 1
        width = sfc_width_half_angles(
            rows, cols, n_rows, n_cols, inv_freq.to(position_ids.device),
            pad_cols=self.pad_cols, max_error=self.max_error,
        )
        out = base.clone()
        out[:, n_freq:] = width.to(base.dtype)
        self.calls += 1
        return out


def width_freq_slots(mrope_section) -> tuple[int, int]:
    section = [int(s) for s in mrope_section]
    return sum(section[:2]), sum(section)


class SpheRoPETextRotary(nn.Module):
    def __init__(self, original: nn.Module, mrope_section, max_error: float = 0.06) -> None:
        super().__init__()
        self.original = original
        self.slots = width_freq_slots(mrope_section)
        self.max_error = float(max_error)
        self.image_positions = None
        self.pad_cols = 0
        self.calls = 0

    def arm(self, image_positions: torch.Tensor, pad_cols: int = 0) -> None:
        self.image_positions = image_positions.to(torch.long)
        self.pad_cols = int(pad_cols)

    def disarm(self) -> None:
        self.image_positions = None
        self.pad_cols = 0

    def forward(self, x, position_ids):
        cos, sin = self.original(x, position_ids)
        if self.image_positions is None or position_ids.ndim != 3:
            return cos, sin
        img = self.image_positions.to(position_ids.device)
        if position_ids.shape[-1] <= int(img.max()):
            return cos, sin
        t = position_ids[0, 0, img]
        rows = position_ids[1, 0, img] - t
        cols = position_ids[2, 0, img] - t
        lo, hi = self.slots
        inv_freq = self.original.inv_freq[lo:hi].to(position_ids.device)
        angles = sfc_width_half_angles(
            rows, cols, int(rows.max()) + 1, int(cols.max()) + 1, inv_freq,
            pad_cols=self.pad_cols, max_error=self.max_error,
        )
        scaling = float(getattr(self.original, "attention_scaling", 1.0))
        half = cos.shape[-1] // 2
        new_cos = (torch.cos(angles) * scaling).to(cos.dtype)
        new_sin = (torch.sin(angles) * scaling).to(sin.dtype)
        cos = cos.clone()
        sin = sin.clone()
        for offset in (0, half):
            cos[2, 0, img, offset + lo:offset + hi] = new_cos
            sin[2, 0, img, offset + lo:offset + hi] = new_sin
        self.calls += 1
        return cos, sin


def circular_pad_image(image: Image.Image, pad_px: int) -> Image.Image:
    if pad_px <= 0:
        return image
    width, height = image.size
    pad_px = min(int(pad_px), width)
    out = Image.new(image.mode, (width + 2 * pad_px, height))
    out.paste(image.crop((width - pad_px, 0, width, height)), (0, 0))
    out.paste(image, (pad_px, 0))
    out.paste(image.crop((0, 0, pad_px, height)), (pad_px + width, 0))
    return out


def pad_columns(pad_px: int, padded_width_px: int, grid_cols: int, merge: int = 1) -> int:
    if pad_px <= 0:
        return 0
    cols = pad_px * grid_cols / padded_width_px
    unit = merge if merge > 1 else 1
    return int(round(cols / unit)) * unit
