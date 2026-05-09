"""Shared transformer building blocks used by the trajectory models."""

from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    """Standard 1-D sinusoidal positional encoding.

    Used along the time axis. We keep it un-learned so we can extrapolate
    cleanly to longer horizons than seen at training.
    """

    def __init__(self, dim: int, max_len: int = 4096) -> None:
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(-2)]


class TrajectoryTokenizer(nn.Module):
    """Project ``(x, y)`` (and optionally a velocity diff) into ``d_model``.

    We feed the model both the absolute position and the per-step delta; this
    is a small but consistent improvement vs. position-only and matches what
    most modern trajectory transformers do.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(4, d_model)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        # xy: (..., T, 2)
        delta = torch.zeros_like(xy)
        delta[..., 1:, :] = xy[..., 1:, :] - xy[..., :-1, :]
        feat = torch.cat([xy, delta], dim=-1)
        return self.proj(feat)
