"""Trajectory-prediction metrics.

We report:

- **ADE** (average displacement error): mean L2 distance between predicted and
  ground-truth positions across the prediction horizon.
- **FDE** (final displacement error): L2 distance at the last predicted frame.
- Both are reported at multiple horizons (5, 10, 20 frames).

All inputs are ``(B, F, N, 2)`` tensors plus a ``(B, F, N)`` validity mask.
Errors are averaged over valid (player, frame) pairs.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def displacement(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L2 distance per (batch, frame, player). Shape: ``(B, F, N)``."""
    return torch.linalg.vector_norm(pred - target, dim=-1)


def ade_fde(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    horizons: list[int] | None = None,
) -> dict[str, float]:
    """Compute ADE and FDE at the requested horizons.

    Args:
        pred: ``(B, F, N, 2)`` predictions.
        target: ``(B, F, N, 2)`` ground-truth positions.
        mask: ``(B, F, N)`` bool, ``True`` where the target is observed.
        horizons: List of frame counts ``f`` for which to report ADE@f and
            FDE@f. Defaults to all available frames.

    Returns:
        Dict like ``{"ade@5": ..., "fde@5": ..., "ade@10": ...}``.
    """
    F = pred.shape[1]
    horizons = horizons or [F]
    d = displacement(pred, target)
    out: dict[str, float] = {}
    for f in horizons:
        if f > F:
            continue
        sub_d = d[:, :f]
        sub_m = mask[:, :f].to(sub_d.dtype)
        denom = sub_m.sum().clamp_min(1.0)
        out[f"ade@{f}"] = float((sub_d * sub_m).sum() / denom)
        last_d = d[:, f - 1]
        last_m = mask[:, f - 1].to(last_d.dtype)
        denom_last = last_m.sum().clamp_min(1.0)
        out[f"fde@{f}"] = float((last_d * last_m).sum() / denom_last)
    return out


@dataclass
class RunningMetrics:
    """Accumulate ADE/FDE numerator/denominator across batches."""

    horizons: list[int]
    _ade_num: dict[int, float] = None
    _ade_den: dict[int, float] = None
    _fde_num: dict[int, float] = None
    _fde_den: dict[int, float] = None

    def __post_init__(self) -> None:
        self._ade_num = {h: 0.0 for h in self.horizons}
        self._ade_den = {h: 0.0 for h in self.horizons}
        self._fde_num = {h: 0.0 for h in self.horizons}
        self._fde_den = {h: 0.0 for h in self.horizons}

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> None:
        d = displacement(pred, target)
        F = d.shape[1]
        for h in self.horizons:
            if h > F:
                continue
            sub_d = d[:, :h]
            sub_m = mask[:, :h].to(sub_d.dtype)
            self._ade_num[h] += float((sub_d * sub_m).sum())
            self._ade_den[h] += float(sub_m.sum())
            last_d = d[:, h - 1]
            last_m = mask[:, h - 1].to(last_d.dtype)
            self._fde_num[h] += float((last_d * last_m).sum())
            self._fde_den[h] += float(last_m.sum())

    def compute(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for h in self.horizons:
            if self._ade_den[h] > 0:
                out[f"ade@{h}"] = self._ade_num[h] / self._ade_den[h]
            if self._fde_den[h] > 0:
                out[f"fde@{h}"] = self._fde_num[h] / self._fde_den[h]
        return out
