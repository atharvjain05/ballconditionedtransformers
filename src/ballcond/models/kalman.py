"""Constant-velocity Kalman filter baseline.

This is a non-learned reference: each player track is rolled forward under a
constant-velocity motion model fit from its history. The same filter doubles
as the physics-informed ball predictor mentioned in the paper.

Conventions
-----------
State per track: ``[x, y, vx, vy]``. We only run the *predict* step over the
horizon (no measurements during the future), which makes the filter degenerate
to deterministic constant-velocity extrapolation; the noise covariance is kept
around for completeness and to support future stochastic evaluation.
"""

from __future__ import annotations

import numpy as np
import torch


def _fit_velocity(history: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit a single ``(pos, vel)`` per player from observed history.

    Args:
        history: ``(H, N, 2)`` positions.
        mask: ``(H, N)`` bool, ``True`` where observed.

    Returns:
        ``(pos[N, 2], vel[N, 2])`` last-observed position and a least-squares
        velocity estimate (slope of position vs. time on observed frames).
    """
    H, N, _ = history.shape
    pos = np.zeros((N, 2), dtype=np.float32)
    vel = np.zeros((N, 2), dtype=np.float32)
    t = np.arange(H, dtype=np.float32)
    for n in range(N):
        m = mask[:, n]
        if not m.any():
            continue
        last_t = int(np.where(m)[0].max())
        pos[n] = history[last_t, n]
        if m.sum() < 2:
            continue
        ts = t[m]
        ps = history[m, n]
        ts_c = ts - ts.mean()
        denom = (ts_c**2).sum()
        if denom < 1e-6:
            continue
        vel[n, 0] = (ts_c * (ps[:, 0] - ps[:, 0].mean())).sum() / denom
        vel[n, 1] = (ts_c * (ps[:, 1] - ps[:, 1].mean())).sum() / denom
    return pos, vel


class ConstantVelocityKalman:
    """A batched, vectorized constant-velocity rollout.

    Implemented as a plain Python class (no nn.Module) since there are no
    learnable parameters. The interface is intentionally aligned with the
    learned models: ``forward(batch) -> (B, F, N, 2)``.
    """

    def __init__(self, horizon: int) -> None:
        self.horizon = horizon

    def __call__(self, batch: dict) -> torch.Tensor:
        return self.predict(batch)

    def predict(self, batch: dict) -> torch.Tensor:
        hp = batch["hist_players"].cpu().numpy()
        hpm = batch["hist_player_mask"].cpu().numpy()
        ppad = batch["player_pad_mask"].cpu().numpy()
        B, H, Nmax, _ = hp.shape
        F = self.horizon
        out = np.zeros((B, F, Nmax, 2), dtype=np.float32)
        for b in range(B):
            real = ppad[b]
            n_real = int(real.sum())
            pos, vel = _fit_velocity(hp[b, :, :n_real], hpm[b, :, :n_real])
            steps = np.arange(1, F + 1, dtype=np.float32)[:, None, None]
            out[b, :, :n_real] = pos[None] + steps * vel[None]
        return torch.from_numpy(out).to(batch["hist_players"].device)
