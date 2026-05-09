"""Synthetic ball-and-player toy data.

A scriptable, deterministic generator that produces ``Sequence`` objects with a
ball and 10 players. Two purposes:

1. End-to-end development before real ball-labeled data is wired in.
2. A controlled testbed where ball-conditioning *should* help. By construction
   the ball follows a smooth random path and each player is attracted toward
   the ball plus a per-player home position, which mimics the regime our paper
   argues for: information flows from the ball to the players.

The generator returns positions in ``[0, 1]^2``.
"""

from __future__ import annotations

import numpy as np

from .types import Sequence


def _smooth_walk(T: int, rng: np.random.Generator, sigma: float = 0.01) -> np.ndarray:
    """A 2D random-walk with low-pass filtering to keep paths smooth."""
    steps = rng.normal(0.0, sigma, size=(T, 2))
    pos = np.cumsum(steps, axis=0)
    pos -= pos.min(axis=0, keepdims=True)
    pos /= max(pos.max() + 1e-6, 1.0)
    pos = 0.1 + 0.8 * pos
    return pos.astype(np.float32)


def synthesize_sequence(
    sequence_id: str,
    T: int = 200,
    n_players: int = 10,
    ball_attraction: float = 0.04,
    home_attraction: float = 0.02,
    noise: float = 0.005,
    seed: int = 0,
) -> Sequence:
    """One synthetic sports clip with believable ball-driven player motion.

    Players get pulled toward the ball with strength ``ball_attraction`` and
    toward a fixed per-player home cell with strength ``home_attraction`` (so
    e.g. a "guard" stays near the perimeter). A small Gaussian jitter is added
    each step.
    """
    rng = np.random.default_rng(seed)

    ball = _smooth_walk(T, rng, sigma=0.015)

    grid_x, grid_y = np.meshgrid(np.linspace(0.15, 0.85, 5), np.linspace(0.2, 0.8, 2))
    homes = np.stack([grid_x.flatten(), grid_y.flatten()], axis=-1).astype(np.float32)
    homes = homes[:n_players]

    players = np.zeros((T, n_players, 2), dtype=np.float32)
    players[0] = homes + rng.normal(0.0, 0.02, size=homes.shape).astype(np.float32)
    velocity = np.zeros((n_players, 2), dtype=np.float32)
    for t in range(1, T):
        to_ball = ball[t] - players[t - 1]
        to_home = homes - players[t - 1]
        accel = ball_attraction * to_ball + home_attraction * to_home
        accel += rng.normal(0.0, noise, size=accel.shape).astype(np.float32)
        velocity = 0.85 * velocity + accel
        players[t] = players[t - 1] + velocity
        np.clip(players[t], 0.02, 0.98, out=players[t])

    ball = ball[:, np.newaxis, :].astype(np.float32, copy=False)  # (T, 1, 2)
    ball_mask = np.ones((T, 1), dtype=bool)

    return Sequence(
        sequence_id=sequence_id,
        sport="synthetic",
        fps=25.0,
        players=players,
        player_mask=np.ones((T, n_players), dtype=bool),
        ball=ball,
        ball_mask=ball_mask,
        scale=np.array([1.0, 1.0], dtype=np.float32),
        player_team=None,
        meta={"source": "synthetic", "seed": seed},
    )


def synthesize_dataset(n: int = 50, T: int = 200, seed: int = 0, **kwargs) -> list[Sequence]:
    """Generate ``n`` independent synthetic clips."""
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31 - 1, size=n)
    return [
        synthesize_sequence(f"synth_{i:04d}", T=T, seed=int(s), **kwargs)
        for i, s in enumerate(seeds)
    ]
