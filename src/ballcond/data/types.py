"""Generic trajectory containers shared by every loader and model.

A ``Sequence`` holds one clip's worth of (already-normalized) positions for
players and (optionally) the ball. We keep player tracks dense on a fixed time
grid by carrying a per-(player, frame) validity mask, since real datasets have
players entering and exiting the frame.

Coordinate convention
---------------------
All positions are float32 in a *normalized* frame:
- For image-plane datasets (SportsMOT, SoccerNet-Tracking) we map pixel
  centers to ``[0, 1] x [0, 1]`` using the video's width and height.
- For court/pitch-coordinate datasets (NBA SportVU) we map to ``[0, 1]^2``
  using the court dimensions.
This lets all models be trained on the same scale and lets ADE/FDE be
reported in either normalized or original units.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Sequence:
    """One contiguous clip.

    Attributes:
        sequence_id: Unique id (e.g. SportsMOT video name).
        sport: ``"basketball" | "soccer" | "volleyball" | ...``.
        fps: Source frame rate. Useful for selecting prediction horizons in
            seconds.
        players: ``(T, N, 2)`` float32 positions for ``N`` distinct players over
            ``T`` frames. Coordinates are normalized to ``[0, 1]``.
        player_mask: ``(T, N)`` bool. ``True`` where the player is observed at
            that frame. ``players[~player_mask]`` is meaningless.
        ball: ``(T, K, 2)`` float32 — one or more ball tracks (`K >= 1`), or
            ``None`` if the dataset has no ball.
        ball_mask: ``(T, K)`` bool, ``True`` when that ball track is observed at
            that frame.
        scale: ``(2,)`` array (width, height) of the original frame, in pixels
            for image-plane datasets and feet/meters for court datasets. Used
            to convert ADE/FDE back to original units.
        meta: Free-form metadata bag (e.g. raw track ids, source path).
        player_team: Optional per-player team id ``(N,)`` int (SoccerNet: 0 =
            team left in frame, 1 = team right, 2 = unknown). ``None`` if unused.
    """

    sequence_id: str
    sport: str
    fps: float
    players: np.ndarray
    player_mask: np.ndarray
    ball: Optional[np.ndarray]
    ball_mask: Optional[np.ndarray]
    scale: np.ndarray
    player_team: Optional[np.ndarray] = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        T, N, two = self.players.shape
        assert two == 2, "players must be (T, N, 2)"
        assert self.player_mask.shape == (T, N), "player_mask must be (T, N)"
        assert self.players.dtype == np.float32
        assert self.player_mask.dtype == np.bool_
        if self.player_team is not None:
            assert self.player_team.shape == (N,), "player_team must be (N,)"
            assert self.player_team.dtype in (
                np.int8,
                np.int16,
                np.int32,
                np.int64,
            ), "player_team dtype must be integer"
        if self.ball is not None:
            assert self.ball.ndim == 3 and self.ball.shape[0] == T and self.ball.shape[2] == 2
            Kb = self.ball.shape[1]
            assert self.ball_mask is not None and self.ball_mask.shape == (T, Kb)
            assert self.ball.dtype == np.float32
            assert self.ball_mask.dtype == np.bool_
        assert self.scale.shape == (2,)

    @property
    def num_frames(self) -> int:
        return int(self.players.shape[0])

    @property
    def num_players(self) -> int:
        return int(self.players.shape[1])

    @property
    def has_ball(self) -> bool:
        return self.ball is not None


@dataclass
class Window:
    """A ``(history, future)`` slice extracted from a ``Sequence``.

    Tensors are kept as numpy arrays here; the ``WindowDataset`` collator
    converts to torch tensors. ``N`` is the number of *active* players in the
    window (we drop players whose history is mostly missing).
    """

    sequence_id: str
    sport: str
    history_players: np.ndarray  # (H, N, 2)
    history_player_mask: np.ndarray  # (H, N) bool
    future_players: np.ndarray  # (F, N, 2)
    future_player_mask: np.ndarray  # (F, N) bool
    history_ball: Optional[np.ndarray]  # (H, K, 2) or None
    history_ball_mask: Optional[np.ndarray]  # (H, K) or None
    future_ball: Optional[np.ndarray]  # (F, K, 2) or None
    future_ball_mask: Optional[np.ndarray]  # (F, K) or None
    history_player_teams: np.ndarray  # (N,) int64, 0/1/2; unknown = 2
    scale: np.ndarray  # (2,)
