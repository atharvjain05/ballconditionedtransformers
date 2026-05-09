"""Build sliding ``(history, future)`` windows from ``Sequence`` objects."""

from __future__ import annotations

from typing import Iterable, Sequence as PySequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .types import Sequence, Window


class WindowDataset(Dataset):
    """Sliding-window dataset.

    Each item is a ``Window`` with ``H`` history frames and ``F`` future frames.
    Windows where fewer than ``min_players`` players have *fully observed*
    history are dropped, since the prediction targets would be too sparse.

    Args:
        sequences: A list of ``Sequence`` objects.
        history: Number of input frames ``H``.
        horizon: Number of frames to predict ``F``.
        stride: Spacing between successive window starts within a sequence.
        min_players: Minimum number of players whose history is fully
            observed; windows below this are skipped. ``2`` is a sane default.
        require_ball_history: If ``True``, drop windows where the ball is
            unobserved during history (relevant if the dataset is ball-aware).
    """

    def __init__(
        self,
        sequences: PySequence[Sequence],
        history: int = 20,
        horizon: int = 10,
        stride: int = 5,
        min_players: int = 2,
        require_ball_history: bool = False,
    ) -> None:
        self.sequences = list(sequences)
        self.history = history
        self.horizon = horizon
        self.stride = stride
        self.min_players = min_players
        self.require_ball_history = require_ball_history
        self._index: list[tuple[int, int, np.ndarray]] = []
        self._build_index()

    def _build_index(self) -> None:
        H, F, S = self.history, self.horizon, self.stride
        for seq_idx, seq in enumerate(self.sequences):
            T = seq.num_frames
            if T < H + F:
                continue
            for start in range(0, T - (H + F) + 1, S):
                hist_mask = seq.player_mask[start : start + H]
                fut_mask = seq.player_mask[start + H : start + H + F]
                fully_observed = hist_mask.all(axis=0) & fut_mask.all(axis=0)
                if fully_observed.sum() < self.min_players:
                    continue
                if self.require_ball_history and seq.ball_mask is not None:
                    bm = seq.ball_mask[start : start + H]
                    if not bm.any(axis=-1).all():
                        continue
                self._index.append((seq_idx, start, fully_observed))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int) -> Window:
        seq_idx, start, active = self._index[i]
        seq = self.sequences[seq_idx]
        H, F = self.history, self.horizon
        hp = seq.players[start : start + H, active].astype(np.float32, copy=False)
        hpm = seq.player_mask[start : start + H, active]
        fp = seq.players[start + H : start + H + F, active].astype(np.float32, copy=False)
        fpm = seq.player_mask[start + H : start + H + F, active]
        if seq.ball is not None and seq.ball_mask is not None:
            hb = seq.ball[start : start + H].astype(np.float32, copy=False)
            hbm = seq.ball_mask[start : start + H]
            fb = seq.ball[start + H : start + H + F].astype(np.float32, copy=False)
            fbm = seq.ball_mask[start + H : start + H + F]
        else:
            hb = hbm = fb = fbm = None
        if seq.player_team is not None:
            hpt = seq.player_team[active].astype(np.int64, copy=False)
        else:
            hpt = np.full(int(active.sum()), 2, dtype=np.int64)
        return Window(
            sequence_id=seq.sequence_id,
            sport=seq.sport,
            history_players=hp,
            history_player_mask=hpm,
            future_players=fp,
            future_player_mask=fpm,
            history_ball=hb,
            history_ball_mask=hbm,
            future_ball=fb,
            future_ball_mask=fbm,
            history_player_teams=hpt,
            scale=seq.scale,
        )


def _pad_players(arrays: list[np.ndarray], max_n: int, dtype) -> tuple[np.ndarray, np.ndarray]:
    """Pad along the player axis to a common ``N`` and produce a padding mask."""
    T = arrays[0].shape[0]
    B = len(arrays)
    out = np.zeros((B, T, max_n, arrays[0].shape[-1]), dtype=dtype)
    pad_mask = np.zeros((B, max_n), dtype=bool)
    for i, a in enumerate(arrays):
        n = a.shape[1]
        out[i, :, :n] = a
        pad_mask[i, :n] = True
    return out, pad_mask


def _pad_teams(arrays: list[np.ndarray], max_n: int) -> np.ndarray:
    B = len(arrays)
    out = np.full((B, max_n), 2, dtype=np.int64)
    for i, a in enumerate(arrays):
        n = a.shape[0]
        out[i, :n] = a
    return out


def _pad_balls(
    hist_arr: list[np.ndarray],
    hist_mask: list[np.ndarray],
    fut_arr: list[np.ndarray],
    fut_mask: list[np.ndarray],
    H: int,
    F: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    max_k = max(a.shape[1] for a in hist_arr)
    B = len(hist_arr)
    hb = np.zeros((B, H, max_k, 2), dtype=np.float32)
    hbm = np.zeros((B, H, max_k), dtype=bool)
    fb = np.zeros((B, F, max_k, 2), dtype=np.float32)
    fbm = np.zeros((B, F, max_k), dtype=bool)
    ball_pad = np.zeros((B, max_k), dtype=bool)
    for i in range(B):
        k = hist_arr[i].shape[1]
        hb[i, :, :k, :] = hist_arr[i]
        hbm[i, :, :k] = hist_mask[i]
        fb[i, :, :k, :] = fut_arr[i]
        fbm[i, :, :k] = fut_mask[i]
        ball_pad[i, :k] = True
    return hb, hbm, fb, fbm, ball_pad


def collate_windows(batch: Iterable[Window]) -> dict:
    """Collate variable-player ``Window`` items into padded tensors.

    Returns a dict with float tensors on cpu:
      - ``hist_players``: ``(B, H, Nmax, 2)``
      - ``hist_player_mask``: ``(B, H, Nmax)`` bool, observed flags
      - ``player_pad_mask``: ``(B, Nmax)`` bool, ``True`` for real players
      - ``player_teams``: ``(B, Nmax)`` int64 team id (0/1/2); padded slots ``2``
      - analogous ``fut_*`` keys
      - ``hist_ball``: ``(B, H, Kmax, 2)``; ``hist_ball_mask``: ``(B, H, Kmax)``
      - ``ball_pad_mask``: ``(B, Kmax)`` bool, ``True`` for real ball tracks
      - ``scale``: ``(B, 2)``
      - ``sport``: list[str]; ``sequence_id``: list[str]
    """
    batch = list(batch)
    H = batch[0].history_players.shape[0]
    F = batch[0].future_players.shape[0]
    max_n = max(w.history_players.shape[1] for w in batch)

    hist_players_list = [w.history_players for w in batch]
    fut_players_list = [w.future_players for w in batch]
    hist_pmask_list = [w.history_player_mask for w in batch]
    fut_pmask_list = [w.future_player_mask for w in batch]

    hp, ppad = _pad_players(hist_players_list, max_n, np.float32)
    fp, _ = _pad_players(fut_players_list, max_n, np.float32)

    def _pad_mask(arrays: list[np.ndarray], T: int) -> np.ndarray:
        out = np.zeros((len(arrays), T, max_n), dtype=bool)
        for i, a in enumerate(arrays):
            out[i, :, : a.shape[1]] = a
        return out

    hpm = _pad_mask(hist_pmask_list, H)
    fpm = _pad_mask(fut_pmask_list, F)
    player_teams = _pad_teams([w.history_player_teams for w in batch], max_n)

    has_ball = all(w.history_ball is not None for w in batch)
    if has_ball:
        hb, hbm, fb, fbm, ball_pad = _pad_balls(
            [w.history_ball for w in batch],
            [w.history_ball_mask for w in batch],
            [w.future_ball for w in batch],
            [w.future_ball_mask for w in batch],
            H,
            F,
        )
    else:
        hb = fb = hbm = fbm = ball_pad = None

    scale = np.stack([w.scale for w in batch], axis=0).astype(np.float32)

    out = {
        "hist_players": torch.from_numpy(hp),
        "hist_player_mask": torch.from_numpy(hpm),
        "fut_players": torch.from_numpy(fp),
        "fut_player_mask": torch.from_numpy(fpm),
        "player_pad_mask": torch.from_numpy(ppad),
        "player_teams": torch.from_numpy(player_teams),
        "scale": torch.from_numpy(scale),
        "sport": [w.sport for w in batch],
        "sequence_id": [w.sequence_id for w in batch],
    }
    if has_ball:
        out["hist_ball"] = torch.from_numpy(hb)
        out["fut_ball"] = torch.from_numpy(fb)
        out["hist_ball_mask"] = torch.from_numpy(hbm)
        out["fut_ball_mask"] = torch.from_numpy(fbm)
        out["ball_pad_mask"] = torch.from_numpy(ball_pad)
    else:
        out["hist_ball"] = None
        out["fut_ball"] = None
        out["hist_ball_mask"] = None
        out["fut_ball_mask"] = None
        out["ball_pad_mask"] = None
    return out
