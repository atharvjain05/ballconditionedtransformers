"""Per-player LSTM baseline.

Each player is treated independently: an LSTM consumes the player's own
position history and emits per-step velocity offsets that are integrated to
produce the future trajectory. The model has no awareness of the ball or of
other players, so it serves as a "context-free" baseline.

We predict velocity deltas rather than absolute positions because positions
are bounded in ``[0, 1]`` while deltas are roughly zero-mean, which makes the
optimization much easier.
"""

from __future__ import annotations

import torch
from torch import nn


class PerPlayerLSTM(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 2,
        horizon: int = 10,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.encoder = nn.LSTM(
            input_size=2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.decoder = nn.LSTM(
            input_size=2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, batch: dict) -> torch.Tensor:
        hp = batch["hist_players"]  # (B, H, N, 2)
        ppad = batch["player_pad_mask"]  # (B, N)
        B, H, N, _ = hp.shape
        F = self.horizon

        x = hp.permute(0, 2, 1, 3).reshape(B * N, H, 2)
        _, (h, c) = self.encoder(x)

        last = x[:, -1, :]  # (B*N, 2)
        cur = last
        out = []
        h_dec, c_dec = h, c
        for _ in range(F):
            step = cur.unsqueeze(1)
            y, (h_dec, c_dec) = self.decoder(step, (h_dec, c_dec))
            delta = self.head(y.squeeze(1))
            cur = cur + delta
            out.append(cur)
        out = torch.stack(out, dim=1)  # (B*N, F, 2)
        out = out.reshape(B, N, F, 2).permute(0, 2, 1, 3)
        out = out * ppad[:, None, :, None].to(out.dtype)
        return out
