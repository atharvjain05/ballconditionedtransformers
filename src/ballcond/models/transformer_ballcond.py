"""Ball-conditioned transformer: the main contribution.

Information flow is *asymmetric*: the ball influences players, but the player
tokens never get attended to *by* the ball. Concretely, each encoder layer
runs:

1. **Self-attention over time per player.** Each player attends to its own
   history.
2. **Self-attention across players per timestep.** Players see each other.
3. **Cross-attention from players (queries) to the ball history (keys/values).**
   This is the privileged ball signal: every player gets to look at the entire
   ball trajectory at every layer.

The ball trajectory is encoded with its own (smaller) temporal encoder before
being used as keys/values in the cross-attention. Cf. Section 1 of the paper:
"information flows from the ball to the players, not symmetrically between
all agents".

When the dataset has no ball (``hist_ball is None``), we fall back to using a
zeroed ball memory; the architecture then degenerates gracefully to a
symmetric-without-ball baseline rather than crashing.
"""

from __future__ import annotations

import torch
from torch import nn

from .blocks import SinusoidalPositionalEncoding, TrajectoryTokenizer


class _BallEncoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_layers: int, dropout: float) -> None:
        super().__init__()
        self.tokenizer = TrajectoryTokenizer(d_model)
        self.pe = SinusoidalPositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, ball: torch.Tensor) -> torch.Tensor:
        x = self.tokenizer(ball)  # (B, H, D)
        x = self.pe(x)
        return self.encoder(x)


class _BallCondLayer(nn.Module):
    """One layer of [time SA] + [agent SA] + [cross-attn from players to ball]."""

    def __init__(self, d_model: int, n_heads: int, ff_mult: int, dropout: float) -> None:
        super().__init__()
        self.time_norm = nn.LayerNorm(d_model)
        self.time_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.agent_norm = nn.LayerNorm(d_model)
        self.agent_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_norm_q = nn.LayerNorm(d_model)
        self.cross_norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,  # (B, N, H, D)
        ball_mem: torch.Tensor,  # (B, H, D)
        agent_pad_mask: torch.Tensor,  # (B, N) bool, True = real
    ) -> torch.Tensor:
        B, N, H, D = x.shape
        residual = x
        x_t = self.time_norm(x).reshape(B * N, H, D)
        out_t, _ = self.time_attn(x_t, x_t, x_t, need_weights=False)
        x = residual + out_t.reshape(B, N, H, D)

        residual = x
        x_a = self.agent_norm(x).permute(0, 2, 1, 3).reshape(B * H, N, D)
        kp = (~agent_pad_mask).unsqueeze(1).expand(B, H, N).reshape(B * H, N)
        out_a, _ = self.agent_attn(x_a, x_a, x_a, key_padding_mask=kp, need_weights=False)
        x = residual + out_a.reshape(B, H, N, D).permute(0, 2, 1, 3)

        residual = x
        q = self.cross_norm_q(x).reshape(B * N, H, D)
        kv = self.cross_norm_kv(ball_mem).repeat_interleave(N, dim=0)  # (B*N, H, D)
        out_c, _ = self.cross_attn(q, kv, kv, need_weights=False)
        x = residual + out_c.reshape(B, N, H, D)

        residual = x
        x = residual + self.ff(self.ff_norm(x))
        return x


class BallConditionedTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 6,
        n_ball_layers: int = 6,
        ff_mult: int = 4,
        dropout: float = 0.1,
        horizon: int = 10,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.player_tokenizer = TrajectoryTokenizer(d_model)
        self.team_emb = nn.Embedding(3, d_model)
        self.player_pe = SinusoidalPositionalEncoding(d_model)
        self.ball_encoder = _BallEncoder(d_model, n_heads, n_ball_layers, dropout)
        self.layers = nn.ModuleList(
            [_BallCondLayer(d_model, n_heads, ff_mult, dropout) for _ in range(n_layers)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, horizon * 2),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        hp = batch["hist_players"]  # (B, H, N, 2)
        ppad = batch["player_pad_mask"]  # (B, N)
        B, H, N, _ = hp.shape
        F = self.horizon

        x = self.player_tokenizer(hp.permute(0, 2, 1, 3))  # (B, N, H, D)
        D = x.shape[-1]
        team = batch.get("player_teams")
        if team is None:
            team = torch.full((B, N), 2, device=hp.device, dtype=torch.long)
        x = x + self.team_emb(team).unsqueeze(2)
        x_flat = x.reshape(B * N, H, D)
        x_flat = self.player_pe(x_flat)
        x = x_flat.reshape(B, N, H, D)

        if batch.get("hist_ball") is not None:
            hb = batch["hist_ball"]
            if hb.dim() == 3:
                hb = hb.unsqueeze(2)
            _, _, K, _ = hb.shape
            bpm = batch.get("ball_pad_mask")
            ball_mem = torch.zeros(B, H, D, device=x.device, dtype=x.dtype)
            for k in range(K):
                if bpm is not None and not bpm[:, k].any():
                    continue
                ball_mem = ball_mem + self.ball_encoder(hb[:, :, k, :])
        else:
            ball_mem = torch.zeros(B, H, D, device=x.device, dtype=x.dtype)

        for layer in self.layers:
            x = layer(x, ball_mem, ppad)

        last = x[:, :, -1, :]  # (B, N, D)
        deltas = self.head(last).view(B, N, F, 2)
        cumulative = torch.cumsum(deltas.permute(0, 2, 1, 3), dim=1)  # (B, F, N, 2)
        last_pos = hp[:, -1].unsqueeze(1)  # (B, 1, N, 2)
        out = last_pos + cumulative
        out = out * ppad[:, None, :, None].to(out.dtype)
        return out
