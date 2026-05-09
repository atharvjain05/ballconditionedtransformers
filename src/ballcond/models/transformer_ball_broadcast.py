"""Ball as a single broadcast context vector added to every player token.

Uses the same factored time / agent transformer blocks as the symmetric model,
but **no** extra ball agent in the agent axis. The ball history is encoded with
the trajectory tokenizer + temporal PE, mean-pooled over time, projected, then
**added** to all player tokens (same vector at every player and timestep) before
the stacked encoder blocks.
"""

from __future__ import annotations

import torch
from torch import nn

from .blocks import SinusoidalPositionalEncoding, TrajectoryTokenizer


class _AxisTransformerBlock(nn.Module):
    """One encoder block applied along a chosen axis (time or agent)."""

    def __init__(self, d_model: int, n_heads: int, ff_mult: int, dropout: float) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.encoder(x, src_key_padding_mask=key_padding_mask)


class BallBroadcastTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 6,
        ff_mult: int = 4,
        dropout: float = 0.1,
        horizon: int = 10,
        use_ball: bool = True,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.use_ball = use_ball
        self.tokenizer = TrajectoryTokenizer(d_model)
        self.team_emb = nn.Embedding(3, d_model)
        self.player_type = nn.Parameter(torch.zeros(d_model))
        self.time_pe = SinusoidalPositionalEncoding(d_model)
        self.ball_context = nn.Linear(d_model, d_model)
        self.time_blocks = nn.ModuleList(
            [_AxisTransformerBlock(d_model, n_heads, ff_mult, dropout) for _ in range(n_layers)]
        )
        self.agent_blocks = nn.ModuleList(
            [_AxisTransformerBlock(d_model, n_heads, ff_mult, dropout) for _ in range(n_layers)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, horizon * 2),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        hp = batch["hist_players"]  # (B, H, N, 2)
        ppad = batch["player_pad_mask"]  # (B, N) bool
        B, H, N, _ = hp.shape

        x = self.tokenizer(hp.permute(0, 2, 1, 3))  # (B, N, H, D)
        x = x + self.player_type.view(1, 1, 1, -1)
        team = batch.get("player_teams")
        if team is None:
            team = torch.full((B, N), 2, device=hp.device, dtype=torch.long)
        x = x + self.team_emb(team).unsqueeze(2)

        if self.use_ball and batch.get("hist_ball") is not None:
            hb = batch["hist_ball"]
            if hb.dim() == 3:
                hb = hb.unsqueeze(2)
            _, _, K, _ = hb.shape
            bpm = batch.get("ball_pad_mask")
            hm = batch.get("hist_ball_mask")
            D = x.shape[-1]
            ball_vec = torch.zeros(B, D, device=hp.device, dtype=x.dtype)
            for k in range(K):
                if bpm is not None and not bpm[:, k].any():
                    continue
                b = self.tokenizer(hb[:, :, k, :])
                b = self.time_pe(b)
                if hm is not None:
                    m = hm[:, :, k].float().unsqueeze(-1)
                    denom = m.sum(dim=1).clamp_min(1.0)
                    pooled = (b * m).sum(dim=1) / denom
                else:
                    pooled = b.mean(dim=1)
                ball_vec = ball_vec + self.ball_context(pooled)
        else:
            ball_vec = torch.zeros(B, x.shape[-1], device=x.device, dtype=x.dtype)

        x = x + ball_vec.unsqueeze(1).unsqueeze(2)

        Bm, Nm, Hm, D = x.shape
        xa = x.reshape(Bm * Nm, Hm, D)
        xa = self.time_pe(xa)
        for blk in self.time_blocks:
            xa = blk(xa)
        x = xa.reshape(Bm, Nm, Hm, D)

        kp_mask = ~ppad
        x = x.permute(0, 2, 1, 3).reshape(Bm * Hm, Nm, D)
        kp = kp_mask.unsqueeze(1).expand(Bm, Hm, Nm).reshape(Bm * Hm, Nm)
        for blk in self.agent_blocks:
            x = blk(x, key_padding_mask=kp)
        x = x.reshape(Bm, Hm, Nm, D).permute(0, 2, 1, 3)

        last_player = x[:, :, -1, :]
        deltas = self.head(last_player).view(B, N, self.horizon, 2)
        last_pos = hp[:, -1]
        cumulative = torch.cumsum(deltas.permute(0, 2, 1, 3), dim=1)
        last_pos_b = last_pos.unsqueeze(1)
        out = last_pos_b + cumulative
        out = out * ppad[:, None, :, None].to(out.dtype)
        return out
