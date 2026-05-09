"""Symmetric transformer baseline.

All agents (players AND the ball) are encoded as identical tokens. This is the
"ball is just another agent" baseline the paper compares against.

Architecture
------------
For each (agent, time) we build a token by concatenating:
- A trajectory embedding from ``TrajectoryTokenizer``.
- A learned agent-type embedding (player / ball). The symmetric model still
  *receives* the ball/player flag (otherwise the comparison would conflate the
  effect of identifying the ball with the effect of the architecture), but
  uses it only as an embedding into the same token stream.

Two transformer encoder layers operate alternately along the time axis (per
agent) and the agent axis (per timestep). This factored attention is cheaper
than full ``(time x agent)``-product attention and is standard in modern
trajectory transformers.

The decoder regresses ``F`` future positions per player from the last-time
token, residually relative to the last observed position.
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


class SymmetricTransformer(nn.Module):
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
        self.type_emb = nn.Embedding(2, d_model)  # 0 = player, 1 = ball
        self.team_emb = nn.Embedding(3, d_model)  # 0/1 = sides, 2 = unknown
        self.time_pe = SinusoidalPositionalEncoding(d_model)
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

        player_tok = self.tokenizer(hp.permute(0, 2, 1, 3))  # (B, N, H, D)
        player_tok = player_tok + self.type_emb.weight[0].view(1, 1, 1, -1)
        team = batch.get("player_teams")
        if team is None:
            team = torch.full((B, N), 2, device=hp.device, dtype=torch.long)
        player_tok = player_tok + self.team_emb(team).unsqueeze(2)
        tokens = player_tok
        agent_pad = ppad

        if self.use_ball and batch.get("hist_ball") is not None:
            hb = batch["hist_ball"]
            if hb.dim() == 3:
                hb = hb.unsqueeze(2)  # legacy (B, H, 2) -> (B, H, 1, 2)
            K = hb.shape[2]
            bpad = batch.get("ball_pad_mask")
            if bpad is None:
                bpad = torch.ones(B, K, dtype=torch.bool, device=ppad.device)
            ball_toks = [
                self.tokenizer(hb[:, :, k, :]) + self.type_emb.weight[1].view(1, 1, -1)
                for k in range(K)
            ]
            ball_cat = torch.stack(ball_toks, dim=1)  # (B, K, H, D)
            tokens = torch.cat([tokens, ball_cat], dim=1)
            agent_pad = torch.cat([ppad, bpad], dim=1)

        Bm, A, Hm, D = tokens.shape

        x = tokens.reshape(Bm * A, Hm, D)
        x = self.time_pe(x)
        for blk in self.time_blocks:
            x = blk(x)
        x = x.reshape(Bm, A, Hm, D)

        kp_mask = ~agent_pad  # padding positions are True for nn.Transformer
        x = x.permute(0, 2, 1, 3).reshape(Bm * Hm, A, D)
        kp = kp_mask.unsqueeze(1).expand(Bm, Hm, A).reshape(Bm * Hm, A)
        for blk in self.agent_blocks:
            x = blk(x, key_padding_mask=kp)
        x = x.reshape(Bm, Hm, A, D).permute(0, 2, 1, 3)  # (B, A, H, D)

        last_player = x[:, :N, -1, :]  # (B, N, D)
        deltas = self.head(last_player).view(B, N, self.horizon, 2)
        last_pos = hp[:, -1]  # (B, N, 2)
        cumulative = torch.cumsum(deltas.permute(0, 2, 1, 3), dim=1)  # (B, F, N, 2)
        last_pos_b = last_pos.unsqueeze(1)  # (B, 1, N, 2)
        out = last_pos_b + cumulative
        out = out * ppad[:, None, :, None].to(out.dtype)
        return out
