"""Entity-level transformers: one token per player (and optionally ball).

Each agent's last ``entity_history`` frames are flattened to ``2 * entity_history``
scalars and projected with a single linear layer to ``d_model``. Stacked transformer
layers then mix information across agents. The prediction head outputs velocity-style
deltas from the last observed position, matching other models in this package.

Modes (``ball_interaction``):
- ``none``: players only.
- ``symmetric``: each ball track is its own token; full self-attention over players
  + all ball tracks.
- ``joint``: mean of per-track ball embeddings (masked) is **broadcast-added** to
  every player token; self-attention runs **only over players** (no ball token in
  the graph).
"""

from __future__ import annotations

import torch
from torch import nn


def _flatten_history_tail(xy: torch.Tensor, n: int) -> torch.Tensor:
    """Take the last ``n`` time steps along dim 1 and flatten coords.

    Args:
        xy: ``(B, H, ..., 2)`` positions (players or one ball track).
        n: Desired history length; if ``H < n``, left-pad with zeros in time.

    Returns:
        ``(B, ..., n * 2)`` flattened features.
    """
    B, H = xy.shape[0], xy.shape[1]
    middle = xy.shape[2:-1]
    if H >= n:
        chunk = xy[:, -n:]
    else:
        pad_shape = (B, n - H) + middle + (2,)
        pad = xy.new_zeros(pad_shape)
        chunk = torch.cat([pad, xy], dim=1)
    return chunk.reshape(B, *middle, n * 2)


class EntitySetTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 6,
        ff_mult: int = 4,
        dropout: float = 0.1,
        horizon: int = 10,
        entity_history: int = 20,
        use_ball: bool = False,
        ball_interaction: str = "none",
    ) -> None:
        super().__init__()
        if ball_interaction not in ("none", "symmetric", "joint"):
            raise ValueError(
                f"ball_interaction must be none|symmetric|joint, got {ball_interaction}"
            )
        if not use_ball:
            ball_interaction = "none"
        self.horizon = horizon
        self.entity_history = entity_history
        self.d_model = d_model
        self.use_ball = use_ball
        self.ball_interaction = ball_interaction

        feat_dim = 2 * entity_history
        self.entity_proj = nn.Linear(feat_dim, d_model)
        self.team_emb = nn.Embedding(3, d_model)
        self.player_type_emb = nn.Parameter(torch.zeros(d_model))
        self.ball_type_emb = nn.Parameter(torch.zeros(d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.self_encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, horizon * 2),
        )

    def _player_raw(self, hp: torch.Tensor) -> torch.Tensor:
        return _flatten_history_tail(hp, self.entity_history)

    def _encode_ball_tracks(self, hb: torch.Tensor) -> torch.Tensor:
        """Ball tracks -> ``(B, K, d)`` (one intermediate vector per track)."""
        if hb.dim() == 3:
            hb = hb.unsqueeze(2)
        _, _, K, _ = hb.shape
        tracks = []
        for k in range(K):
            raw = _flatten_history_tail(hb[:, :, k, :], self.entity_history)
            tracks.append(self.entity_proj(raw) + self.ball_type_emb)
        return torch.stack(tracks, dim=1)

    def _mean_ball_embedding(self, hb: torch.Tensor, bpad: torch.Tensor | None) -> torch.Tensor:
        """Mean of per-track ball embeddings over real tracks: ``(B, d)``."""
        tracks = self._encode_ball_tracks(hb)
        B, K, _ = tracks.shape
        if bpad is None:
            bpad = torch.ones(B, K, dtype=torch.bool, device=tracks.device)
        w = bpad.to(tracks.dtype).unsqueeze(-1)
        denom = w.squeeze(-1).sum(dim=1, keepdim=True).clamp_min(1.0)
        return (tracks * w).sum(dim=1) / denom

    def forward(self, batch: dict) -> torch.Tensor:
        hp = batch["hist_players"]
        ppad = batch["player_pad_mask"]
        B, _, N, _ = hp.shape
        F = self.horizon

        player_raw = self._player_raw(hp)
        x = self.entity_proj(player_raw) + self.player_type_emb
        team = batch.get("player_teams")
        if team is None:
            team = torch.full((B, N), 2, device=hp.device, dtype=torch.long)
        x = x + self.team_emb(team)

        hb = batch.get("hist_ball") if self.use_ball else None
        agent_kp = ~ppad

        if self.ball_interaction == "symmetric" and hb is not None:
            ball_toks = self._encode_ball_tracks(hb)
            Kb = ball_toks.shape[1]
            bpad = batch.get("ball_pad_mask")
            if bpad is None:
                bpad = torch.ones(B, Kb, dtype=torch.bool, device=ppad.device)
            x = torch.cat([x, ball_toks], dim=1)
            agent_kp = torch.cat([~ppad, ~bpad], dim=1)
            x = self.self_encoder(x, src_key_padding_mask=agent_kp)
            x = x[:, :N, :]
        elif self.ball_interaction == "joint" and hb is not None:
            bpad = batch.get("ball_pad_mask")
            ball_mean = self._mean_ball_embedding(hb, bpad)
            x = x + ball_mean.unsqueeze(1)
            x = self.self_encoder(x, src_key_padding_mask=agent_kp)
        else:
            x = self.self_encoder(x, src_key_padding_mask=agent_kp)

        deltas = self.head(x).view(B, N, F, 2)
        cumulative = torch.cumsum(deltas.permute(0, 2, 1, 3), dim=1)
        last_pos = hp[:, -1].unsqueeze(1)
        out = last_pos + cumulative
        out = out * ppad[:, None, :, None].to(out.dtype)
        return out


def entity_transformer_players_only(**kwargs) -> EntitySetTransformer:
    return EntitySetTransformer(ball_interaction="none", use_ball=False, **kwargs)


def entity_transformer_ball_symmetric(**kwargs) -> EntitySetTransformer:
    return EntitySetTransformer(ball_interaction="symmetric", use_ball=True, **kwargs)


def entity_transformer_ball_joint(**kwargs) -> EntitySetTransformer:
    return EntitySetTransformer(ball_interaction="joint", use_ball=True, **kwargs)
