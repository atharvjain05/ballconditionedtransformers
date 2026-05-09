"""Smoke tests: build a tiny synthetic dataset and run every model for one step.

These aren't quality tests — they exist to catch shape/device bugs early.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from ballcond.data import WindowDataset, collate_windows
from ballcond.data.synthetic import synthesize_dataset
from ballcond.metrics import RunningMetrics
from ballcond.models import (
    BallBroadcastTransformer,
    BallConditionedTransformer,
    ConstantVelocityKalman,
    PerPlayerLSTM,
    SymmetricTransformer,
    entity_transformer_ball_joint,
    entity_transformer_ball_symmetric,
    entity_transformer_players_only,
)


def _loader():
    seqs = synthesize_dataset(n=4, T=80, seed=0)
    ds = WindowDataset(seqs, history=20, horizon=10, stride=10, min_players=2)
    return DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_windows)


def test_kalman_runs():
    loader = _loader()
    model = ConstantVelocityKalman(horizon=10)
    batch = next(iter(loader))
    out = model(batch)
    assert out.shape[1] == 10 and out.shape[-1] == 2


def test_lstm_runs():
    loader = _loader()
    model = PerPlayerLSTM(hidden_dim=32, num_layers=1, horizon=10)
    batch = next(iter(loader))
    out = model(batch)
    target = batch["fut_players"]
    assert out.shape == target.shape


def test_symmetric_runs():
    loader = _loader()
    model = SymmetricTransformer(d_model=32, n_heads=2, n_layers=1, horizon=10)
    batch = next(iter(loader))
    out = model(batch)
    assert out.shape == batch["fut_players"].shape


def test_ballcond_runs():
    loader = _loader()
    model = BallConditionedTransformer(
        d_model=32, n_heads=2, n_layers=1, n_ball_layers=1, horizon=10
    )
    batch = next(iter(loader))
    out = model(batch)
    assert out.shape == batch["fut_players"].shape


def test_ball_broadcast_runs():
    loader = _loader()
    model = BallBroadcastTransformer(d_model=32, n_heads=2, n_layers=1, horizon=10, use_ball=True)
    batch = next(iter(loader))
    out = model(batch)
    assert out.shape == batch["fut_players"].shape


def test_entity_transformer_runs():
    loader = _loader()
    model = entity_transformer_players_only(
        d_model=32, n_heads=2, n_layers=1, horizon=10, entity_history=20
    )
    batch = next(iter(loader))
    out = model(batch)
    assert out.shape == batch["fut_players"].shape


def test_entity_ball_symmetric_runs():
    loader = _loader()
    model = entity_transformer_ball_symmetric(
        d_model=32, n_heads=2, n_layers=1, horizon=10, entity_history=20
    )
    batch = next(iter(loader))
    out = model(batch)
    assert out.shape == batch["fut_players"].shape


def test_entity_ball_joint_runs():
    loader = _loader()
    model = entity_transformer_ball_joint(
        d_model=32, n_heads=2, n_layers=1, horizon=10, entity_history=20
    )
    batch = next(iter(loader))
    out = model(batch)
    assert out.shape == batch["fut_players"].shape


def test_metrics():
    rm = RunningMetrics(horizons=[5, 10])
    pred = torch.zeros(2, 10, 3, 2)
    target = torch.ones(2, 10, 3, 2)
    mask = torch.ones(2, 10, 3, dtype=torch.bool)
    rm.update(pred, target, mask)
    out = rm.compute()
    assert "ade@5" in out and "fde@10" in out
    assert abs(out["ade@5"] - (2**0.5)) < 1e-6
