"""Training loop for trajectory prediction.

Usage::

    python -m ballcond.train --config configs/synthetic_ballcond.yaml

The training loop is intentionally minimal (no PyTorch Lightning, no Hydra)
since the project is small. It optimizes a masked MSE on the future positions
and reports running ADE/FDE on a held-out split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import WindowDataset, collate_windows
from .data.synthetic import synthesize_dataset
from .data.soccernet import load_soccernet_split
from .data.sportsmot import load_sportsmot_split
from .metrics import RunningMetrics
from .models import (
    BallBroadcastTransformer,
    BallConditionedTransformer,
    ConstantVelocityKalman,
    PerPlayerLSTM,
    SymmetricTransformer,
    entity_transformer_ball_joint,
    entity_transformer_ball_symmetric,
    entity_transformer_players_only,
)
from .utils import ensure_dir, get_device, set_seed

MODEL_REGISTRY: dict[str, Callable[..., nn.Module]] = {
    "lstm": PerPlayerLSTM,
    "transformer_symmetric": SymmetricTransformer,
    "transformer_ball_broadcast": BallBroadcastTransformer,
    "transformer_ballcond": BallConditionedTransformer,
    "transformer_entity": entity_transformer_players_only,
    "transformer_entity_ball_symmetric": entity_transformer_ball_symmetric,
    "transformer_entity_ball_joint": entity_transformer_ball_joint,
}


def build_dataset(cfg) -> tuple[list, list]:
    """Build train and val ``Sequence`` lists from the config."""
    name = cfg.data.name
    if name == "synthetic":
        seqs = synthesize_dataset(
            n=cfg.data.num_sequences,
            T=cfg.data.frames_per_sequence,
            seed=cfg.seed,
            ball_attraction=cfg.data.get("ball_attraction", 0.04),
            home_attraction=cfg.data.get("home_attraction", 0.02),
            noise=cfg.data.get("noise", 0.005),
            n_players=cfg.data.get("n_players", 10),
        )
        n_val = max(1, int(len(seqs) * cfg.data.val_fraction))
        return seqs[:-n_val], seqs[-n_val:]
    if name == "sportsmot":
        train_root = Path(cfg.data.root) / cfg.data.train_split
        val_root = Path(cfg.data.root) / cfg.data.val_split
        train = load_sportsmot_split(train_root, limit=cfg.data.get("limit", None))
        val = load_sportsmot_split(val_root, limit=cfg.data.get("limit_val", None))
        return train, val
    if name == "soccernet":
        train_root = Path(cfg.data.root) / cfg.data.train_split
        val_root = Path(cfg.data.root) / cfg.data.val_split
        threshold = cfg.data.get("ball_area_threshold", 2000.0)
        prefer_gi = cfg.data.get("prefer_gameinfo_ball", True)
        train = load_soccernet_split(
            train_root,
            limit=cfg.data.get("limit", None),
            ball_area_threshold=threshold,
            prefer_gameinfo_ball=prefer_gi,
        )
        val = load_soccernet_split(
            val_root,
            limit=cfg.data.get("limit_val", None),
            ball_area_threshold=threshold,
            prefer_gameinfo_ball=prefer_gi,
        )
        return train, val
    raise ValueError(f"unknown dataset: {name}")


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    err = (pred - target) ** 2
    m = mask.unsqueeze(-1).to(err.dtype)
    return (err * m).sum() / m.sum().clamp_min(1.0) / 2.0


def evaluate(model, loader, device, horizons) -> dict[str, float]:
    model.eval() if isinstance(model, nn.Module) else None
    rm = RunningMetrics(horizons=horizons)
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            pred = model(batch)
            target = batch["fut_players"]
            mask = batch["fut_player_mask"] & batch["player_pad_mask"].unsqueeze(1)
            rm.update(pred, target, mask)
    return rm.compute()


def train_one(cfg, train_loader, val_loader, device) -> dict:
    """Train (or, for Kalman, just evaluate) one model."""
    model_name = cfg.model.name
    if model_name == "kalman":
        model = ConstantVelocityKalman(horizon=cfg.data.horizon)
        results = evaluate(model, val_loader, device, cfg.eval.horizons)
        return {"val": results, "history": []}

    model = MODEL_REGISTRY[model_name](horizon=cfg.data.horizon, **cfg.model.kwargs).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.train.epochs)
    history: list[dict] = []
    for epoch in range(cfg.train.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{cfg.train.epochs}", leave=False)
        running = 0.0
        n = 0
        for batch in pbar:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            pred = model(batch)
            target = batch["fut_players"]
            mask = batch["fut_player_mask"] & batch["player_pad_mask"].unsqueeze(1)
            loss = masked_mse(pred, target, mask)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()
            running += float(loss.detach()) * pred.shape[0]
            n += pred.shape[0]
            pbar.set_postfix(loss=f"{running/max(n,1):.5f}")
        sched.step()
        val = evaluate(model, val_loader, device, cfg.eval.horizons)
        history.append({"epoch": epoch + 1, "train_loss": running / max(n, 1), "val": val})
        if (epoch + 1) % cfg.train.log_every == 0 or epoch == cfg.train.epochs - 1:
            tqdm.write(f"epoch {epoch+1}: train_loss={running/max(n,1):.5f} val={val}")
    final = evaluate(model, val_loader, device, cfg.eval.horizons)
    return {"val": final, "history": history, "state_dict": model.state_dict()}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--out", type=str, default="results")
    args = parser.parse_args(argv)

    cfg = OmegaConf.load(args.config)
    set_seed(cfg.seed)
    device = get_device(cfg.get("device", None))

    train_seqs, val_seqs = build_dataset(cfg)
    train_ds = WindowDataset(
        train_seqs,
        history=cfg.data.history,
        horizon=cfg.data.horizon,
        stride=cfg.data.stride,
        min_players=cfg.data.min_players,
        require_ball_history=cfg.data.get("require_ball_history", False),
    )
    val_ds = WindowDataset(
        val_seqs,
        history=cfg.data.history,
        horizon=cfg.data.horizon,
        stride=cfg.data.horizon,  # non-overlapping for fair eval
        min_players=cfg.data.min_players,
        require_ball_history=cfg.data.get("require_ball_history", False),
    )
    print(f"Train windows: {len(train_ds)}    Val windows: {len(val_ds)}")
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        collate_fn=collate_windows,
        num_workers=cfg.train.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        collate_fn=collate_windows,
        num_workers=cfg.train.num_workers,
    )

    print(f"Device: {device}    Model: {cfg.model.name}")
    out = train_one(cfg, train_loader, val_loader, device)

    out_dir = ensure_dir(Path(args.out) / cfg.run_name)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"val": out["val"], "history": out["history"]}, f, indent=2)
    if "state_dict" in out:
        torch.save(out["state_dict"], out_dir / "model.pt")
    OmegaConf.save(cfg, out_dir / "config.yaml")
    print(f"\nFinal: {out['val']}")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
