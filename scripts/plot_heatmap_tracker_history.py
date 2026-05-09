"""Plot learning curves from soccernet_detect.train_heatmap_tracker history.json.

Example::

    python scripts/plot_heatmap_tracker_history.py \\
        --history /content/heatmap_tracker_full/history.json \\
        --out /content/heatmap_tracker_full/learning_curves.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--history", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args(argv)

    data = json.loads(args.history.read_text(encoding="utf-8"))
    if not data:
        raise SystemExit("empty history")

    out = args.out
    if out is None:
        out = args.history.parent / "learning_curves.png"

    epochs = [int(row["epoch"]) for row in data]
    has_val_loss = any("val_loss" in row for row in data)
    has_player = any("val_player_precision" in row for row in data)
    has_ball = any("val_ball_precision" in row for row in data)

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    # Losses
    ax = axes[0]
    ax.plot(epochs, [float(row["loss"]) for row in data], label="train loss", marker="o", ms=3)
    if has_val_loss:
        ax.plot(
            epochs,
            [float(row["val_loss"]) for row in data],
            label="val loss",
            marker="s",
            ms=3,
        )
    ax.set_ylabel("loss")
    ax.set_title("Heatmap tracker training")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Player detection (stored as 0–1 in history)
    ax = axes[1]
    if has_player:
        ax.plot(
            epochs,
            [100.0 * float(row["val_player_precision"]) for row in data],
            label="player precision",
            marker="o",
            ms=3,
        )
        ax.plot(
            epochs,
            [100.0 * float(row["val_player_recall"]) for row in data],
            label="player recall",
            marker="s",
            ms=3,
        )
        ax.set_ylabel("%")
        ax.set_ylim(0, 105)
        ax.legend()
    else:
        ax.text(
            0.5, 0.5, "no val_player_* in history", ha="center", va="center", transform=ax.transAxes
        )
    ax.set_title("Validation player detection")
    ax.grid(True, alpha=0.3)

    # Ball detection
    ax = axes[2]
    if has_ball:
        ax.plot(
            epochs,
            [100.0 * float(row["val_ball_precision"]) for row in data],
            label="ball precision",
            marker="o",
            ms=3,
        )
        ax.plot(
            epochs,
            [100.0 * float(row["val_ball_recall"]) for row in data],
            label="ball recall",
            marker="s",
            ms=3,
        )
        ax.set_ylabel("%")
        ax.set_xlabel("epoch")
        ax.set_ylim(0, 105)
        ax.legend()
    else:
        ax.text(
            0.5, 0.5, "no val_ball_* in history", ha="center", va="center", transform=ax.transAxes
        )
    ax.set_title("Validation ball detection")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
