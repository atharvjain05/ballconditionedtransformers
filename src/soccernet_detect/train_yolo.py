"""Thin wrapper around Ultralytics YOLO detection training.

Requires: ``pip install -r requirements-detect.txt``

Example::

    python -m soccernet_detect.train_yolo --data runs/soccernet_player/data.yaml \\
        --model yolo11n.pt --epochs 50 --imgsz 1280 --batch 16 --device 0

Equivalent CLI::

    yolo detect train model=yolo11n.pt data=runs/soccernet_player/data.yaml \\
        epochs=50 imgsz=1280 batch=16 device=0
"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit("Install ultralytics: pip install -r requirements-detect.txt") from e

    p = argparse.ArgumentParser(description="Train YOLO detector on exported data.yaml")
    p.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    p.add_argument("--model", type=str, default="yolo11n.pt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", type=str, default="0")
    args = p.parse_args(argv)

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )


if __name__ == "__main__":
    main()
