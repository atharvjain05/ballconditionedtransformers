"""Train a small player+ball heatmap detector directly from SoccerNet-Tracking.

This intentionally avoids YOLO export. Images are loaded from ``img1/`` and
targets are generated from each sequence's MOT ``gt.txt`` at training time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .direct_dataset import CLASS_NAMES, SoccerNetFrameDetections
from .eval_heatmap_tracker import _empty_counts, _summarize_counts, _update_detection_counts
from .heatmap_model import HeatmapBoxDetector, detection_loss
from .simple_tracker import decode_detections


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tracking-root", type=Path, required=True)
    p.add_argument("--split", default="train")
    p.add_argument(
        "--val-split",
        default=None,
        help="optional labeled validation split; when set, eval runs every epoch and best.pt is saved",
    )
    p.add_argument("--out", type=Path, default=Path("runs/heatmap_tracker"))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--output-stride", type=int, default=4)
    p.add_argument("--frame-stride", type=int, default=5)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--max-val-frames", type=int, default=None)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--base-channels", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default=None)
    p.add_argument(
        "--no-index-progress", action="store_true", help="hide pre-training dataset scan bars"
    )
    p.add_argument(
        "--no-val-det-metrics",
        action="store_true",
        help="skip per-epoch val detection P/R (faster; still logs val loss)",
    )
    p.add_argument("--val-score-thresh", type=float, default=0.3)
    p.add_argument("--val-player-iou", type=float, default=0.5)
    p.add_argument("--val-ball-center-dist", type=float, default=20.0)
    args = p.parse_args(argv)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Indexing {args.split!r} frames from {args.tracking_root} ...", flush=True)
    train_ds = SoccerNetFrameDetections(
        args.tracking_root,
        split=args.split,
        imgsz=args.imgsz,
        output_stride=args.output_stride,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        progress=not args.no_index_progress,
    )
    val_ds = None
    if args.val_split:
        print(f"Indexing {args.val_split!r} frames from {args.tracking_root} ...", flush=True)
        val_ds = SoccerNetFrameDetections(
            args.tracking_root,
            split=args.val_split,
            imgsz=args.imgsz,
            output_stride=args.output_stride,
            frame_stride=args.frame_stride,
            max_frames=args.max_val_frames,
            progress=not args.no_index_progress,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=args.batch,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        if val_ds is not None
        else None
    )

    model = HeatmapBoxDetector(num_classes=len(CLASS_NAMES), base_channels=args.base_channels).to(
        device
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    args.out.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int | None]] = []
    best_val_loss = float("inf")
    best_epoch: int | None = None
    print(f"Device: {device}")
    print(f"Train frames: {len(train_ds)}")
    if val_ds is not None:
        print(f"Val frames: {len(val_ds)}")
    else:
        print("No --val-split provided; best.pt will not be written.")
    print("Starting training loop ...", flush=True)

    for epoch in range(1, args.epochs + 1):
        train_stats = _run_epoch(
            model, train_loader, device, opt, desc=f"epoch {epoch}/{args.epochs}"
        )
        if val_loader:
            if args.no_val_det_metrics:
                val_stats = _run_epoch(model, val_loader, device, None, desc="val")
            else:
                val_stats = _run_val_with_det_metrics(
                    model,
                    val_loader,
                    val_ds,
                    device,
                    imgsz=args.imgsz,
                    output_stride=args.output_stride,
                    score_thresh=args.val_score_thresh,
                    player_iou=args.val_player_iou,
                    ball_center_dist=args.val_ball_center_dist,
                )
        else:
            val_stats = None
        sched.step()

        row: dict[str, float | int | None] = {"epoch": epoch, **train_stats}
        if val_stats:
            row.update({f"val_{k}": v for k, v in val_stats.items()})
        history.append(row)

        is_best = False
        if val_stats is not None and val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            best_epoch = epoch
            is_best = True

        val_msg = f" val_loss={val_stats['loss']:.4f}" if val_stats else ""
        if val_stats and not args.no_val_det_metrics and "player_precision" in val_stats:
            pp = float(val_stats["player_precision"]) * 100.0
            pr = float(val_stats["player_recall"]) * 100.0
            bp = float(val_stats["ball_precision"]) * 100.0
            br = float(val_stats["ball_recall"]) * 100.0
            val_msg += f" val_player P={pp:.1f}% R={pr:.1f}% val_ball P={bp:.1f}% R={br:.1f}%"
        print(f"epoch {epoch}: loss={train_stats['loss']:.4f}{val_msg}")
        checkpoint = {
            "model": model.state_dict(),
            "args": _jsonable_args(args),
            "class_names": CLASS_NAMES,
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "val_loss": val_stats["loss"] if val_stats else None,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss if best_epoch is not None else None,
        }
        torch.save(checkpoint, args.out / "last.pt")
        if is_best:
            torch.save(checkpoint, args.out / "best.pt")
            print(f"  new best: val_loss={best_val_loss:.4f} at epoch {epoch}")

    torch.save(
        {
            "model": model.state_dict(),
            "args": _jsonable_args(args),
            "class_names": CLASS_NAMES,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss if best_epoch is not None else None,
        },
        args.out / "model.pt",
    )
    with open(args.out / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Saved to {args.out}")


def _run_val_with_det_metrics(
    model: HeatmapBoxDetector,
    loader: DataLoader,
    ds: SoccerNetFrameDetections,
    device: torch.device,
    *,
    imgsz: int,
    output_stride: int,
    score_thresh: float,
    player_iou: float,
    ball_center_dist: float,
) -> dict[str, float]:
    """Validation loss plus detection precision/recall (same matching as eval_heatmap_tracker)."""
    model.eval()
    totals = {"loss": 0.0, "heatmap_loss": 0.0, "wh_loss": 0.0}
    counts = _empty_counts()
    n = 0
    with torch.no_grad():
        pbar = tqdm(loader, desc="val", leave=False)
        for batch in pbar:
            image = batch["image"].to(device)
            heatmap = batch["heatmap"].to(device)
            wh = batch["wh"].to(device)
            wh_mask = batch["wh_mask"].to(device)
            pred = model(image)
            _, stats = detection_loss(pred, heatmap, wh, wh_mask)
            batch_size = image.shape[0]
            n += batch_size
            for k in totals:
                totals[k] += stats[k] * batch_size
            for bi in range(batch_size):
                idx = int(batch["index"][bi])
                pred_dets = decode_detections(
                    pred["heatmap_logits"][bi],
                    pred["wh"][bi],
                    imgsz=imgsz,
                    output_stride=output_stride,
                    score_thresh=score_thresh,
                )
                gt_boxes = ds.ground_truth_boxes(idx)
                _update_detection_counts(
                    counts,
                    pred_dets,
                    gt_boxes,
                    player_iou=player_iou,
                    ball_center_dist=ball_center_dist,
                )
            pbar.set_postfix(loss=f"{totals['loss'] / max(n, 1):.4f}")
    out: dict[str, float] = {k: v / max(n, 1) for k, v in totals.items()}
    det = _summarize_counts(counts)
    for k, v in det.items():
        out[k] = float(v) if isinstance(v, (int, float)) else float(v)
    return out


def _run_epoch(
    model: HeatmapBoxDetector,
    loader: DataLoader | None,
    device: torch.device,
    opt: torch.optim.Optimizer | None,
    desc: str,
) -> dict[str, float]:
    if loader is None:
        return {}
    training = opt is not None
    model.train(training)
    totals = {"loss": 0.0, "heatmap_loss": 0.0, "wh_loss": 0.0}
    n = 0
    with torch.set_grad_enabled(training):
        pbar = tqdm(loader, desc=desc, leave=False)
        for batch in pbar:
            image = batch["image"].to(device)
            heatmap = batch["heatmap"].to(device)
            wh = batch["wh"].to(device)
            wh_mask = batch["wh_mask"].to(device)
            pred = model(image)
            loss, stats = detection_loss(pred, heatmap, wh, wh_mask)
            if training:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            batch_size = image.shape[0]
            n += batch_size
            for k in totals:
                totals[k] += stats[k] * batch_size
            pbar.set_postfix(loss=f"{totals['loss'] / max(n, 1):.4f}")
    return {k: v / max(n, 1) for k, v in totals.items()}


def _jsonable_args(args: argparse.Namespace) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


if __name__ == "__main__":
    main()
