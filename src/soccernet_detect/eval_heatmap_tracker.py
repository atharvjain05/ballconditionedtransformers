"""Evaluate a trained heatmap detector on a held-out SoccerNet split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from tqdm import tqdm

from .direct_dataset import BALL_CLASS, CLASS_NAMES, PLAYER_CLASS, SoccerNetFrameDetections
from .heatmap_model import HeatmapBoxDetector, detection_loss
from .simple_tracker import Detection, box_iou, center_distance, decode_detections


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tracking-root", type=Path, required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--out", type=Path, default=Path("runs/heatmap_eval"))
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--output-stride", type=int, default=None)
    p.add_argument("--base-channels", type=int, default=None)
    p.add_argument("--frame-stride", type=int, default=5)
    p.add_argument("--max-frames", type=int, default=2000)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default=None)
    p.add_argument("--score-thresh", type=float, default=0.3)
    p.add_argument("--player-iou", type=float, default=0.5)
    p.add_argument("--ball-center-dist", type=float, default=20.0)
    p.add_argument("--save-vis", type=int, default=8, help="number of prediction overlays to save")
    args = p.parse_args(argv)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    train_args = ckpt.get("args", {})
    imgsz = int(args.imgsz or train_args.get("imgsz", 512))
    output_stride = int(args.output_stride or train_args.get("output_stride", 4))
    base_channels = int(args.base_channels or train_args.get("base_channels", 42))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"Indexing {args.split!r} eval frames from {args.tracking_root} ...", flush=True)
    ds = SoccerNetFrameDetections(
        args.tracking_root,
        split=args.split,
        imgsz=imgsz,
        output_stride=output_stride,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = HeatmapBoxDetector(num_classes=len(CLASS_NAMES), base_channels=base_channels).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    args.out.mkdir(parents=True, exist_ok=True)
    vis_dir = args.out / "vis"
    if args.save_vis > 0:
        vis_dir.mkdir(parents=True, exist_ok=True)

    totals = {"loss": 0.0, "heatmap_loss": 0.0, "wh_loss": 0.0}
    counts = _empty_counts()
    n = 0
    vis_saved = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"eval {args.split}", unit="batch"):
            image = batch["image"].to(device)
            heatmap = batch["heatmap"].to(device)
            wh = batch["wh"].to(device)
            wh_mask = batch["wh_mask"].to(device)
            pred = model(image)
            loss, stats = detection_loss(pred, heatmap, wh, wh_mask)
            batch_size = image.shape[0]
            n += batch_size
            for key in totals:
                totals[key] += stats[key] * batch_size

            for bi in range(batch_size):
                idx = int(batch["index"][bi])
                pred_dets = decode_detections(
                    pred["heatmap_logits"][bi],
                    pred["wh"][bi],
                    imgsz=imgsz,
                    output_stride=output_stride,
                    score_thresh=args.score_thresh,
                )
                gt_boxes = ds.ground_truth_boxes(idx)
                _update_detection_counts(
                    counts,
                    pred_dets,
                    gt_boxes,
                    player_iou=args.player_iou,
                    ball_center_dist=args.ball_center_dist,
                )
                if vis_saved < args.save_vis:
                    _save_overlay(
                        ds,
                        idx,
                        pred_dets,
                        gt_boxes,
                        vis_dir
                        / f"{vis_saved:03d}_{batch['seq'][bi]}_{int(batch['frame'][bi]):06d}.jpg",
                    )
                    vis_saved += 1

    metrics = {key: value / max(n, 1) for key, value in totals.items()}
    metrics.update(_summarize_counts(counts))
    with open(args.out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Saved eval outputs to {args.out}")


def _empty_counts() -> dict[str, dict[str, int]]:
    return {name: {"tp": 0, "fp": 0, "fn": 0} for name in CLASS_NAMES}


def _update_detection_counts(
    counts: dict[str, dict[str, int]],
    pred_dets: list[Detection],
    gt_boxes: list[tuple[int, float, float, float, float]],
    *,
    player_iou: float,
    ball_center_dist: float,
) -> None:
    for cls, name in ((PLAYER_CLASS, "player"), (BALL_CLASS, "ball")):
        preds = [d for d in pred_dets if d.cls == cls]
        gts = [g for g in gt_boxes if g[0] == cls]
        matched_gt: set[int] = set()
        for pred in sorted(preds, key=lambda d: d.score, reverse=True):
            best_j = None
            best_score = float("-inf")
            for j, gt in enumerate(gts):
                if j in matched_gt:
                    continue
                gt_box = gt[1:]
                score = (
                    -center_distance(pred.box_xyxy, gt_box)
                    if cls == BALL_CLASS
                    else box_iou(pred.box_xyxy, gt_box)
                )
                if score > best_score:
                    best_score = score
                    best_j = j
            if best_j is None:
                counts[name]["fp"] += 1
                continue
            if cls == BALL_CLASS:
                ok = -best_score <= ball_center_dist
            else:
                ok = best_score >= player_iou
            if ok:
                matched_gt.add(best_j)
                counts[name]["tp"] += 1
            else:
                counts[name]["fp"] += 1
        counts[name]["fn"] += len(gts) - len(matched_gt)


def _summarize_counts(counts: dict[str, dict[str, int]]) -> dict[str, float | int]:
    out: dict[str, float | int] = {}
    for name, c in counts.items():
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        out[f"{name}_tp"] = tp
        out[f"{name}_fp"] = fp
        out[f"{name}_fn"] = fn
        out[f"{name}_precision"] = tp / max(tp + fp, 1)
        out[f"{name}_recall"] = tp / max(tp + fn, 1)
    return out


def _save_overlay(
    ds: SoccerNetFrameDetections,
    idx: int,
    pred_dets: list[Detection],
    gt_boxes: list[tuple[int, float, float, float, float]],
    out_path: Path,
) -> None:
    rec = ds.records[idx]
    image = Image.open(rec.image_path).convert("RGB").resize((ds.imgsz, ds.imgsz))
    draw = ImageDraw.Draw(image)
    for cls, x1, y1, x2, y2 in gt_boxes:
        color = "lime" if cls == PLAYER_CLASS else "yellow"
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
    for det in pred_dets:
        color = "red" if det.cls == PLAYER_CLASS else "orange"
        draw.rectangle(det.box_xyxy, outline=color, width=2)
        draw.text(
            (det.box_xyxy[0], det.box_xyxy[1]),
            f"{CLASS_NAMES[det.cls]} {det.score:.2f}",
            fill=color,
        )
    image.save(out_path)


if __name__ == "__main__":
    main()
