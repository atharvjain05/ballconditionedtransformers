"""Save images with HeatmapBoxDetector prediction boxes overlaid.

Example:

    python -m soccernet_detect.visualize_heatmap_predictions \
      --checkpoint /content/heatmap_tracker_full/model.pt \
      --tracking-root /content/drive/MyDrive/soccernet/tracking \
      --split test \
      --out /content/heatmap_pred_vis \
      --num-images 24 \
      --frame-stride 20 \
      --score-thresh 0.35 \
      --device cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

from .direct_dataset import BALL_CLASS, CLASS_NAMES, PLAYER_CLASS, SoccerNetFrameDetections
from .heatmap_model import HeatmapBoxDetector
from .simple_tracker import Detection, decode_detections


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tracking-root", type=Path, required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--out", type=Path, default=Path("runs/heatmap_prediction_vis"))
    p.add_argument("--num-images", type=int, default=24)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--output-stride", type=int, default=None)
    p.add_argument("--base-channels", type=int, default=None)
    p.add_argument("--frame-stride", type=int, default=5)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--score-thresh", type=float, default=0.3)
    p.add_argument("--topk", type=int, default=100)
    p.add_argument("--device", default=None)
    p.add_argument("--show-gt", action="store_true", help="also draw ground-truth boxes")
    args = p.parse_args(argv)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt["model"]
    train_args = ckpt.get("args", {})
    imgsz = int(args.imgsz or train_args.get("imgsz", 512))
    output_stride = int(args.output_stride or train_args.get("output_stride", 4))
    base_channels = int(
        args.base_channels or train_args.get("base_channels") or _infer_base_channels(state)
    )
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    print(
        f"Loading model: imgsz={imgsz}, output_stride={output_stride}, "
        f"base_channels={base_channels}",
        flush=True,
    )
    model = HeatmapBoxDetector(num_classes=len(CLASS_NAMES), base_channels=base_channels).to(device)
    model.load_state_dict(state)
    model.eval()

    ds = SoccerNetFrameDetections(
        args.tracking_root,
        split=args.split,
        imgsz=imgsz,
        output_stride=output_stride,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    all_detections: list[dict[str, object]] = []
    end = min(len(ds), args.start_index + args.num_images)
    indices = range(args.start_index, end)

    with torch.no_grad():
        for save_i, idx in enumerate(tqdm(indices, desc="visualize", unit="img")):
            sample = ds[idx]
            image_tensor = sample["image"].unsqueeze(0).to(device)
            pred = model(image_tensor)
            detections = decode_detections(
                pred["heatmap_logits"][0],
                pred["wh"][0],
                imgsz=imgsz,
                output_stride=output_stride,
                score_thresh=args.score_thresh,
                topk=args.topk,
            )

            rec = ds.records[idx]
            out_path = args.out / f"{save_i:03d}_{rec.seq_dir.name}_{rec.frame:06d}.jpg"
            gt_boxes = ds.ground_truth_boxes(idx) if args.show_gt else []
            _save_prediction_overlay(ds, idx, detections, gt_boxes, out_path)
            all_detections.append(
                {
                    "image": str(out_path),
                    "sequence": rec.seq_dir.name,
                    "frame": rec.frame,
                    "detections": [
                        {
                            "class": CLASS_NAMES[d.cls],
                            "score": d.score,
                            "box_xyxy": list(d.box_xyxy),
                        }
                        for d in detections
                    ],
                }
            )

    with open(args.out / "detections.json", "w", encoding="utf-8") as f:
        json.dump(all_detections, f, indent=2)
    print(f"Saved {len(all_detections)} visualizations to {args.out}")


def _infer_base_channels(state: dict[str, torch.Tensor]) -> int:
    """Infer model width from the first convolution weight in older checkpoints."""
    key = "stem.net.0.weight"
    if key not in state:
        raise ValueError(
            "checkpoint does not include args.base_channels and stem weight is missing"
        )
    return int(state[key].shape[0])


def _save_prediction_overlay(
    ds: SoccerNetFrameDetections,
    idx: int,
    detections: list[Detection],
    gt_boxes: list[tuple[int, float, float, float, float]],
    out_path: Path,
) -> None:
    rec = ds.records[idx]
    image = Image.open(rec.image_path).convert("RGB").resize((ds.imgsz, ds.imgsz))
    draw = ImageDraw.Draw(image)

    for cls, x1, y1, x2, y2 in gt_boxes:
        color = "lime" if cls == PLAYER_CLASS else "yellow"
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
        draw.text((x1, y1), f"gt {CLASS_NAMES[cls]}", fill=color)

    for det in detections:
        color = "red" if det.cls == PLAYER_CLASS else "orange"
        x1, y1, x2, y2 = det.box_xyxy
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        draw.text((x1, max(0.0, y1 - 12.0)), f"{CLASS_NAMES[det.cls]} {det.score:.2f}", fill=color)

    image.save(out_path)


if __name__ == "__main__":
    main()
