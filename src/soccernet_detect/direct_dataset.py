"""In-place SoccerNet frame dataset for a small player+ball detector.

This dataset reads SoccerNet-Tracking's native MOT layout directly:

    tracking/<split>/SNMOT-*/img1/000001.jpg
    tracking/<split>/SNMOT-*/gt/gt.txt
    tracking/<split>/SNMOT-*/gameinfo.ini

No YOLO export, copied images, or label files are needed. Targets are generated
in memory as CenterNet-style heatmaps plus width/height regression maps.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from ballcond.data.soccernet import (
    _ball_id_area_heuristic,
    _ball_track_ids_from_gameinfo,
    _parse_gameinfo_tracklets,
    _read_gt,
    _read_seqinfo,
    _soccernet_player_track_ids,
)

PLAYER_CLASS = 0
BALL_CLASS = 1
CLASS_NAMES = ("player", "ball")


@dataclass(frozen=True)
class FrameRecord:
    seq_dir: Path
    frame: int
    image_path: Path
    width: int
    height: int


class SoccerNetFrameDetections(Dataset):
    """Frame-level player+ball detection dataset generated from MOT annotations."""

    def __init__(
        self,
        tracking_root: Path,
        split: str = "train",
        imgsz: int = 512,
        output_stride: int = 4,
        frame_stride: int = 1,
        max_frames: int | None = None,
        ball_area_threshold: float = 2000.0,
        prefer_gameinfo_ball: bool = True,
        progress: bool = True,
    ) -> None:
        self.tracking_root = Path(tracking_root)
        self.split = split
        self.imgsz = int(imgsz)
        self.output_stride = int(output_stride)
        self.out_size = self.imgsz // self.output_stride
        self.frame_stride = max(1, int(frame_stride))
        self.annotations: dict[tuple[str, int], np.ndarray] = {}
        self.records: list[FrameRecord] = []

        split_root = self.tracking_root / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"not a directory: {split_root}")

        seq_dirs = sorted(p for p in split_root.iterdir() if p.is_dir())
        seq_iter = seq_dirs
        if progress:
            seq_iter = tqdm(seq_dirs, desc=f"index {split}", unit="seq")
        for seq_dir in seq_iter:
            if not (seq_dir / "seqinfo.ini").is_file() or not (seq_dir / "gt" / "gt.txt").is_file():
                continue
            self._add_sequence(seq_dir, ball_area_threshold, prefer_gameinfo_ball)
            if progress:
                seq_iter.set_postfix(frames=len(self.records))
            if max_frames is not None and len(self.records) >= max_frames:
                self.records = self.records[:max_frames]
                if progress:
                    seq_iter.set_postfix(frames=len(self.records), max_frames=max_frames)
                break

        if not self.records:
            raise RuntimeError(f"no frames found under {split_root}")

    def _add_sequence(
        self,
        seq_dir: Path,
        ball_area_threshold: float,
        prefer_gameinfo_ball: bool,
    ) -> None:
        info = _read_seqinfo(seq_dir / "seqinfo.ini")
        gt = _read_gt(seq_dir / "gt" / "gt.txt")
        tracklet_classes = (
            _parse_gameinfo_tracklets(seq_dir / "gameinfo.ini")
            if (seq_dir / "gameinfo.ini").is_file()
            else {}
        )

        gt_ids = set(gt["id"].astype(int).unique())
        ball_ids: list[int] = []
        if prefer_gameinfo_ball and (seq_dir / "gameinfo.ini").is_file():
            ball_ids = [
                tid
                for tid in _ball_track_ids_from_gameinfo(seq_dir / "gameinfo.ini")
                if tid in gt_ids
            ]
        if not ball_ids:
            ball_id = _ball_id_area_heuristic(gt, ball_area_threshold)
            if ball_id is not None:
                ball_ids = [ball_id]

        ball_id_set = set(ball_ids)
        player_rows = gt[~gt["id"].isin(ball_id_set)] if ball_id_set else gt
        player_ids = set(_soccernet_player_track_ids(player_rows, tracklet_classes))

        rows: list[tuple[int, int, float, float, float, float]] = []
        for row in gt.itertuples(index=False):
            track_id = int(row.id)
            if track_id in player_ids:
                cls = PLAYER_CLASS
            elif track_id in ball_id_set:
                cls = BALL_CLASS
            else:
                continue
            rows.append(
                (int(row.frame), cls, float(row.x), float(row.y), float(row.w), float(row.h))
            )

        ann = pd.DataFrame(rows, columns=["frame", "cls", "x", "y", "w", "h"])
        if ann.empty:
            return

        for frame, frame_rows in ann.groupby("frame"):
            f = int(frame)
            if (f - 1) % self.frame_stride != 0:
                continue
            image_path = seq_dir / "img1" / f"{f:06d}.jpg"
            if not image_path.is_file():
                continue
            key = (seq_dir.name, f)
            self.annotations[key] = frame_rows[["cls", "x", "y", "w", "h"]].to_numpy(np.float32)
            self.records.append(
                FrameRecord(
                    seq_dir=seq_dir,
                    frame=f,
                    image_path=image_path,
                    width=int(info["imWidth"]),
                    height=int(info["imHeight"]),
                )
            )

    def __len__(self) -> int:
        return len(self.records)

    def ground_truth_boxes(self, idx: int) -> list[tuple[int, float, float, float, float]]:
        """Return resized image-space ``(cls, x1, y1, x2, y2)`` boxes for one frame."""
        rec = self.records[idx]
        anns = self.annotations[(rec.seq_dir.name, rec.frame)]
        sx = self.imgsz / rec.width
        sy = self.imgsz / rec.height
        boxes: list[tuple[int, float, float, float, float]] = []
        for cls_f, x, y, w, h in anns:
            x1 = max(0.0, float(x * sx))
            y1 = max(0.0, float(y * sy))
            x2 = min(float(self.imgsz), float((x + w) * sx))
            y2 = min(float(self.imgsz), float((y + h) * sy))
            if x2 > x1 and y2 > y1:
                boxes.append((int(cls_f), x1, y1, x2, y2))
        return boxes

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | int]:
        rec = self.records[idx]
        image = Image.open(rec.image_path).convert("RGB").resize((self.imgsz, self.imgsz))
        image_arr = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_arr).permute(2, 0, 1)

        heatmap = torch.zeros((len(CLASS_NAMES), self.out_size, self.out_size), dtype=torch.float32)
        wh = torch.zeros((2, self.out_size, self.out_size), dtype=torch.float32)
        wh_mask = torch.zeros((1, self.out_size, self.out_size), dtype=torch.float32)

        anns = self.annotations[(rec.seq_dir.name, rec.frame)]
        sx = self.imgsz / rec.width
        sy = self.imgsz / rec.height
        for cls_f, x, y, w, h in anns:
            cls = int(cls_f)
            cx = (x + w / 2.0) * sx
            cy = (y + h / 2.0) * sy
            bw = w * sx
            bh = h * sy
            gx = int(cx / self.output_stride)
            gy = int(cy / self.output_stride)
            if not (0 <= gx < self.out_size and 0 <= gy < self.out_size):
                continue
            radius = 1 if cls == BALL_CLASS else 2
            _draw_gaussian(heatmap[cls], gx, gy, radius)
            wh[:, gy, gx] = torch.tensor([bw / self.imgsz, bh / self.imgsz], dtype=torch.float32)
            wh_mask[:, gy, gx] = 1.0

        return {
            "image": image_tensor,
            "heatmap": heatmap,
            "wh": wh,
            "wh_mask": wh_mask,
            "seq": rec.seq_dir.name,
            "frame": rec.frame,
            "index": idx,
        }


def _draw_gaussian(heatmap: torch.Tensor, gx: int, gy: int, radius: int) -> None:
    """Draw a tiny normalized Gaussian centered at an output-grid location."""
    h, w = heatmap.shape
    sigma = max(radius / 2.0, 0.5)
    for yy in range(max(0, gy - radius), min(h, gy + radius + 1)):
        for xx in range(max(0, gx - radius), min(w, gx + radius + 1)):
            dist2 = float((xx - gx) ** 2 + (yy - gy) ** 2)
            val = float(np.exp(-dist2 / (2.0 * sigma * sigma)))
            heatmap[yy, xx] = max(float(heatmap[yy, xx]), val)
