"""Decode heatmap detections and link them into simple player+ball tracks."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from scipy.optimize import linear_sum_assignment

from .direct_dataset import BALL_CLASS, PLAYER_CLASS


@dataclass
class Detection:
    cls: int
    score: float
    box_xyxy: tuple[float, float, float, float]


@dataclass
class Track:
    track_id: int
    cls: int
    box_xyxy: tuple[float, float, float, float]
    missed: int = 0


def decode_detections(
    heatmap_logits: torch.Tensor,
    wh: torch.Tensor,
    *,
    imgsz: int,
    output_stride: int = 4,
    score_thresh: float = 0.3,
    topk: int = 100,
) -> list[Detection]:
    """Decode model outputs for one image into image-space boxes."""
    scores = torch.sigmoid(heatmap_logits).detach().cpu()
    wh = wh.detach().cpu()
    detections: list[Detection] = []

    for cls in range(scores.shape[0]):
        flat_scores = scores[cls].flatten()
        k = min(topk, flat_scores.numel())
        vals, inds = torch.topk(flat_scores, k=k)
        width = scores.shape[-1]
        for val, ind in zip(vals.tolist(), inds.tolist(), strict=True):
            if val < score_thresh:
                continue
            gy, gx = divmod(ind, width)
            bw = float(wh[0, gy, gx]) * imgsz
            bh = float(wh[1, gy, gx]) * imgsz
            cx = (gx + 0.5) * output_stride
            cy = (gy + 0.5) * output_stride
            detections.append(
                Detection(
                    cls=cls,
                    score=float(val),
                    box_xyxy=(
                        max(0.0, cx - bw / 2.0),
                        max(0.0, cy - bh / 2.0),
                        min(float(imgsz), cx + bw / 2.0),
                        min(float(imgsz), cy + bh / 2.0),
                    ),
                )
            )
    return non_max_suppression(detections)


def non_max_suppression(detections: list[Detection], iou_thresh: float = 0.5) -> list[Detection]:
    """Small class-aware NMS for decoded heatmap peaks."""
    out: list[Detection] = []
    for cls in (PLAYER_CLASS, BALL_CLASS):
        cls_dets = sorted(
            (d for d in detections if d.cls == cls), key=lambda d: d.score, reverse=True
        )
        while cls_dets:
            best = cls_dets.pop(0)
            out.append(best)
            cls_dets = [d for d in cls_dets if box_iou(best.box_xyxy, d.box_xyxy) < iou_thresh]
    return out


class SimpleTracker:
    """IoU/distance tracker for players, nearest-center tracker for ball."""

    def __init__(
        self, max_missed: int = 10, iou_weight: float = 1.0, dist_weight: float = 0.002
    ) -> None:
        self.max_missed = max_missed
        self.iou_weight = iou_weight
        self.dist_weight = dist_weight
        self.next_id = 1
        self.tracks: list[Track] = []

    def update(self, detections: list[Detection]) -> list[Track]:
        updated: list[Track] = []
        for cls in (PLAYER_CLASS, BALL_CLASS):
            cls_tracks = [t for t in self.tracks if t.cls == cls]
            cls_dets = [d for d in detections if d.cls == cls]
            updated.extend(self._update_class(cls_tracks, cls_dets, cls))
        self.tracks = [t for t in updated if t.missed <= self.max_missed]
        return list(self.tracks)

    def _update_class(
        self, tracks: list[Track], detections: list[Detection], cls: int
    ) -> list[Track]:
        if not tracks:
            return [self._new_track(d) for d in detections]
        if not detections:
            return [Track(t.track_id, t.cls, t.box_xyxy, t.missed + 1) for t in tracks]

        cost = torch.zeros((len(tracks), len(detections)), dtype=torch.float32)
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                distance = center_distance(track.box_xyxy, det.box_xyxy)
                if cls == BALL_CLASS:
                    cost[i, j] = distance
                else:
                    cost[i, j] = (
                        -self.iou_weight * box_iou(track.box_xyxy, det.box_xyxy)
                        + self.dist_weight * distance
                    )

        row_ind, col_ind = linear_sum_assignment(cost.numpy())
        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        out: list[Track] = []
        for i, j in zip(row_ind.tolist(), col_ind.tolist(), strict=True):
            if cls == BALL_CLASS and float(cost[i, j]) > 80.0:
                continue
            if cls == PLAYER_CLASS and float(cost[i, j]) > 0.5:
                continue
            out.append(Track(tracks[i].track_id, cls, detections[j].box_xyxy, missed=0))
            matched_tracks.add(i)
            matched_dets.add(j)

        for i, track in enumerate(tracks):
            if i not in matched_tracks:
                out.append(Track(track.track_id, track.cls, track.box_xyxy, track.missed + 1))
        for j, det in enumerate(detections):
            if j not in matched_dets:
                out.append(self._new_track(det))
        return out

    def _new_track(self, det: Detection) -> Track:
        track = Track(self.next_id, det.cls, det.box_xyxy, missed=0)
        self.next_id += 1
        return track


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def center_distance(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    acx, acy = (a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0
    bcx, bcy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
    return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5
