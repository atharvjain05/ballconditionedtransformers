"""Export SoccerNet-Tracking MOT annotations to Ultralytics YOLO detection format.

Single class ``player`` (field players + goalkeepers). Uses the same ``gameinfo.ini``
logic as ``ballcond.data.soccernet`` via ``_soccernet_player_track_ids`` and ball
track exclusion.

Outputs
-------
Under ``--out`` (default layout)::

    <out>/
      images/train/<seq>_<frame>.jpg   # symlinks or copies of img1/XXXXXX.jpg
      images/val/...
      labels/train/<seq>_<frame>.txt  # one line per box: 0 cx cy w h (normalized)
      labels/val/...
      data.yaml                       # path, train, val, names, nc

By default, **image** copy/link is **skipped** when ``dest_img`` already exists
(on disk). Labels are still written every run. Use ``--force-recopy`` to replace images.

Train with Ultralytics (after ``pip install ultralytics``)::

    yolo detect train model=yolo11n.pt data=<out>/data.yaml epochs=50 imgsz=1280 \\
        batch=16 device=0

Or: ``python -m soccernet_detect.train_yolo --data <out>/data.yaml``
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import warnings
from pathlib import Path

import pandas as pd
import yaml
from tqdm.auto import tqdm  # notebook/Colab: widget bar; terminal: same as tqdm

from ballcond.data.soccernet import (
    _ball_track_ids_from_gameinfo,
    _parse_gameinfo_tracklets,
    _read_gt,
    _read_seqinfo,
    _soccernet_player_track_ids,
)


def mot_xywh_to_yolo_normalized(
    x: float,
    y: float,
    w: float,
    h: float,
    im_w: float,
    im_h: float,
) -> tuple[float, float, float, float]:
    """MOT top-left box (pixels) to YOLO normalized cx, cy, w, h in [0, 1]."""
    cx = (x + w / 2.0) / im_w
    cy = (y + h / 2.0) / im_h
    nw = w / im_w
    nh = h / im_h
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    nw = max(0.0, min(1.0, nw))
    nh = max(0.0, min(1.0, nh))
    return cx, cy, nw, nh


def _resolve_ball_ids(
    seq_dir: Path,
    gt: pd.DataFrame,
    prefer_gameinfo_ball: bool,
) -> list[int]:
    gt_ids = set(gt["id"].astype(int).unique())
    ball_ids: list[int] = []
    gameinfo_path = seq_dir / "gameinfo.ini"
    if prefer_gameinfo_ball and gameinfo_path.is_file():
        gi_ball_list = _ball_track_ids_from_gameinfo(gameinfo_path)
        cand = [tid for tid in gi_ball_list if tid in gt_ids]
        if cand:
            ball_ids = cand
    return ball_ids


def _player_id_set(
    seq_dir: Path,
    gt: pd.DataFrame,
    prefer_gameinfo_ball: bool,
) -> set[int]:
    gameinfo_path = seq_dir / "gameinfo.ini"
    tracklet_classes = _parse_gameinfo_tracklets(gameinfo_path) if gameinfo_path.is_file() else {}
    ball_ids = _resolve_ball_ids(seq_dir, gt, prefer_gameinfo_ball)
    ball_id_set = set(ball_ids)
    if ball_id_set:
        player_rows = gt[~gt["id"].isin(ball_id_set)]
    else:
        player_rows = gt
    player_ids = _soccernet_player_track_ids(player_rows, tracklet_classes)
    return set(player_ids)


def _link_or_copy_image(src: Path, dest: Path, copy: bool) -> None:
    """Copy or symlink ``src`` to ``dest`` (overwrites if present)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    if copy:
        shutil.copy2(src, dest)
        return
    try:
        os.symlink(src.resolve(), dest, target_is_directory=False)
    except OSError:
        shutil.copy2(src, dest)


def export_sequence_split(
    seq_dir: Path,
    split: str,
    out_root: Path,
    frame_stride: int,
    copy_images: bool,
    prefer_gameinfo_ball: bool,
    progress: bool = True,
    skip_existing_images: bool = True,
) -> int:
    """Export one sequence into ``out_root/images/{split}`` and ``labels/{split}``.

    Returns the number of image/label pairs written.

    ``progress``: if True, show a per-frame tqdm bar (off when called from
    :func:`export_dataset`, which uses a sequence-level bar only).
    """
    seq_dir = Path(seq_dir)
    seq_name = seq_dir.name
    gt_path = seq_dir / "gt" / "gt.txt"
    img_root = seq_dir / "img1"
    if not gt_path.is_file():
        warnings.warn(f"skip {seq_name}: missing {gt_path}", stacklevel=2)
        return 0
    if not (seq_dir / "seqinfo.ini").is_file():
        warnings.warn(f"skip {seq_name}: missing seqinfo.ini", stacklevel=2)
        return 0

    info = _read_seqinfo(seq_dir / "seqinfo.ini")
    im_w, im_h = float(info["imWidth"]), float(info["imHeight"])
    gt = _read_gt(gt_path)
    keep_ids = _player_id_set(seq_dir, gt, prefer_gameinfo_ball)

    img_out = out_root / "images" / split
    lbl_out = out_root / "labels" / split

    frames = sorted(gt["frame"].astype(int).unique().tolist())
    if frame_stride > 1:
        frames = frames[::frame_stride]

    n_written = 0
    frame_iter = frames
    if progress:
        frame_iter = tqdm(frames, desc=seq_name, leave=False, unit="frm")
    for f in frame_iter:
        src_img = img_root / f"{f:06d}.jpg"
        if not src_img.is_file():
            continue
        stem = f"{seq_name}_{f:06d}"
        dest_img = img_out / f"{stem}.jpg"
        dest_lbl = lbl_out / f"{stem}.txt"

        sub = gt[(gt["frame"] == f) & (gt["id"].astype(int).isin(keep_ids))]
        lines: list[str] = []
        for row in sub.itertuples(index=False):
            cx, cy, nw, nh = mot_xywh_to_yolo_normalized(
                float(row.x),
                float(row.y),
                float(row.w),
                float(row.h),
                im_w,
                im_h,
            )
            if nw <= 0 or nh <= 0:
                continue
            lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

        # Skip re-copy/re-link when resuming; labels below are still refreshed.
        if (not skip_existing_images) or (not dest_img.exists()):
            _link_or_copy_image(src_img, dest_img, copy_images)
        dest_lbl.parent.mkdir(parents=True, exist_ok=True)
        if lines:
            dest_lbl.write_text("".join(lines), encoding="utf-8")
        else:
            dest_lbl.write_text("", encoding="utf-8")
        n_written += 1
    return n_written


def export_dataset(
    tracking_root: Path,
    split_folder: str,
    out_root: Path,
    val_fraction: float = 0.15,
    seed: int = 0,
    frame_stride: int = 1,
    max_seqs: int | None = None,
    copy_images: bool = False,
    prefer_gameinfo_ball: bool = True,
    progress: bool = True,
    skip_existing_images: bool = True,
) -> Path:
    """Export ``tracking_root / split_folder`` to YOLO layout under ``out_root``.

    Returns path to ``data.yaml``.
    If ``progress`` is True, shows one tqdm bar over sequences (not per-frame).
    """
    track_split = Path(tracking_root) / split_folder
    if not track_split.is_dir():
        raise FileNotFoundError(f"not a directory: {track_split}")

    seq_dirs = sorted(
        p for p in track_split.iterdir() if p.is_dir() and (p / "seqinfo.ini").is_file()
    )
    if max_seqs is not None:
        seq_dirs = seq_dirs[:max_seqs]

    rng = random.Random(seed)
    seq_dirs_shuffled = list(seq_dirs)
    rng.shuffle(seq_dirs_shuffled)
    n_val = int(round(len(seq_dirs_shuffled) * val_fraction))
    if len(seq_dirs_shuffled) >= 2 and n_val == 0:
        n_val = 1
    if n_val >= len(seq_dirs_shuffled):
        n_val = max(0, len(seq_dirs_shuffled) - 1)
    val_set = set(seq_dirs_shuffled[:n_val])

    out_root = Path(out_root)
    (out_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / "val").mkdir(parents=True, exist_ok=True)

    total = 0
    seq_iter = seq_dirs
    if progress:
        seq_iter = tqdm(seq_dirs, desc="YOLO export", unit="seq")
    for d in seq_iter:
        sp = "val" if d in val_set else "train"
        total += export_sequence_split(
            d,
            sp,
            out_root,
            frame_stride,
            copy_images,
            prefer_gameinfo_ball,
            progress=False,
            skip_existing_images=skip_existing_images,
        )

    if total == 0:
        warnings.warn("no images exported — check paths and gt/img1", stacklevel=2)

    data = {
        "path": str(out_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["player"],
    }
    yaml_path = out_root / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)

    return yaml_path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="SoccerNet MOT → YOLO player detection dataset")
    p.add_argument(
        "--tracking-root",
        type=Path,
        required=True,
        help="e.g. /path/to/soccernet/tracking",
    )
    p.add_argument(
        "--split",
        default="train",
        help="subfolder under tracking (usually train)",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="output dataset directory (will contain images/, labels/, data.yaml)",
    )
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--max-seqs", type=int, default=None)
    p.add_argument(
        "--copy-images",
        action="store_true",
        help="copy JPEGs instead of symlinks (use on Drive/Windows if symlinks fail)",
    )
    p.add_argument(
        "--no-prefer-gameinfo-ball",
        action="store_true",
        help="do not exclude gameinfo ball ids from gt before player filter",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="disable tqdm progress bars",
    )
    p.add_argument(
        "--force-recopy",
        action="store_true",
        help="re-copy/re-link every image even if destination already exists",
    )
    args = p.parse_args(argv)

    yaml_path = export_dataset(
        tracking_root=args.tracking_root,
        split_folder=args.split,
        out_root=args.out,
        val_fraction=args.val_fraction,
        seed=args.seed,
        frame_stride=args.frame_stride,
        max_seqs=args.max_seqs,
        copy_images=args.copy_images,
        prefer_gameinfo_ball=not args.no_prefer_gameinfo_ball,
        progress=not args.no_progress,
        skip_existing_images=not args.force_recopy,
    )
    print(f"Wrote {yaml_path}")
    print(
        "Train: yolo detect train model=yolo11n.pt data={} epochs=50 imgsz=1280 batch=16 device=0".format(
            yaml_path
        )
    )


if __name__ == "__main__":
    main()
