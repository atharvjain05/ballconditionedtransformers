from __future__ import annotations

import textwrap
from pathlib import Path

from PIL import Image

from soccernet_detect.direct_dataset import BALL_CLASS, PLAYER_CLASS, SoccerNetFrameDetections
from soccernet_detect.eval_heatmap_tracker import (
    _empty_counts,
    _summarize_counts,
    _update_detection_counts,
)
from soccernet_detect.simple_tracker import Detection, SimpleTracker, box_iou


def test_direct_dataset_builds_player_and_ball_targets(tmp_path: Path) -> None:
    seq = tmp_path / "tracking" / "train" / "SNMOT-fake"
    _write_sequence(seq)

    ds = SoccerNetFrameDetections(tmp_path / "tracking", imgsz=128, output_stride=4, progress=False)
    item = ds[0]

    assert item["image"].shape == (3, 128, 128)
    assert item["heatmap"].shape == (2, 32, 32)
    assert item["heatmap"][PLAYER_CLASS].max() == 1.0
    assert item["heatmap"][BALL_CLASS].max() == 1.0
    assert item["wh_mask"].sum() == 2


def test_simple_tracker_keeps_matching_player_id() -> None:
    tracker = SimpleTracker()
    tracks_1 = tracker.update([Detection(PLAYER_CLASS, 0.9, (10, 10, 30, 50))])
    tracks_2 = tracker.update([Detection(PLAYER_CLASS, 0.8, (12, 11, 32, 51))])

    assert tracks_1[0].track_id == tracks_2[0].track_id
    assert box_iou(tracks_1[0].box_xyxy, tracks_2[0].box_xyxy) > 0.7


def test_eval_counts_player_and_ball_matches() -> None:
    counts = _empty_counts()
    preds = [
        Detection(PLAYER_CLASS, 0.9, (10, 10, 30, 50)),
        Detection(BALL_CLASS, 0.8, (100, 100, 108, 108)),
    ]
    gt = [
        (PLAYER_CLASS, 11, 11, 31, 51),
        (BALL_CLASS, 101, 101, 109, 109),
    ]

    _update_detection_counts(counts, preds, gt, player_iou=0.5, ball_center_dist=20.0)
    metrics = _summarize_counts(counts)

    assert metrics["player_precision"] == 1.0
    assert metrics["player_recall"] == 1.0
    assert metrics["ball_precision"] == 1.0
    assert metrics["ball_recall"] == 1.0


def _write_sequence(seq_dir: Path) -> None:
    (seq_dir / "img1").mkdir(parents=True)
    (seq_dir / "gt").mkdir()
    Image.new("RGB", (640, 360), color=(0, 128, 0)).save(seq_dir / "img1" / "000001.jpg")
    (seq_dir / "seqinfo.ini").write_text(
        textwrap.dedent("""\
            [Sequence]
            name=SNMOT-fake
            imWidth=640
            imHeight=360
            seqLength=1
            frameRate=25
            """),
        encoding="utf-8",
    )
    (seq_dir / "gameinfo.ini").write_text(
        textwrap.dedent("""\
            [Sequence]
            trackletID_1= player team left;10
            trackletID_2= ball;1
            trackletID_3= referee;main
            """),
        encoding="utf-8",
    )
    (seq_dir / "gt" / "gt.txt").write_text(
        "1,1,100,100,40,90,1,-1,-1,-1\n"
        "1,2,300,120,8,8,1,-1,-1,-1\n"
        "1,3,500,100,40,90,1,-1,-1,-1\n",
        encoding="utf-8",
    )
