from __future__ import annotations

import textwrap
from pathlib import Path

from unittest.mock import patch

import pytest

from soccernet_detect.export_yolo import (
    export_dataset,
    export_sequence_split,
    mot_xywh_to_yolo_normalized,
    _link_or_copy_image,
)


def test_mot_xywh_to_yolo_normalized() -> None:
    cx, cy, nw, nh = mot_xywh_to_yolo_normalized(100.0, 200.0, 400.0, 200.0, 1000.0, 800.0)
    assert cx == pytest.approx(300 / 1000)
    assert cy == pytest.approx(300 / 800)
    assert nw == pytest.approx(400 / 1000)
    assert nh == pytest.approx(200 / 800)


def test_link_or_copy_creates_dest(tmp_path: Path) -> None:
    src = tmp_path / "a.jpg"
    dest = tmp_path / "out" / "b.jpg"
    src.write_bytes(b"jpeg-bytes-here")
    _link_or_copy_image(src, dest, copy=True)
    assert dest.read_bytes() == b"jpeg-bytes-here"


def test_link_or_copy_second_call_updates_content(tmp_path: Path) -> None:
    src = tmp_path / "a.jpg"
    dest = tmp_path / "out" / "b.jpg"
    src.write_bytes(b"v1")
    _link_or_copy_image(src, dest, copy=True)
    assert dest.read_bytes() == b"v1"
    src.write_bytes(b"v2")
    _link_or_copy_image(src, dest, copy=True)
    assert dest.read_bytes() == b"v2"


def test_export_sequence_split_second_run_skips_copy2(tmp_path: Path) -> None:
    seq = tmp_path / "SNMOT-fake"
    _write_minimal_mot_sequence(seq, with_gameinfo=True)
    out_root = tmp_path / "yolo"
    export_sequence_split(
        seq,
        "train",
        out_root,
        frame_stride=1,
        copy_images=True,
        prefer_gameinfo_ball=True,
        progress=False,
    )
    shutil_mod = "soccernet_detect.export_yolo.shutil.copy2"
    with patch(shutil_mod) as mock_copy:
        export_sequence_split(
            seq,
            "train",
            out_root,
            frame_stride=1,
            copy_images=True,
            prefer_gameinfo_ball=True,
            progress=False,
        )
        mock_copy.assert_not_called()


def _write_minimal_mot_sequence(
    seq_dir: Path,
    *,
    with_gameinfo: bool,
    gt_extra_ball: bool = True,
) -> None:
    seq_dir.mkdir(parents=True, exist_ok=True)
    (seq_dir / "img1").mkdir(parents=True, exist_ok=True)
    (seq_dir / "gt").mkdir(parents=True, exist_ok=True)
    (seq_dir / "img1" / "000001.jpg").touch()

    (seq_dir / "seqinfo.ini").write_text(
        textwrap.dedent("""\
            [Sequence]
            name=FAKE
            imWidth=1000
            imHeight=800
            seqLength=10
            frameRate=25
            """),
        encoding="utf-8",
    )

    lines = [
        "1,1,100,200,50,100,1,-1,-1,-1\n",
    ]
    if gt_extra_ball:
        lines.append("1,2,10,10,5,5,1,-1,-1,-1\n")  # small ball box
    lines.append("1,3,500,100,40,90,1,-1,-1,-1\n")  # referee row if gameinfo

    (seq_dir / "gt" / "gt.txt").write_text("".join(lines), encoding="utf-8")

    if with_gameinfo:
        (seq_dir / "gameinfo.ini").write_text(
            textwrap.dedent("""\
                [Sequence]
                trackletID_1= player team left;10
                trackletID_2= ball;1
                trackletID_3= referee;main
                """),
            encoding="utf-8",
        )


def test_export_sequence_split_filters_ball_and_ref(tmp_path: Path) -> None:
    seq = tmp_path / "SNMOT-fake"
    _write_minimal_mot_sequence(seq, with_gameinfo=True)

    out_root = tmp_path / "yolo"
    n = export_sequence_split(
        seq,
        "train",
        out_root,
        frame_stride=1,
        copy_images=True,
        prefer_gameinfo_ball=True,
        progress=False,
    )
    assert n == 1
    lbl = (out_root / "labels" / "train" / "SNMOT-fake_000001.txt").read_text(encoding="utf-8")
    parts = lbl.strip().split()
    assert len(parts) == 5
    assert parts[0] == "0"
    assert float(parts[3]) > 0 and float(parts[4]) > 0


def test_export_dataset_writes_yaml(tmp_path: Path) -> None:
    tracking = tmp_path / "tracking" / "train"
    for name in ("SNMOT-a", "SNMOT-b", "SNMOT-c"):
        _write_minimal_mot_sequence(tracking / name, with_gameinfo=True)

    out = tmp_path / "dataset"
    yaml_path = export_dataset(
        tmp_path / "tracking",
        "train",
        out,
        val_fraction=0.34,
        seed=0,
        frame_stride=1,
        copy_images=True,
        progress=False,
    )

    assert yaml_path.is_file()
    text = yaml_path.read_text(encoding="utf-8")
    assert "player" in text
    assert (out / "images" / "train").is_dir()
    assert (out / "images" / "val").is_dir()
