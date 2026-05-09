"""Loader for SportsMOT clips.

SportsMOT ships in MOT17 format::

    seq/
      gt/gt.txt           # frame, id, x, y, w, h, conf, cls, vis
      img1/000001.jpg ... # rgb frames
      seqinfo.ini         # contains imWidth / imHeight / frameRate

We convert each clip to a ``Sequence`` whose player array is dense over time
with a per-(player, frame) validity mask. Coordinates use the bottom-center of
the bounding box (foot point), normalized to ``[0, 1]`` by ``imWidth`` and
``imHeight``.

Note (important for the project):
  SportsMOT does **not** annotate the ball (only the ``person`` class). The
  loader therefore sets ``Sequence.ball = None``. To run ball-conditioned
  experiments end-to-end, pair this loader with a separate ball source
  (synthetic, an external ball detector, or a different dataset like NBA
  SportVU / SoccerNet-Tracking).
"""

from __future__ import annotations

import configparser
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .types import Sequence

SPORT_FROM_PREFIX = {"v_": "volleyball", "b_": "basketball", "f_": "soccer"}


def _infer_sport(seq_name: str) -> str:
    """SportsMOT uses prefixes that map to the sport in some splits."""
    for prefix, sport in SPORT_FROM_PREFIX.items():
        if seq_name.lower().startswith(prefix):
            return sport
    return "unknown"


def _read_seqinfo(path: Path) -> dict:
    cfg = configparser.ConfigParser()
    cfg.read(path)
    s = cfg["Sequence"]
    return {
        "name": s.get("name", path.parent.name),
        "imWidth": int(s.get("imWidth")),
        "imHeight": int(s.get("imHeight")),
        "seqLength": int(s.get("seqLength")),
        "frameRate": float(s.get("frameRate", 25.0)),
    }


def _read_gt(path: Path) -> pd.DataFrame:
    cols = ["frame", "id", "x", "y", "w", "h", "conf", "cls", "vis"]
    df = pd.read_csv(path, header=None, names=cols)
    return df


def load_sportsmot_sequence(seq_dir: Path, sport_override: str | None = None) -> Sequence:
    """Load one SportsMOT clip directory."""
    seq_dir = Path(seq_dir)
    info = _read_seqinfo(seq_dir / "seqinfo.ini")
    gt = _read_gt(seq_dir / "gt" / "gt.txt")

    T = info["seqLength"]
    W, H = info["imWidth"], info["imHeight"]
    track_ids = sorted(gt["id"].unique())
    id_to_col = {tid: i for i, tid in enumerate(track_ids)}
    N = len(track_ids)

    players = np.zeros((T, N, 2), dtype=np.float32)
    mask = np.zeros((T, N), dtype=bool)

    for row in gt.itertuples(index=False):
        f = int(row.frame) - 1  # MOT frames are 1-indexed
        if not (0 <= f < T):
            continue
        col = id_to_col[row.id]
        cx = row.x + row.w / 2.0
        cy = row.y + row.h  # foot point = bottom-center of bbox
        players[f, col, 0] = cx / W
        players[f, col, 1] = cy / H
        mask[f, col] = True

    sport = sport_override or _infer_sport(info["name"])

    return Sequence(
        sequence_id=info["name"],
        sport=sport,
        fps=info["frameRate"],
        players=players,
        player_mask=mask,
        ball=None,
        ball_mask=None,
        scale=np.array([W, H], dtype=np.float32),
        player_team=None,
        meta={
            "source": "sportsmot",
            "track_ids": list(map(int, track_ids)),
            "path": str(seq_dir),
        },
    )


def load_sportsmot_split(split_root: Path, limit: int | None = None) -> list[Sequence]:
    """Load all clips in a split directory (e.g. ``dataset/train``)."""
    split_root = Path(split_root)
    seq_dirs: Iterable[Path] = sorted(p for p in split_root.iterdir() if p.is_dir())
    out: list[Sequence] = []
    for i, d in enumerate(seq_dirs):
        if not (d / "seqinfo.ini").exists():
            continue
        out.append(load_sportsmot_sequence(d))
        if limit is not None and len(out) >= limit:
            break
    return out
