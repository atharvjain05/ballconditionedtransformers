"""Loader for SoccerNet-Tracking sequences.

SoccerNet-Tracking ships in MOT format (similar to SportsMOT)::

    seq_dir/
      gt/gt.txt           # frame, id, x, y, w, h, conf, -1, -1, -1
      img1/000001.jpg ... # rgb frames
      seqinfo.ini         # contains imWidth / imHeight / frameRate / seqLength
      gameinfo.ini        # optional: trackletID_<mot_id>= "<class>;<jersey>"

The class column in ``gt.txt`` is ``-1`` for all objects. If ``gameinfo.ini``
exists, it is always parsed for per-track class (excluding referees from player
targets and attaching team ids). Ball id(s): from ``gameinfo.ini`` every ``trackletID_*`` row with class ``ball``
     and present in ``gt.txt`` becomes a ball track. Multiple rows yield
     ``K > 1`` trajectories in ``Sequence.ball`` ``(T, K, 2)``. The area
     heuristic still returns a single track when gameinfo is unused.
"""

from __future__ import annotations

import configparser
import re
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .types import Sequence

_TRACKLET_KEY_RE = re.compile(r"^trackletID_(\d+)$", re.IGNORECASE)


def _soccernet_is_predict_target(cls: str) -> bool:
    """True for outfield players / goalkeepers; false for ball, refs, staff."""
    c = cls.strip().lower()
    if c == "ball":
        return False
    if c == "referee" or c.startswith("referee"):
        return False
    if "staff" in c:
        return False
    if c == "other":
        return False
    if "player team" in c:
        return True
    if "goalkeeper" in c or "goalkeepers" in c:
        return True
    return False


def _soccernet_team_for_class(cls: str) -> int:
    """0 = team left in frame, 1 = team right, 2 = unknown."""
    c = cls.strip().lower()
    if "team left" in c:
        return 0
    if "team right" in c:
        return 1
    return 2


def _soccernet_player_track_ids(
    player_rows: pd.DataFrame,
    tracklet_classes: dict[int, str],
) -> list[int]:
    """MOT ids used as prediction targets (excludes refs when gameinfo lists classes)."""
    raw_ids = sorted(int(x) for x in player_rows["id"].unique())
    if not tracklet_classes:
        return raw_ids
    out: list[int] = []
    for tid in raw_ids:
        cls = tracklet_classes.get(tid)
        if cls is None:
            out.append(tid)
            continue
        if _soccernet_is_predict_target(cls):
            out.append(tid)
    return out


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
    """Read MOT-format ground truth.  SoccerNet uses 10 columns."""
    cols = ["frame", "id", "x", "y", "w", "h", "conf", "c1", "c2", "c3"]
    df = pd.read_csv(path, header=None, names=cols[: _detect_ncols(path)])
    return df


def _detect_ncols(path: Path) -> int:
    with open(path) as f:
        first = f.readline().strip()
    return len(first.split(","))


def _parse_gameinfo_tracklets(path: Path) -> dict[int, str]:
    """Map MOT track id -> semantic class string (before first ``;``), lowercased."""
    cfg = configparser.ConfigParser(interpolation=None)
    cfg.optionxform = str
    if not cfg.read(path) or "Sequence" not in cfg:
        return {}
    out: dict[int, str] = {}
    for key, raw in cfg["Sequence"].items():
        m = _TRACKLET_KEY_RE.match(key.strip())
        if not m:
            continue
        tid = int(m.group(1))
        value = (raw or "").strip()
        cls = value.split(";", 1)[0].strip().lower()
        out[tid] = cls
    return out


def _ball_track_ids_from_gameinfo(path: Path) -> list[int]:
    """All MOT ids labeled ``ball`` in gameinfo (sorted)."""
    m = _parse_gameinfo_tracklets(path)
    return sorted(tid for tid, cls in m.items() if cls == "ball")


def _ball_id_area_heuristic(gt: pd.DataFrame, ball_area_threshold: float) -> int | None:
    median_area = gt.assign(area=gt["w"] * gt["h"]).groupby("id")["area"].median()
    ball_candidates = median_area[median_area < ball_area_threshold]
    return int(ball_candidates.idxmin()) if len(ball_candidates) > 0 else None


def load_soccernet_sequence(
    seq_dir: Path,
    ball_area_threshold: float = 2000.0,
    prefer_gameinfo_ball: bool = True,
) -> Sequence:
    """Load one SoccerNet-Tracking clip directory.

    Args:
        seq_dir: Path to a sequence directory (e.g. ``SNMOT-060``).
        ball_area_threshold: Max median bbox area (px²) for a track to be
            considered the ball when ``gameinfo.ini`` is missing or unusable.
        prefer_gameinfo_ball: If True and ``gameinfo.ini`` lists a ``ball``
            tracklet present in ``gt.txt``, use that MOT id instead of the area
            heuristic.
    """
    seq_dir = Path(seq_dir)
    info = _read_seqinfo(seq_dir / "seqinfo.ini")
    gt = _read_gt(seq_dir / "gt" / "gt.txt")

    T = info["seqLength"]
    W, H = info["imWidth"], info["imHeight"]

    gt_ids = set(gt["id"].astype(int).unique())
    ball_ids: list[int] = []
    ball_source: str | None = None
    tracklet_classes: dict[int, str] = {}

    gameinfo_path = seq_dir / "gameinfo.ini"
    if gameinfo_path.is_file():
        tracklet_classes = _parse_gameinfo_tracklets(gameinfo_path)

    if prefer_gameinfo_ball and gameinfo_path.is_file():
        gi_ball_list = _ball_track_ids_from_gameinfo(gameinfo_path)
        cand = [tid for tid in gi_ball_list if tid in gt_ids]
        if cand:
            ball_ids = cand
            ball_source = "gameinfo"
        elif gi_ball_list:
            missing = [tid for tid in gi_ball_list if tid not in gt_ids]
            warnings.warn(
                f"{seq_dir}: gameinfo ball ids not in gt.txt: {missing}; "
                "falling back to area heuristic",
                stacklevel=2,
            )

    if not ball_ids:
        ball_id_heur = _ball_id_area_heuristic(gt, ball_area_threshold)
        if ball_id_heur is not None:
            ball_ids = [ball_id_heur]
            ball_source = "area_heuristic"

    # --- rows: all players + balls (exclude every ball id from player tensor) ---
    ball_id_set = set(ball_ids)
    if ball_id_set:
        player_rows = gt[~gt["id"].isin(ball_id_set)]
    else:
        player_rows = gt

    # --- build player array (field players + goalkeepers only when gameinfo exists) ---
    player_ids = _soccernet_player_track_ids(player_rows, tracklet_classes)
    id_to_col = {tid: i for i, tid in enumerate(player_ids)}
    N = len(player_ids)

    team_vals: list[int] = []
    for tid in player_ids:
        cls = tracklet_classes.get(tid) if tracklet_classes else None
        team_vals.append(_soccernet_team_for_class(cls) if cls is not None else 2)
    player_team_arr = np.array(team_vals, dtype=np.int64) if N else np.zeros((0,), dtype=np.int64)

    players = np.zeros((T, N, 2), dtype=np.float32)
    player_mask = np.zeros((T, N), dtype=bool)

    for row in player_rows.itertuples(index=False):
        if int(row.id) not in id_to_col:
            continue
        f = int(row.frame) - 1  # MOT frames are 1-indexed
        if not (0 <= f < T):
            continue
        col = id_to_col[row.id]
        cx = row.x + row.w / 2.0
        cy = row.y + row.h  # foot point = bottom-center
        players[f, col, 0] = cx / W
        players[f, col, 1] = cy / H
        player_mask[f, col] = True

    # --- build ball array (T, K, 2) ---
    Kb = len(ball_ids)
    if Kb > 0:
        ball = np.zeros((T, Kb, 2), dtype=np.float32)
        ball_mask_arr = np.zeros((T, Kb), dtype=bool)
        for ki, bid in enumerate(ball_ids):
            for row in gt[gt["id"] == bid].itertuples(index=False):
                f = int(row.frame) - 1
                if not (0 <= f < T):
                    continue
                cx = row.x + row.w / 2.0
                cy = row.y + row.h / 2.0  # ball center (not foot point)
                ball[f, ki, 0] = cx / W
                ball[f, ki, 1] = cy / H
                ball_mask_arr[f, ki] = True
    else:
        ball = None
        ball_mask_arr = None

    return Sequence(
        sequence_id=info["name"],
        sport="soccer",
        fps=info["frameRate"],
        players=players,
        player_mask=player_mask,
        ball=ball,
        ball_mask=ball_mask_arr,
        scale=np.array([W, H], dtype=np.float32),
        player_team=player_team_arr if N > 0 else None,
        meta={
            "source": "soccernet",
            "ball_track_ids": list(map(int, ball_ids)),
            "ball_track_id": ball_ids[0] if ball_ids else None,
            "ball_track_source": ball_source,
            "tracklet_classes": tracklet_classes,
            "player_track_ids": list(map(int, player_ids)),
            "path": str(seq_dir),
        },
    )


def load_soccernet_split(
    split_root: Path,
    limit: int | None = None,
    ball_area_threshold: float = 2000.0,
    prefer_gameinfo_ball: bool = True,
) -> list[Sequence]:
    """Load all clips in a SoccerNet split directory."""
    split_root = Path(split_root)
    seq_dirs: Iterable[Path] = sorted(p for p in split_root.iterdir() if p.is_dir())
    out: list[Sequence] = []
    for d in seq_dirs:
        if not (d / "seqinfo.ini").exists():
            continue
        out.append(
            load_soccernet_sequence(
                d,
                ball_area_threshold=ball_area_threshold,
                prefer_gameinfo_ball=prefer_gameinfo_ball,
            )
        )
        if limit is not None and len(out) >= limit:
            break
    return out
