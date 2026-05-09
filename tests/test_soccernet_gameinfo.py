from __future__ import annotations

import textwrap
from pathlib import Path

from ballcond.data.soccernet import _ball_track_ids_from_gameinfo, _parse_gameinfo_tracklets


def test_parse_gameinfo_sample_format(tmp_path: Path) -> None:
    ini = tmp_path / "gameinfo.ini"
    ini.write_text(
        textwrap.dedent("""\
            [Sequence]
            name=SNMOT-060
            num_tracklets=3
            trackletID_1= player team left;10
            trackletID_18= ball;1
            trackletID_26= referee;side bottom
            """),
        encoding="utf-8",
    )
    m = _parse_gameinfo_tracklets(ini)
    assert m[1] == "player team left"
    assert m[18] == "ball"
    assert m[26] == "referee"
    assert _ball_track_ids_from_gameinfo(ini) == [18]


def test_ball_track_ids_from_gameinfo_trackletid_case_insensitive(tmp_path: Path) -> None:
    ini = tmp_path / "gameinfo.ini"
    ini.write_text(
        "[Sequence]\ntrackletID_5= Ball;x\n",
        encoding="utf-8",
    )
    assert _ball_track_ids_from_gameinfo(ini) == [5]


def test_ball_track_ids_from_gameinfo_empty_when_missing(tmp_path: Path) -> None:
    ini = tmp_path / "gameinfo.ini"
    ini.write_text(
        "[Sequence]\nname=x\ntrackletID_1= player team left;1\n",
        encoding="utf-8",
    )
    assert _ball_track_ids_from_gameinfo(ini) == []


def test_ball_track_ids_multiple_returns_sorted(tmp_path: Path) -> None:
    ini = tmp_path / "gameinfo.ini"
    ini.write_text(
        textwrap.dedent("""\
            [Sequence]
            trackletID_20= ball;1
            trackletID_5= ball;2
            """),
        encoding="utf-8",
    )
    assert _ball_track_ids_from_gameinfo(ini) == [5, 20]
