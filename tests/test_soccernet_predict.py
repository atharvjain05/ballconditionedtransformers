"""SoccerNet player filtering / team ids from gameinfo strings."""

from __future__ import annotations

import pandas as pd

from ballcond.data.soccernet import (
    _soccernet_is_predict_target,
    _soccernet_player_track_ids,
    _soccernet_team_for_class,
)


def test_predict_target_classes() -> None:
    assert _soccernet_is_predict_target("player team left") is True
    assert _soccernet_is_predict_target("goalkeeper team right") is True
    assert _soccernet_is_predict_target("goalkeepers team left") is True
    assert _soccernet_is_predict_target("ball") is False
    assert _soccernet_is_predict_target("referee") is False
    assert _soccernet_is_predict_target("weird") is False


def test_team_for_class() -> None:
    assert _soccernet_team_for_class("player team left;3") == 0
    assert _soccernet_team_for_class("goalkeeper team right;x") == 1
    assert _soccernet_team_for_class("referee;main") == 2


def test_player_track_ids_filters_ref_and_ball() -> None:
    rows = pd.DataFrame(
        {
            "frame": [1, 1, 1, 1],
            "id": [1, 14, 18, 99],
            "x": [0.0, 0.0, 0.0, 0.0],
            "y": [0.0, 0.0, 0.0, 0.0],
            "w": [10, 10, 5, 10],
            "h": [10, 10, 5, 10],
            "conf": [1, 1, 1, 1],
            "c1": [-1, -1, -1, -1],
            "c2": [-1, -1, -1, -1],
            "c3": [-1, -1, -1, -1],
        }
    )
    classes = {
        1: "player team left",
        14: "referee",
        18: "ball",
        99: "player team right",
    }
    ids = _soccernet_player_track_ids(rows, classes)
    assert ids == [1, 99]

    ids_all = _soccernet_player_track_ids(rows, {})
    assert set(ids_all) == {1, 14, 18, 99}
