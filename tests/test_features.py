import polars as pl

from presnap.features import frame_features


def test_cb_leverage_and_depth_ranking():
    base = dict(source="t", game_id=1, play_id=1, vx=0.0, vy=0.0)
    rows = [
        dict(base, nfl_id=1, side="off", position="WR", pos_group="WR", x_rel=-1.0, y_rel=20.0),
        dict(base, nfl_id=2, side="off", position="QB", pos_group="QB", x_rel=-5.0, y_rel=0.0),
        dict(base, nfl_id=3, side="def", position="CB", pos_group="CB", x_rel=6.0, y_rel=22.0),   # outside the WR
        dict(base, nfl_id=4, side="def", position="FS", pos_group="S", x_rel=14.0, y_rel=0.0),
        dict(base, nfl_id=5, side="def", position="SS", pos_group="S", x_rel=9.0, y_rel=-8.0),
    ]
    f = frame_features(pl.DataFrame(rows), "snap").row(0, named=True)
    assert f["snap_d1_x_rel"] == 14.0 and f["snap_d2_x_rel"] == 9.0
    assert f["snap_cb1_lev"] == 2.0          # outside leverage is positive
    assert f["snap_cb1_cushion"] == 7.0
    assert f["snap_d4_x_rel"] is None        # fewer than 4 defenders -> null, never a roster count
