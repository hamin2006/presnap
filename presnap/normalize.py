"""Put every source in one frame: 10 Hz, frame 0 = snap, offense moving toward +x.

x_rel: yards past the line of scrimmage (offense < 0, defense > 0)
y_rel: yards from the middle of the field, + = offense's left (L hash is about +3.1, R hash about -3.1)

Verified against FTN starting_hash (see tests): BDB y grows toward the offense's left on plays going right,
StatsBomb y is flipped relative to BDB. StatsBomb x runs goal line to goal line (LOS = yardline when the
offense goes left to right, 100 - yardline otherwise); BDB x includes end zones and LOS = absolute_yardline_number.
"""
import gzip
import json
from multiprocessing import Pool

import polars as pl

from presnap.io import RAW

MID = 53.3 / 2
HZ = 10

POS_GROUP = {
    "CB": "CB", "DB": "DB", "FS": "S", "SS": "S", "S": "S", "SAF": "S",
    "LB": "LB", "ILB": "LB", "MLB": "LB", "OLB": "LB",
    "DE": "DL", "DT": "DL", "NT": "DL", "DL": "DL",
    "T": "OL", "G": "OL", "C": "OL", "OT": "OL", "OG": "OL", "OL": "OL",
    "QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE",
}


def pos_group(col: str) -> pl.Expr:
    return pl.col(col).replace_strict(POS_GROUP, default="OTHER")


def to_rel(x_field: pl.Expr, y_field: pl.Expr, los: pl.Expr, toward_plus_x: pl.Expr, y_flipped: bool):
    """x_field/y_field/los in the source's yard frame; toward_plus_x = offense moving to +x."""
    sx = pl.when(toward_plus_x).then(1.0).otherwise(-1.0)
    sy = -sx if y_flipped else sx
    return ((x_field - los) * sx).alias("x_rel"), ((y_field - MID) * sy).alias("y_rel")


def add_velocity(df: pl.DataFrame) -> pl.DataFrame:
    """Central-difference velocity in the normalized frame (yd/s); one-sided at track ends and gaps."""
    k = ["source", "game_id", "play_id", "nfl_id"]
    df = df.sort(k + ["frame"])
    out = []
    for c in ("x_rel", "y_rel"):
        nxt, prv = pl.col(c).shift(-1).over(k), pl.col(c).shift(1).over(k)
        fn, fp = pl.col("frame").shift(-1).over(k), pl.col("frame").shift(1).over(k)
        hi, lo = pl.coalesce(nxt, pl.col(c)), pl.coalesce(prv, pl.col(c))
        dt = (pl.coalesce(fn, pl.col("frame")) - pl.coalesce(fp, pl.col("frame"))) / HZ
        out.append(pl.when(dt > 0).then((hi - lo) / dt).otherwise(None).alias("v" + c[0]))
    return df.with_columns(out)


def bdb26_tracking(inp: pl.DataFrame) -> pl.DataFrame:
    x, y = to_rel(pl.col("x"), pl.col("y"), pl.col("absolute_yardline_number"),
                  pl.col("play_direction") == "right", y_flipped=False)
    return inp.select(
        source=pl.lit("bdb26"), game_id="game_id", play_id="play_id", nfl_id="nfl_id",
        frame=pl.col("frame_id") - 1,  # frame 1 is the snap (audit: 9,510/9,519 tracks)
        side=pl.when(pl.col("player_side") == "Offense").then(pl.lit("off")).otherwise(pl.lit("def")),
        position="player_position", role="player_role", x_rel=x, y_rel=y,
    ).with_columns(pos_group=pos_group("position"))


def _sb_game(path):
    """One StatsBomb game -> (play rows, 10 Hz step rows, track rows)."""
    d = json.load(gzip.open(path))
    gid = d["nfl_old_game_id"]
    plays, steps, tracks = [], [], []
    for p in d["plays"]:
        pid = p.get("gsis_play_id")
        if pid is None or p.get("play_type") not in ("Pass", "Rush"):
            continue
        ltr, off = p["offense_left_to_right"], p["play_offense_team_id"]
        los = p["play_yardline"] if ltr else 100 - p["play_yardline"]
        tss = [s["time_since_snap"] for t in p["tracks"] for s in t["steps"] if s.get("time_since_snap") is not None]
        plays.append(dict(game_id=gid, play_id=pid, sb_presnap_s=-min(tss) if tss else None))
        for t in p["tracks"]:
            pl_ = t.get("player") or {}
            nid = pl_.get("gsis_player_id")  # this is the NFL nfl_id, not a GSIS id
            side = "off" if t["team_id"] == off else "def"
            tracks.append(dict(game_id=gid, play_id=pid, nfl_id=nid, side=side, sb_position=pl_.get("position_code"),
                               on_camera_ratio=t.get("on_camera_ratio")))
            if nid is None:
                continue
            for s in t["steps"]:
                ts, x = s.get("time_since_snap"), s.get("x")
                if ts is None or x is None:
                    continue
                f = round(ts * HZ)
                if abs(ts - f / HZ) < 0.5 / 30 + 1e-9:  # nearest 30 Hz step to each 10 Hz tick
                    steps.append((gid, pid, nid, f, side, x, s["y"], los, ltr))
    cols = ["game_id", "play_id", "nfl_id", "frame", "side", "x", "y", "los", "ltr"]
    return plays, pl.DataFrame(steps, schema=cols, orient="row"), tracks


def statsbomb():
    files = sorted((RAW / "statsbomb/tracking").glob("*.json.gz"))
    with Pool(8) as pool:
        res = pool.map(_sb_game, files)
    plays = pl.DataFrame([r for a, _, _ in res for r in a])
    tracks = pl.DataFrame([r for _, _, c in res for r in c], infer_schema_length=None)
    raw = pl.concat([b for _, b, _ in res])
    # 14 plays appear twice in the source under one gsis_play_id (likely re-snaps): ambiguous, drop all copies
    dup = plays.filter(pl.struct("game_id", "play_id").is_duplicated()).select("game_id", "play_id").unique()
    plays, tracks, raw = (d.join(dup, on=["game_id", "play_id"], how="anti") for d in (plays, tracks, raw))
    x, y = to_rel(pl.col("x"), pl.col("y"), pl.col("los"), pl.col("ltr"), y_flipped=True)
    steps = raw.select(source=pl.lit("statsbomb"), game_id="game_id", play_id="play_id", nfl_id="nfl_id",
                       frame="frame", side="side", x_rel=x, y_rel=y).unique(["game_id", "play_id", "nfl_id", "frame"])
    return plays, steps, tracks
