"""Per-play features for the baselines (Phase 1B).

Leakage rules:
  * Only information available at the frame used. Situation features are pre-snap charting only
    (no play_action, dropback_type, pass rushers or anything else decided after the snap).
  * BDB 2026 tracks only defenders who dropped into coverage, so rushers are missing from the data. Never use the
    count of tracked defenders (it equals 11 - rushers on 88% of plays). Depth-ranked features use the 5 deepest
    defenders only, which every play has.
"""
import json

import polars as pl

from presnap import io
from presnap.shell import KEY, shell_at

SITUATION_NUM = ["down", "ydstogo", "yardline_100", "qtr", "half_seconds_remaining", "score_differential",
                 "shotgun", "no_huddle", "defenders_in_box", "ftn_is_motion",
                 "off_rb", "off_te", "off_wr", "def_dl", "def_lb", "def_db"]
SITUATION_CAT = ["offense_formation", "receiver_alignment"]
PERSONNEL = {"off_rb": ("offense_personnel", ["RB", "FB"]), "off_te": ("offense_personnel", ["TE"]),
             "off_wr": ("offense_personnel", ["WR"]), "def_dl": ("defense_personnel", ["DL", "DE", "DT", "NT"]),
             "def_lb": ("defense_personnel", ["LB", "ILB", "MLB", "OLB"]),
             "def_db": ("defense_personnel", ["DB", "CB", "S", "FS", "SS"])}


def personnel_counts(df: pl.DataFrame) -> pl.DataFrame:
    """nflverse personnel strings ("1 RB, 2 TE, 2 WR" or full position lists) -> counts per group."""
    def count(col, pos):
        return pl.sum_horizontal([pl.col(col).str.extract(rf"(\d+) {p}(?:,|$)").cast(pl.Int8).fill_null(0) for p in pos])
    return df.with_columns([count(col, pos).alias(name) for name, (col, pos) in PERSONNEL.items()])
N_DEEP = 5


def shell_params():
    p = json.loads((io.ROOT / "reports/phase1a_checks.json").read_text())["params"]
    return p["depth_snap"], p["width"], p["gap"], p["sep"]


def frame_features(fr: pl.DataFrame, tag: str) -> pl.DataFrame:
    """fr: tracking rows for one frame per play. Returns one row per play, columns prefixed with `tag`."""
    d = fr.filter((pl.col("side") == "def") & (pl.col("pos_group") != "DL"))
    deep = d.sort("x_rel", descending=True).group_by(KEY).agg(
        [pl.col(c).get(i, null_on_oob=True).alias(f"d{i + 1}_{c}") for i in range(N_DEEP) for c in ("x_rel", "y_rel", "vx", "vy")]
        + [(pl.col("x_rel") >= k).sum().alias(f"n_ge{k}") for k in (8, 12)])

    depth, width, gap, sep = shell_params()
    sh = shell_at(d, depth, width, gap, sep).select(*KEY, "shell", "x1", "x2", "y1", "y2").with_columns(
        gap12=pl.col("x1") - pl.col("x2"), sep12=(pl.col("y1") - pl.col("y2")).abs())

    # two widest corners: depth, width, cushion and leverage vs the nearest receiver (not the QB)
    cb = d.filter(pl.col("position") == "CB").select(KEY + ["nfl_id", "x_rel", "y_rel"])
    rec = fr.filter((pl.col("side") == "off") & (pl.col("position") != "QB")).select(KEY + [pl.col("x_rel").alias("wx"), pl.col("y_rel").alias("wy")])
    m = (cb.join(rec, on=KEY).with_columns(ady=(pl.col("y_rel") - pl.col("wy")).abs()).sort("ady")
         .group_by(KEY + ["nfl_id"]).first()
         .with_columns(cushion=pl.col("x_rel") - pl.col("wx"),
                       lev=(pl.col("y_rel") - pl.col("wy")) * pl.col("wy").sign(),  # + = outside leverage
                       absy=pl.col("y_rel").abs()))
    cbf = m.sort("absy", descending=True).group_by(KEY).agg(
        [pl.col(c).get(i, null_on_oob=True).alias(f"cb{i + 1}_{c}") for i in range(2) for c in ("x_rel", "absy", "cushion", "lev")])

    out = deep.join(sh, on=KEY, how="left").join(cbf, on=KEY, how="left")
    return out.rename({c: f"{tag}_{c}" for c in out.columns if c not in KEY})


def build(source: str, frames: dict) -> pl.DataFrame:
    """frames: {tag: frame-selection} where a selection is an int frame or "last" (BDB only) or ("min_last", k)."""
    t = pl.scan_parquet(io.PROCESSED / "tracking.parquet").filter(pl.col("source") == source)
    last = t.group_by(KEY).agg(pl.col("frame").max().alias("last"))
    out = None
    for tag, sel in frames.items():
        if sel == "last":
            cond = pl.col("frame") == pl.col("last")
        elif isinstance(sel, tuple):
            cond = pl.col("frame") == pl.min_horizontal("last", pl.lit(sel[1]))
        else:
            cond = pl.col("frame") == sel
        fr = t.join(last, on=KEY).filter(cond).collect()
        f = frame_features(fr, tag)
        out = f if out is None else out.join(f, on=KEY, how="left")
    return out
