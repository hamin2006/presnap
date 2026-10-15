"""Sequence tensors for the coverage model (Phase 1C). numpy only, so it runs and tests without torch.

One play = T frames (frame 0 = snap) x P player slots x F features, plus a situation vector and labels.
Player features: x_rel/10, y_rel/10, vx/5, vy/5, is_def, position one-hot.

`max_def` keeps only the N deepest defenders at the snap. BDB 2026 doesn't track rushers, so the number of
defender slots in use equals 11 - rushers; capping at 5 (every play has >= 5) removes that count as a signal.
"""
import numpy as np
import polars as pl

from presnap import features, io, labels
from presnap.shell import KEY

POS = ["CB", "S", "LB", "DL", "OL", "QB", "RB", "WR", "TE"]  # anything else -> OTHER
F = 4 + 1 + len(POS) + 1
T_MAX = 40  # 4.0 s after the snap


def situation(plays: pl.DataFrame) -> np.ndarray:
    p = features.personnel_counts(plays)
    cols = [(pl.col("down") == d).cast(pl.Float32) for d in (1, 2, 3, 4)] + [
        pl.col("ydstogo") / 10, pl.col("yardline_100") / 100, pl.col("qtr") / 4, pl.col("half_seconds_remaining") / 1800,
        (pl.col("score_differential") / 14).clip(-2, 2), pl.col("shotgun"), pl.col("no_huddle"), pl.col("defenders_in_box") / 8,
        pl.col("ftn_is_motion").cast(pl.Float32), pl.col("ftn_is_motion").is_null().cast(pl.Float32)] + [
        pl.col(c) / 5 for c in ("off_rb", "off_te", "off_wr", "def_dl", "def_lb", "def_db")]
    return p.select([c.cast(pl.Float32).fill_null(0).alias(f"s{i}") for i, c in enumerate(cols)]).to_numpy()


def build(source="bdb26", max_def=None, t_max=T_MAX, t_min=0):
    """Frames t_min..t_max (frame 0 = snap; StatsBomb has frames before the snap). Index = frame - t_min."""
    plays = pl.read_parquet(io.PROCESSED / "plays.parquet").filter(pl.col("source") == source)
    if source == "statsbomb":
        plays = plays.filter(pl.col("sb_quality_ok"))
    plays = plays.sort(KEY)
    t = (pl.scan_parquet(io.PROCESSED / "tracking.parquet")
         .filter((pl.col("source") == source) & pl.col("frame").is_between(t_min, t_max))
         .join(plays.lazy().select(KEY), on=KEY).collect())

    snap = t.filter(pl.col("frame") == 0)
    d = snap.filter(pl.col("side") == "def").sort("x_rel", descending=True).with_columns(
        rank=pl.int_range(pl.len()).over(KEY))
    if max_def is not None:
        d = d.filter(pl.col("rank") < max_def)
    o = snap.filter(pl.col("side") == "off").with_columns(rank=pl.int_range(pl.len()).over(KEY) + 100)
    slots = pl.concat([d, o]).select(KEY + ["nfl_id", "rank"]).sort(KEY + ["rank"]).with_columns(
        slot=pl.int_range(pl.len()).over(KEY))
    P = int(slots["slot"].max()) + 1

    idx = plays.select(KEY).with_row_index("pi")
    t = t.join(slots, on=KEY + ["nfl_id"]).join(idx, on=KEY)
    T = t_max - t_min + 1
    X = np.zeros((plays.height, T, P, F), np.float32)
    M = np.zeros((plays.height, T, P), bool)
    pos = t["pos_group"].replace_strict({p: i for i, p in enumerate(POS)}, default=len(POS)).to_numpy()
    pi, fr, sl = t["pi"].to_numpy(), t["frame"].to_numpy() - t_min, t["slot"].to_numpy()
    for j, (c, s) in enumerate([("x_rel", 10), ("y_rel", 10), ("vx", 5), ("vy", 5)]):
        X[pi, fr, sl, j] = np.nan_to_num(t[c].to_numpy() / s)
    X[pi, fr, sl, 4] = (t["side"] == "def").to_numpy()
    X[pi, fr, sl, 5 + pos] = 1
    M[pi, fr, sl] = True

    last = t.group_by("pi").agg(pl.col("frame").max()).sort("pi")
    L = np.zeros(plays.height, np.int64)
    L[last["pi"].to_numpy()] = last["frame"].to_numpy() - t_min

    def lab(col, classes):
        return plays[col].replace_strict({c: i for i, c in enumerate(classes)}, default=-1, return_dtype=pl.Int64).to_numpy()

    # blitz = 5+ pass rushers (NGS charting). Only on StatsBomb: in BDB 2026 the rushers aren't tracked, so a blitz
    # head trained there would learn to count missing defenders.
    y_blitz = (plays["n_pass_rushers"].fill_null(-1).to_numpy() >= 5).astype(np.int64) if source == "statsbomb" else np.full(plays.height, -1)
    if source == "statsbomb":
        y_blitz[plays["n_pass_rushers"].is_null().to_numpy()] = -1
    return dict(X=X, M=M, last=L, t0=np.array(-t_min), S=situation(plays), y_cov=lab("coverage", labels.COVERAGE7), y_blitz=y_blitz,
                y_man=lab("man_zone", ["ZONE", "MAN"]), y_shell=lab("shell_played", [0, 1, 2]),
                split=plays["split"].to_numpy(), game=plays["game_id"].to_numpy(),
                agree=plays["labels_agree"].fill_null(False).to_numpy() if "labels_agree" in plays.columns else None)
