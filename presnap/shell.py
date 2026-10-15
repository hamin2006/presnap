"""Phase 1A: rule-based safety shell (MOFO / MOFC), validated without a film audit.

Rule, applied to the defenders at one frame (normalized coordinates, see normalize.py):
  deep    = defenders at least `depth` yards past the LOS and inside |y_rel| <= `width`
  0-high  = nobody deep
  1-high  = someone deep in the middle (|y_rel| < `mid`), or deep defenders on one side only (MOFC)
  2-high  = deep defenders on both sides and nobody deep in the middle (MOFO)

Tuning never uses the played coverage at the snap:
  * `mid` and `width` (plus the throw depth) are tuned at the THROW against the NGS played shell,
    when rotation is over and the shell shown equals the shell played.
  * the snap depth is tuned per safety against BDB 2027 `coverage_assignment_at_snap` (deep vs underneath).
All tuning uses the train split only; metrics are reported on val + test.

Run (on the PC): uv run python -m presnap.shell
"""
import itertools
import json
from pathlib import Path

import polars as pl

from presnap import io

ROOT = Path(__file__).resolve().parents[1]
KEY = ["source", "game_id", "play_id"]
DEEP_ASSIGN = ("DEEP_", "QUARTERS_", "PREVENT")


def shell_at(defenders: pl.DataFrame, depth: float, mid: float, width: float) -> pl.DataFrame:
    """defenders: one frame per play (KEY + x_rel, y_rel). Returns KEY + shell (0/1/2) + n_deep."""
    d = defenders.with_columns(deep=(pl.col("x_rel") >= depth) & (pl.col("y_rel").abs() <= width))
    g = d.group_by(KEY).agg(
        n_deep=pl.col("deep").sum(),
        mid_deep=(pl.col("deep") & (pl.col("y_rel").abs() < mid)).any(),
        left=(pl.col("deep") & (pl.col("y_rel") >= mid)).any(),
        right=(pl.col("deep") & (pl.col("y_rel") <= -mid)).any(),
    )
    return g.select(KEY + ["n_deep"], shell=pl.when(pl.col("n_deep") == 0).then(0)
                    .when(~pl.col("mid_deep") & pl.col("left") & pl.col("right")).then(2).otherwise(1).cast(pl.Int8))


def frames(tracking: pl.LazyFrame, plays: pl.DataFrame):
    """Defender rows at the snap (all sources) and at the throw (BDB 2026 only: its last input frame)."""
    d = tracking.filter((pl.col("side") == "def") & (pl.col("pos_group") != "DL"))
    snap = d.filter(pl.col("frame") == 0).select(KEY + ["nfl_id", "position", "x_rel", "y_rel"]).collect()
    b = d.filter(pl.col("source") == "bdb26")
    last = b.group_by(KEY).agg(pl.col("frame").max().alias("last"))
    throw = b.join(last, on=KEY).filter(pl.col("frame") == pl.col("last")).select(KEY + ["x_rel", "y_rel", "last"]).collect()
    return snap, throw


def accuracy(pred, plays, split):
    j = pred.join(plays.select(KEY + ["shell_played", "split"]), on=KEY).filter(
        pl.col("shell_played").is_not_null() & pl.col("split").is_in(split))
    return (j["shell"] == j["shell_played"]).mean(), j.height


def safety_check(snap, labels, depth, split_games):
    s = snap.filter(pl.col("position").is_in(["FS", "SS", "S"]) & (pl.col("source") == "bdb26")).join(
        labels, on=["game_id", "play_id", "nfl_id"]).join(split_games, on=["game_id"])
    s = s.with_columns(rule_deep=pl.col("x_rel") >= depth)
    return (s["rule_deep"] == s["ngs_deep"]).mean(), s.height


def main():
    P = io.PROCESSED
    plays = pl.read_parquet(P / "plays.parquet")
    snap, throw = frames(pl.scan_parquet(P / "tracking.parquet"), plays)
    labels = pl.read_parquet(P / "player_labels.parquet").filter(
        pl.col("coverage_assignment_at_snap").is_not_null() & (pl.col("coverage_assignment_at_snap") != "NO_ASSIGNMENT")
    ).select("game_id", "play_id", "nfl_id",
             ngs_deep=pl.col("coverage_assignment_at_snap").str.starts_with(DEEP_ASSIGN[0])
             | pl.col("coverage_assignment_at_snap").str.starts_with(DEEP_ASSIGN[1])
             | (pl.col("coverage_assignment_at_snap") == DEEP_ASSIGN[2]))
    games = plays.filter(pl.col("source") == "bdb26").select("game_id", "split").unique()
    train_g, eval_g = games.filter(split="train").select("game_id"), games.filter(pl.col("split") != "train").select("game_id")

    # (a) tune mid / width / throw depth at the throw, on train
    grid = list(itertools.product([10, 12, 14, 16, 18], [3, 4, 5, 6, 7, 8], [12, 14, 16, 18, 20]))
    scores = [(accuracy(shell_at(throw, d, m, w), plays, ["train"])[0], d, m, w) for d, m, w in grid]
    _, d_throw, mid, width = max(scores)
    acc_a, n_a = accuracy(shell_at(throw, d_throw, mid, width), plays, ["val", "test"])

    # (b) tune snap depth per safety, on train
    s_scores = [(safety_check(snap, labels, d, train_g)[0], d) for d in [7, 8, 9, 10, 11, 12, 13]]
    _, d_snap = max(s_scores)
    acc_b, n_b = safety_check(snap, labels, d_snap, eval_g)

    # shells at the snap (all sources) and the throw
    s_snap = shell_at(snap, d_snap, mid, width).rename({"shell": "shell_snap", "n_deep": "n_deep_snap"})
    s_throw = shell_at(throw, d_throw, mid, width).rename({"shell": "shell_throw", "n_deep": "n_deep_throw"})
    out = s_snap.join(s_throw, on=KEY, how="left")
    out.write_parquet(P / "shell.parquet")

    # (c) league-level sanity on 2023 BDB dropbacks
    b = out.filter(source="bdb26").join(plays.select(KEY + ["shell_played"]), on=KEY)
    two_high_snap = (b["shell_snap"] == 2).mean()
    rotated = b.filter(pl.col("shell_throw").is_not_null())
    rotation = (rotated["shell_snap"] != rotated["shell_throw"]).mean()
    snap_vs_played = (b.drop_nulls("shell_played")["shell_snap"] == b.drop_nulls("shell_played")["shell_played"]).mean()
    sb = out.filter(source="statsbomb").join(plays.filter(pl.col("sb_quality_ok")).select(KEY + ["shell_played"]), on=KEY)

    res = dict(params=dict(depth_snap=d_snap, depth_throw=d_throw, mid=mid, width=width),
               a_throw_vs_played=dict(acc=round(acc_a, 4), n=n_a),
               b_safety_deep_at_snap=dict(acc=round(acc_b, 4), n=n_b),
               c_league=dict(two_high_at_snap=round(two_high_snap, 4), rotation_snap_to_throw=round(rotation, 4),
                             snap_equals_played=round(snap_vs_played, 4),
                             statsbomb_two_high_at_snap=round((sb["shell_snap"] == 2).mean(), 4) if sb.height else None,
                             statsbomb_snap_equals_played=round((sb["shell_snap"] == sb["shell_played"]).mean(), 4) if sb.height else None),
               confusion_throw=(s_throw.join(plays.select(KEY + ["shell_played", "split"]), on=KEY)
                                .filter(pl.col("split") != "train").drop_nulls("shell_played")
                                .group_by("shell_played", "shell_throw").len().sort("shell_played", "shell_throw").rows()))
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/phase1a_checks.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
