"""Phase 1A: rule-based safety shell (MOFO / MOFC), validated without a film audit.

Rule, applied to the two deepest defenders inside |y_rel| <= `width` at one frame (normalized coords):
  0-high  = the deepest is shallower than `depth`
  2-high  = both are at least `depth` deep, within `gap` yards of each other in depth, on opposite sides of
            the ball and at least `sep` yards apart laterally (MOFO)
  1-high  = anything else: one clearly deepest player, or the two deepest on the same side (MOFC)
A single depth cutoff can't work: at the snap the second safety sits at a median 10.5 yd in Cover 4 and
8.5 yd in Cover 3, so the two deepest have to be compared with each other.

Tuning never uses the played coverage at the snap:
  * `width`, `gap`, `sep` (plus the post-snap depth) are tuned at +2.0 s (or the throw, if sooner) against
    the NGS played shell: rotation is done by then, but the safeties aren't chasing the ball yet.
    The exploration in scratch/phase1a_explore.py showed +2.0 s beats the throw frame (82% vs 78% on train).
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
POST_FRAME = 20  # +2.0 s


def shell_at(defenders: pl.DataFrame, depth: float, width: float, gap: float, sep: float) -> pl.DataFrame:
    """defenders: one frame per play (KEY + x_rel, y_rel). Returns KEY + shell (0/1/2) + the two deepest."""
    d = defenders.filter(pl.col("y_rel").abs() <= width).sort("x_rel", descending=True)
    g = d.group_by(KEY).agg(x1=pl.col("x_rel").first(), x2=pl.col("x_rel").get(1, null_on_oob=True),
                            y1=pl.col("y_rel").first(), y2=pl.col("y_rel").get(1, null_on_oob=True))
    split = ((pl.col("y1") > 1) & (pl.col("y2") < -1)) | ((pl.col("y1") < -1) & (pl.col("y2") > 1))
    two = (pl.col("x2") >= depth) & (pl.col("x1") - pl.col("x2") < gap) & ((pl.col("y1") - pl.col("y2")).abs() >= sep) & split
    return g.with_columns(shell=pl.when(pl.col("x1") < depth).then(0).when(two).then(2).otherwise(1).cast(pl.Int8))


def frames(tracking: pl.LazyFrame, plays: pl.DataFrame):
    """Defender rows at the snap (all sources) and at +2.0 s or the throw if sooner (BDB 2026 only)."""
    d = tracking.filter((pl.col("side") == "def") & (pl.col("pos_group") != "DL"))
    snap = d.filter(pl.col("frame") == 0).select(KEY + ["nfl_id", "position", "x_rel", "y_rel"]).collect()
    b = d.filter(pl.col("source") == "bdb26")
    last = b.group_by(KEY).agg(pl.col("frame").max().alias("last"))
    throw = b.join(last, on=KEY).filter(pl.col("frame") == pl.min_horizontal("last", pl.lit(POST_FRAME))).select(
        KEY + ["x_rel", "y_rel", "last"]).collect()
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

    # (a) tune width / gap / sep / post-snap depth at +2.0 s, on train
    grid = list(itertools.product([8, 9, 10, 11, 12], [12, 14, 16, 18], [3, 4, 5, 6, 8], [6, 8, 10, 12]))
    scores = [(accuracy(shell_at(throw, d, w, g, s), plays, ["train"])[0], d, w, g, s) for d, w, g, s in grid]
    _, d_throw, width, gap, sep = max(scores)
    acc_a, n_a = accuracy(shell_at(throw, d_throw, width, gap, sep), plays, ["val", "test"])

    # (b) tune snap depth per safety, on train
    s_scores = [(safety_check(snap, labels, d, train_g)[0], d) for d in [7, 8, 9, 10, 11, 12, 13]]
    _, d_snap = max(s_scores)
    acc_b, n_b = safety_check(snap, labels, d_snap, eval_g)

    # shells at the snap (all sources) and the throw
    s_snap = shell_at(snap, d_snap, width, gap, sep).select(KEY, shell_snap="shell", x1_snap="x1", x2_snap="x2", y1_snap="y1", y2_snap="y2")
    s_throw = shell_at(throw, d_throw, width, gap, sep).select(KEY, shell_throw="shell")
    out = s_snap.join(s_throw, on=KEY, how="left")
    out.write_parquet(P / "shell.parquet")

    # (c) league-level sanity on 2023 BDB dropbacks
    b = out.filter(source="bdb26").join(plays.select(KEY + ["shell_played"]), on=KEY)
    two_high_snap = (b["shell_snap"] == 2).mean()
    rotated = b.filter(pl.col("shell_throw").is_not_null())
    rotation = (rotated["shell_snap"] != rotated["shell_throw"]).mean()
    snap_vs_played = (b.drop_nulls("shell_played")["shell_snap"] == b.drop_nulls("shell_played")["shell_played"]).mean()
    sb = out.filter(source="statsbomb").join(plays.filter(pl.col("sb_quality_ok")).select(KEY + ["shell_played"]), on=KEY)

    fam = (s_throw.join(plays.select(KEY + ["shell_played", "split", "coverage"]), on=KEY).filter(pl.col("split") != "train")
           .drop_nulls("shell_played").group_by("coverage").agg(n=pl.len(), acc=(pl.col("shell_throw") == pl.col("shell_played")).mean().round(3))
           .sort("n", descending=True))
    res = dict(params=dict(depth_snap=d_snap, depth_post=d_throw, width=width, gap=gap, sep=sep),
               a_by_coverage=fam.rows(),
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
