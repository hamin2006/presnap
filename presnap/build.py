"""Build the Phase 0 modeling tables in data/processed/.

  plays.parquet          one row per play (BDB 2026 + StatsBomb): situation, labels, label_source, split
  tracking.parquet       one row per player per 10 Hz frame, normalized (see presnap/normalize.py)
  player_labels.parquet  BDB 2026 coverage defenders with any BDB 2027 per-player label

Run: uv run python -m presnap.build
"""
from pathlib import Path

import numpy as np
import polars as pl

from presnap import io, labels, normalize

SPLITS = Path(__file__).resolve().parents[1] / "splits/splits.csv"
SAFETY = ["FS", "SS", "S", "SAF"]


def situation():
    cols = ["old_game_id", "play_id", "season", "week", "season_type", "posteam", "defteam", "down", "ydstogo",
            "yardline_100", "qtr", "half_seconds_remaining", "score_differential", "shotgun", "no_huddle", "qb_dropback"]
    p = io.pbp(cols).rename({"old_game_id": "game_id", "posteam": "offense", "defteam": "defense"})
    part = io.participation().select(
        game_id="old_game_id", play_id="play_id", offense_formation="offense_formation", offense_personnel="offense_personnel",
        defense_personnel="defense_personnel", defenders_in_box="defenders_in_box", n_pass_rushers="number_of_pass_rushers",
        coverage_nflverse=labels.canon("defense_coverage_type"))
    f = io.ftn().select(game_id="old_game_id", play_id="play_id", ftn_n_blitzers="n_blitzers", ftn_n_pass_rushers="n_pass_rushers",
                        ftn_is_motion="is_motion", ftn_starting_hash="starting_hash")
    return p.join(part, on=["game_id", "play_id"], how="left").join(f, on=["game_id", "play_id"], how="left")


def bdb26_plays(inp):
    tracked = inp.select("game_id", "play_id").unique()
    s = io.bdb26_supp().join(tracked, on=["game_id", "play_id"]).select(
        "game_id", "play_id", "receiver_alignment", "play_action", "dropback_type",
        coverage=labels.canon("team_coverage_type"))
    return s.with_columns(source=pl.lit("bdb26"), label_source=pl.lit("ngs"))


def statsbomb_plays(sb_plays, sb_tracks, sb_steps):
    pos = io.players()
    t = sb_tracks.join(pos, on="nfl_id", how="left")
    at_snap = sb_steps.filter(pl.col("frame") == 0).select("game_id", "play_id", "nfl_id").with_columns(at_snap=pl.lit(True))
    saf = (t.filter((pl.col("side") == "def") & (pl.col("sb_position") == "DB") & pl.col("position").is_in(SAFETY))
           .join(at_snap, on=["game_id", "play_id", "nfl_id"], how="left")
           .group_by("game_id", "play_id").agg(sb_n_safeties=pl.len(), sb_safety_min_on_camera=pl.col("on_camera_ratio").min(),
                                               sb_safeties_at_snap=pl.col("at_snap").fill_null(False).all()))
    csv = io.statsbomb_plays_csv().select(
        "game_id", "play_id",
        sb_blitz=pl.col("play_blitz").str.to_lowercase() == "true",  # never False in the source: null means no blitz
        sb_simulated_pressure=pl.col("play_simulated_pressure").str.to_lowercase() == "true",
        sb_drop_eight=pl.col("play_drop_eight").str.to_lowercase() == "true")
    return (sb_plays.join(saf, on=["game_id", "play_id"], how="left").join(csv, on=["game_id", "play_id"], how="left")
            .with_columns(pl.col("sb_blitz").fill_null(False), source=pl.lit("statsbomb")))


def make_splits(plays):
    """BDB 2026 by week (1-12 / 13-15 / 16-18); StatsBomb by game (70/15/15, seed 0). Frozen once written."""
    if SPLITS.exists():
        return pl.read_csv(SPLITS)
    b = plays.filter(pl.col("source") == "bdb26").select("source", "game_id", "week").unique().with_columns(
        split=pl.when(pl.col("week") <= 12).then(pl.lit("train")).when(pl.col("week") <= 15).then(pl.lit("val")).otherwise(pl.lit("test")))
    g = plays.filter(pl.col("source") == "statsbomb").select("game_id").unique().sort("game_id")["game_id"].to_numpy()
    g = np.random.default_rng(0).permutation(g)
    n = len(g)
    s = pl.DataFrame({"source": "statsbomb", "game_id": g,
                      "split": ["train"] * round(n * .7) + ["val"] * round(n * .15) + ["test"] * (n - round(n * .7) - round(n * .15))})
    out = pl.concat([b.select("source", "game_id", "split"), s]).sort("source", "game_id")
    SPLITS.parent.mkdir(exist_ok=True)
    out.write_csv(SPLITS)
    return out


def player_labels(bdb_track):
    d = io.bdb27_player_play().filter(pl.col("team_abbr") != pl.col("possession_team")).select(
        "game_id", "play_id", "nfl_id", "lined_up_position", "coverage_assignment", "coverage_assignment_at_snap",
        # BDB 2027 "blitzing" = rushed the passer (true for ~95% of DL); a real blitz is a non-DL rusher
        rushed="blitzing")
    defenders = bdb_track.filter((pl.col("frame") == 0) & (pl.col("side") == "def")).select("game_id", "play_id", "nfl_id", "position")
    return defenders.join(d, on=["game_id", "play_id", "nfl_id"])


def main():
    io.PROCESSED.mkdir(parents=True, exist_ok=True)
    inp = io.bdb26_input()
    bdb_track = normalize.bdb26_tracking(inp)
    sb_plays, sb_steps, sb_tracks = normalize.statsbomb()

    pos = io.players()
    sb_track = sb_steps.join(pos, on="nfl_id", how="left").join(
        sb_tracks.select("game_id", "play_id", "nfl_id", "sb_position").unique(["game_id", "play_id", "nfl_id"]),
        on=["game_id", "play_id", "nfl_id"], how="left"
    ).with_columns(pl.coalesce("position", "sb_position").alias("position"), role=pl.lit(None, pl.Utf8)).with_columns(
        pos_group=normalize.pos_group("position")).drop("sb_position")
    tracking = normalize.add_velocity(pl.concat([bdb_track, sb_track.select(bdb_track.columns)]))

    plays = pl.concat([bdb26_plays(inp), statsbomb_plays(sb_plays, sb_tracks, sb_steps)], how="diagonal")
    plays = plays.join(situation(), on=["game_id", "play_id"], how="left")
    plays = plays.with_columns(
        # StatsBomb labels come from nflverse: NGS through 2022, FTN after (FTN never overlaps SB seasons)
        coverage=pl.coalesce("coverage", pl.when(pl.col("season") <= 2022).then("coverage_nflverse")),
        coverage_ftn=pl.when(pl.col("season") >= 2023).then("coverage_nflverse"),
    ).with_columns(label_source=pl.when(pl.col("coverage").is_not_null()).then(pl.lit("ngs")),
                   labels_agree=pl.col("coverage") == pl.col("coverage_ftn")).drop("coverage_nflverse")
    plays = labels.with_label_columns(plays)
    plays = plays.with_columns(sb_quality_ok=(pl.col("source") == "statsbomb") & (pl.col("qb_dropback") == 1)
                               & (pl.col("sb_presnap_s") > 1) & pl.col("coverage").is_not_null()
                               & (pl.col("sb_safety_min_on_camera") >= 0.9) & pl.col("sb_safeties_at_snap").fill_null(False))
    plays = plays.join(make_splits(plays), on=["source", "game_id"], how="left")

    plays.write_parquet(io.PROCESSED / "plays.parquet")
    tracking.write_parquet(io.PROCESSED / "tracking.parquet")
    player_labels(bdb_track).write_parquet(io.PROCESSED / "player_labels.parquet")
    print(plays.group_by("source", "split").len().sort("source", "split"))
    print("tracking rows", tracking.height)


if __name__ == "__main__":
    main()
