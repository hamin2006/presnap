"""Label-target class counts on the BDB 2026 tracking plays, plus BDB 2027 sparse-tracking players with assignment labels."""
import sys; sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import *

key = ["game_id", "play_id"]
s = bdb26_supp().filter(pl.col("season") == 2023)
pa = part().select(game_id="old_game_id", play_id="play_id", ftn_cov="defense_coverage_type", rushers="number_of_pass_rushers")
pbp = pl.scan_parquet(RAW / "pbp.parquet").filter(pl.col("season") == 2023).select(game_id=pl.col("old_game_id").cast(pl.Int64), play_id=pl.col("play_id").cast(pl.Int64), ngid="game_id").collect()
ftn = pl.read_parquet(RAW / "ftn_charting.parquet").filter(pl.col("season") == 2023).select(ngid="nflverse_game_id", play_id=pl.col("nflverse_play_id").cast(pl.Int64), n_blitzers="n_blitzers")
x = s.join(pa, on=key, how="left").join(pbp, on=key, how="left").join(ftn, on=["ngid", "play_id"], how="left")
shell = (pl.when(pl.col("team_coverage_type") == "COVER_0_MAN").then(pl.lit("zero")).when(pl.col("team_coverage_type").is_in(["COVER_1_MAN", "COVER_3_ZONE"])).then(pl.lit("single-high"))
         .when(pl.col("team_coverage_type").is_in(["COVER_2_ZONE", "COVER_2_MAN", "COVER_4_ZONE", "COVER_6_ZONE"])).then(pl.lit("two-high")).otherwise(pl.col("team_coverage_type")))
print("shell (from NGS label):", x.group_by(shell.alias("shell")).len().sort("len", descending=True).to_dicts())
print("FTN rushers on bdb26 plays:", x["rushers"].value_counts().sort("rushers").to_dicts())
print("FTN blitz (n_blitzers>0):", x.group_by(pl.col("n_blitzers") > 0).len().to_dicts(), "n_blitzers:", x["n_blitzers"].value_counts().sort("n_blitzers").to_dicts())
print("FTN coverage on bdb26 plays:", x["ftn_cov"].value_counts().sort("count", descending=True).to_dicts())

t = bdb27_track().select(key + ["nfl_id"]).unique()
d = bdb27_pp().filter(pl.col("team_abbr") != pl.col("possession_team"))
tj = t.join(d.select(key + ["nfl_id", "coverage_assignment", "blitzing", "lined_up_position"]), on=key + ["nfl_id"], how="left")
print("\nbdb27 2023 tracked (play,player):", t.height, "defenders among them:", tj["lined_up_position"].is_not_null().sum(),
      "with coverage_assignment:", tj["coverage_assignment"].is_not_null().sum(), "with blitzing not null:", tj["blitzing"].is_not_null().sum(), "blitzing True:", tj["blitzing"].sum())
print("their assignment values", tj["coverage_assignment"].value_counts().sort("count", descending=True).to_dicts())
print("bdb27 2023 player_play defender rows with coverage_assignment:", d.join(pl.read_csv(RAW / "bdb2027/games.csv").filter(pl.col("season") == 2023).select("game_id"), on="game_id")["coverage_assignment"].is_not_null().sum())
