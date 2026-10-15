"""Pull the free nflverse tables we use and save them as parquet in data/raw/.
Then print, for each season, how many plays have coverage labels."""
import nflreadpy as nfl
import polars as pl
from pathlib import Path

RAW = Path(__file__).parent / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
SEASONS = list(range(2016, 2026))

tables = {
    "pbp": lambda: nfl.load_pbp(SEASONS),
    "participation": lambda: nfl.load_participation(True),  # every season nflverse has published
    "ftn_charting": lambda: nfl.load_ftn_charting(True),
    "nextgen_passing": lambda: nfl.load_nextgen_stats(SEASONS, "passing"),
    "rosters_weekly": lambda: nfl.load_rosters_weekly(SEASONS),
    "players": nfl.load_players,
}

for name, load in tables.items():
    df = load()
    df.write_parquet(RAW / f"{name}.parquet")
    print(f"{name}: {df.height:,} rows, {df.width} cols")

p = pl.read_parquet(RAW / "participation.parquet")
season = pl.col("nflverse_game_id").str.slice(0, 4).alias("season")
cov_cols = [c for c in ("defense_coverage_type", "defense_man_zone_type", "offense_formation",
                        "offense_personnel", "defense_personnel", "defenders_in_box",
                        "number_of_pass_rushers", "route", "time_to_throw") if c in p.columns]
print("\nShare of plays with each field filled in, by season:")
print(p.group_by(season).agg([pl.col(c).is_not_null().mean().round(2) for c in cov_cols]).sort("season"))
