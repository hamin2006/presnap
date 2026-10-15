"""Raw-data loaders. Big CSVs are cached as parquet in data/interim/ on first read."""
from pathlib import Path
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
INTERIM = ROOT / "data/interim"
PROCESSED = ROOT / "data/processed"


def _cached(name, build):
    p = INTERIM / f"{name}.parquet"
    if not p.exists():
        INTERIM.mkdir(parents=True, exist_ok=True)
        build().write_parquet(p)
    return pl.read_parquet(p)


def bdb26_input():
    return _cached("bdb26_input", lambda: pl.read_csv(
        sorted((RAW / "bdb2026/train").glob("input_*.csv")), null_values=["NA"], infer_schema_length=100000))


def bdb26_supp():
    return pl.read_csv(RAW / "bdb2026/supplementary_data.csv", null_values=["NA", ""], infer_schema_length=100000)


def bdb27_player_play():
    return _cached("bdb27_player_play", lambda: pl.read_csv(
        RAW / "bdb2027/player_play.csv", null_values=["NA"], infer_schema_length=200000))


def participation():
    # old_game_id is str and play_id is f64 in the raw file
    return pl.read_parquet(RAW / "participation.parquet").with_columns(
        pl.col("old_game_id").cast(pl.Int64, strict=False), pl.col("play_id").cast(pl.Int64))


def pbp(columns):
    return pl.read_parquet(RAW / "pbp.parquet", columns=columns).with_columns(
        pl.col("old_game_id").cast(pl.Int64, strict=False), pl.col("play_id").cast(pl.Int64))


def ftn():
    """FTN charting keyed by old_game_id + play_id (FTN only carries nflverse string ids)."""
    keys = pbp(["game_id", "old_game_id", "play_id"])
    f = pl.read_parquet(RAW / "ftn_charting.parquet").with_columns(play_id=pl.col("nflverse_play_id").cast(pl.Int64))
    return f.join(keys, left_on=["nflverse_game_id", "play_id"], right_on=["game_id", "play_id"]).drop("season")


def players():
    return pl.read_parquet(RAW / "players.parquet").select(
        nfl_id=pl.col("nfl_id").cast(pl.Int64, strict=False), position="position").drop_nulls("nfl_id").unique("nfl_id")


def statsbomb_plays_csv():
    S = pl.concat([pl.read_csv(f, infer_schema_length=0) for f in sorted((RAW / "statsbomb/plays").glob("*.csv"))], how="diagonal")
    # ids are text like "41.0" with some "NA"
    return S.with_columns(
        game_id=pl.col("gsis_old_game_id").cast(pl.Float64, strict=False).cast(pl.Int64, strict=False),
        play_id=pl.col("gsis_play_id").cast(pl.Float64, strict=False).cast(pl.Int64, strict=False),
    ).drop_nulls(["game_id", "play_id"]).unique(["game_id", "play_id"])
