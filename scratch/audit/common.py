"""Shared loaders for the data audit. Caches big CSVs as parquet in scratch/audit/cache."""
from pathlib import Path
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw"
CACHE = Path(__file__).parent / "cache"
CACHE.mkdir(exist_ok=True)
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_cols(30); pl.Config.set_fmt_str_lengths(60); pl.Config.set_tbl_width_chars(200)


def _cached(name, build):
    p = CACHE / f"{name}.parquet"
    if not p.exists():
        build().write_parquet(p)
    return pl.read_parquet(p)


def bdb26_input():
    return _cached("bdb26_input", lambda: pl.read_csv(sorted((RAW / "bdb2026/train").glob("input_*.csv")), null_values=["NA"], infer_schema_length=100000))


def bdb26_output():
    return _cached("bdb26_output", lambda: pl.concat([
        pl.read_csv(f, null_values=["NA"]).with_columns(week=pl.lit(int(f.stem[-2:])))
        for f in sorted((RAW / "bdb2026/train").glob("output_*.csv"))]))


def bdb26_supp():
    return pl.read_csv(RAW / "bdb2026/supplementary_data.csv", null_values=["NA", ""], infer_schema_length=100000)


def bdb27_pp():
    return _cached("bdb27_pp", lambda: pl.read_csv(RAW / "bdb2027/player_play.csv", null_values=["NA"], infer_schema_length=200000))


def bdb27_track():
    return _cached("bdb27_track", lambda: pl.read_csv(RAW / "bdb2027/game_tracking_2023.csv", null_values=["NA"], infer_schema_length=200000))


def part():
    # old_game_id is str and play_id is f64 in the raw file: normalise to Int64 for joins
    return pl.read_parquet(RAW / "participation.parquet").with_columns(
        pl.col("old_game_id").cast(pl.Int64, strict=False), pl.col("play_id").cast(pl.Int64))
