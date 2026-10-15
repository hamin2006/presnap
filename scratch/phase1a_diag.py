"""Where do shell misses come from? Per-family accuracy at +2.0 s, and safety geometry at the snap by played family."""
import polars as pl
from presnap import io
from presnap.shell import KEY
import importlib.util, sys
spec = importlib.util.spec_from_file_location("ex", "scratch/phase1a_explore.py")

P = io.PROCESSED
plays = pl.read_parquet(P / "plays.parquet")
lab = plays.select(KEY + ["shell_played", "split", "family", "coverage"]).drop_nulls("shell_played")
t = pl.scan_parquet(P / "tracking.parquet").filter((pl.col("side") == "def") & (pl.col("pos_group") != "DL"))

# depth/width of the two deepest defenders inside |y|<=16 at the snap, by played coverage
def geo(frame_df):
    d = frame_df.filter(pl.col("y_rel").abs() <= 16).sort("x_rel", descending=True)
    return d.group_by(KEY).agg(x1=pl.col("x_rel").first(), x2=pl.col("x_rel").get(1, null_on_oob=True),
                               y1=pl.col("y_rel").first(), y2=pl.col("y_rel").get(1, null_on_oob=True),
                               pos1=pl.col("position").first(), pos2=pl.col("position").get(1, null_on_oob=True))
for src, f in [("bdb26", 0), ("statsbomb", 0), ("statsbomb", -10)]:
    fr = t.filter((pl.col("source") == src) & (pl.col("frame") == f)).collect()
    g = geo(fr).join(lab, on=KEY)
    if src == "statsbomb":
        g = g.join(plays.filter(pl.col("sb_quality_ok")).select(KEY), on=KEY)
    print(f"\n== {src} frame {f}: two deepest inside the numbers, medians by played coverage")
    print(g.group_by("coverage").agg(n=pl.len(), x1=pl.col("x1").median(), x2=pl.col("x2").median(),
          lat_sep=(pl.col("y1") - pl.col("y2")).abs().median(), opp_sides=(pl.col("y1").sign() != pl.col("y2").sign()).mean().round(2),
          x2_ge9=(pl.col("x2") >= 9).mean().round(2)).sort("n", descending=True))
    print("deepest-two positions:", g.group_by("pos1").len().sort("len", descending=True).head(6).rows())
