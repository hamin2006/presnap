"""Phase 1A exploration: which frame and rule variant best matches the NGS played shell after the snap,
and how the two-high rate changes before the snap (StatsBomb). Train split only for selection."""
import itertools

import polars as pl

from presnap import io
from presnap.shell import KEY, shell_at

P = io.PROCESSED
plays = pl.read_parquet(P / "plays.parquet")
lab = plays.select(KEY + ["shell_played", "split", "family"]).drop_nulls("shell_played")
t = pl.scan_parquet(P / "tracking.parquet").filter((pl.col("side") == "def") & (pl.col("pos_group") != "DL"))


def two_deepest(d, depth, width, gap, sep):
    """1-high if the deepest inside defender is `gap`+ yards deeper than the next, or the two deepest are on the
    same side / within `sep` yards laterally; 2-high if the two deepest are both deep and split."""
    d = d.filter(pl.col("y_rel").abs() <= width).sort("x_rel", descending=True)
    g = d.group_by(KEY).agg(x1=pl.col("x_rel").first(), x2=pl.col("x_rel").get(1, null_on_oob=True),
                            y1=pl.col("y_rel").first(), y2=pl.col("y_rel").get(1, null_on_oob=True))
    return g.select(KEY, shell=pl.when(pl.col("x1") < depth).then(0)
                    .when((pl.col("x2") >= depth) & (pl.col("x1") - pl.col("x2") < gap) & ((pl.col("y1") - pl.col("y2")).abs() >= sep)
                          & (pl.col("y1").sign() != pl.col("y2").sign())).then(2).otherwise(1).cast(pl.Int8))


def acc(pred, split=("train",)):
    j = pred.join(lab.filter(pl.col("split").is_in(split)), on=KEY)
    return round((j["shell"] == j["shell_played"]).mean(), 4), j.height


b = t.filter(pl.col("source") == "bdb26")
last = b.group_by(KEY).agg(pl.col("frame").max().alias("last"))
print("== after the snap: best MOF rule and best two-deepest rule per frame (train)")
for f in [0, 3, 5, 8, 10, 15, 20, "last"]:
    if f == "last":
        fr = b.join(last, on=KEY).filter(pl.col("frame") == pl.col("last")).collect()
    else:  # plays whose throw comes earlier keep their last frame
        fr = b.join(last, on=KEY).filter(pl.col("frame") == pl.min_horizontal(pl.col("last"), pl.lit(f))).collect()
    mof = max((acc(shell_at(fr, d, m, w))[0], d, m, w) for d, m, w in itertools.product([9, 10, 12, 14], [2, 3, 4, 6], [14, 16, 20]))
    td = max((acc(two_deepest(fr, d, w, g, s))[0], d, w, g, s) for d, w, g, s in itertools.product([9, 10, 12], [14, 16, 20], [3, 5, 8], [6, 8, 10, 12]))
    print(f"frame {f}: MOF {mof}  two-deepest {td}")

print("\n== before the snap: StatsBomb two-high share by time before the snap (MOF rule, depth 9, mid 3, width 16)")
sb = t.filter((pl.col("source") == "statsbomb") & pl.col("frame").is_in([-30, -20, -15, -10, -5, -2, 0])).collect()
ok = plays.filter(pl.col("sb_quality_ok")).select(KEY + ["season", "shell_played"])
sb = sb.join(ok, on=KEY)
for f in [-30, -20, -15, -10, -5, -2, 0]:
    s = shell_at(sb.filter(pl.col("frame") == f), 9, 3, 16).join(ok, on=KEY)
    print(f"t={f / 10:+.1f}s n={s.height} two-high={(s['shell'] == 2).mean():.3f} one-high={(s['shell'] == 1).mean():.3f} "
          f"equals played={(s['shell'] == s['shell_played']).mean():.3f}")
