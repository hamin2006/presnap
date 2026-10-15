"""BDB 2026 audit: frame 1 = snap?, frames per play, which players, coverage counts, label distribution."""
import sys; sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import *

i = bdb26_input(); o = bdb26_output(); s = bdb26_supp()
key = ["game_id", "play_id"]
plays = i.select(key).unique()
print("input rows", i.height, "plays", plays.height, "games", i["game_id"].n_unique())
print("supp rows", s.height, s.group_by("season").len().sort("season"))
print("supp dup keys", s.height - s.select(key).unique().height)
print("input plays missing from supp", plays.join(s, on=key, how="anti").height)
print("supp 2023 plays missing from input", s.filter(pl.col("season") == 2023).join(plays, on=key, how="anti").height)

print("\n== roles / sides"); print(i.filter(pl.col("frame_id") == 1).group_by("player_side", "player_role").len().sort("len"))
f1 = i.filter(pl.col("frame_id") == 1)
print("positions by role"); print(f1.group_by("player_role", "player_position").len().sort("player_role", "len", descending=[False, True]))
per = f1.group_by(key).agg(
    n=pl.len(), n_cov=(pl.col("player_role") == "Defensive Coverage").sum(),
    n_def=(pl.col("player_side") == "Defense").sum(), n_route=(pl.col("player_role") == "Other Route Runner").sum() + (pl.col("player_role") == "Targeted Receiver").sum(),
    n_pass=(pl.col("player_role") == "Passer").sum(), n_tgt=(pl.col("player_role") == "Targeted Receiver").sum())
for c in ["n", "n_cov", "n_def", "n_route", "n_pass", "n_tgt"]:
    print(c, per.group_by(c).len().sort(c).to_dicts())
print("defense side but not coverage role:", f1.filter((pl.col("player_side") == "Defense") & (pl.col("player_role") != "Defensive Coverage")).height)
print("frame-1 rows per play == distinct nfl_id per play:", (f1.group_by(key).agg(pl.col("nfl_id").n_unique() == pl.len())["nfl_id"].all()))

print("\n== frames per play (input = snap..throw)")
nf = i.group_by(key).agg(nfr=pl.col("frame_id").max(), nmin=pl.col("frame_id").min(), nout=pl.col("num_frames_output").first(),
                         nrows=pl.len(), nply=pl.col("nfl_id").n_unique())
print("min frame_id values", nf["nmin"].value_counts().to_dicts())
print("all players have all frames:", (nf["nrows"] == nf["nfr"] * nf["nply"]).all())
print(nf.select(pl.col("nfr").quantile(q).alias(f"p{int(q*100)}") for q in [0, .05, .25, .5, .75, .95, 1]))
print("mean input frames", nf["nfr"].mean(), "-> seconds", nf["nfr"].mean() / 10)
print(nf.select(pl.col("nout").quantile(q).alias(f"p{int(q*100)}") for q in [0, .05, .5, .95, 1]))
print("plays with >=50 input frames", (nf["nfr"] >= 50).sum(), ">=80", (nf["nfr"] >= 80).sum())

print("\n== kinematics at frames 1..15 by role (median s, a; share with s<0.3)")
k = (i.filter(pl.col("frame_id") <= 15).group_by("player_role", "frame_id")
       .agg(s_med=pl.col("s").median(), s_mean=pl.col("s").mean(), a_med=pl.col("a").median(), still=(pl.col("s") < 0.3).mean())
       .sort("player_role", "frame_id"))
print(k)
# passer displacement from frame 1 over first 10 frames (QB should start moving immediately after snap)
pas = i.filter(pl.col("player_role") == "Passer", pl.col("frame_id") <= 11).sort(key + ["frame_id"])
d = pas.group_by(key).agg(dx0=(pl.col("x").get(1) - pl.col("x").first()).abs(), dx10=(pl.col("x").last() - pl.col("x").first()).abs())
print("passer |dx| frame1->2 median", d["dx0"].median(), " frame1->11 median", d["dx10"].median())

print("\n== play_direction and coordinates")
print(f1.group_by("play_direction").agg(pl.len(), x_mean=pl.col("x").mean(), los=pl.col("absolute_yardline_number").mean()))
print("x range", i["x"].min(), i["x"].max(), "y range", i["y"].min(), i["y"].max())
# offense at frame1 should sit behind LOS in direction of play: check sign
f1o = f1.filter(pl.col("player_side") == "Offense").with_columns(
    rel=pl.when(pl.col("play_direction") == "right").then(pl.col("x") - pl.col("absolute_yardline_number")).otherwise(pl.col("absolute_yardline_number") - pl.col("x")))
print("offense x relative to absolute_yardline_number (pos = downfield) by direction:")
print(f1o.group_by("play_direction").agg(pl.col("rel").median(), pl.col("rel").quantile(.05).alias("p5"), pl.col("rel").quantile(.95).alias("p95")))
f1d = f1.filter(pl.col("player_side") == "Defense").with_columns(
    rel=pl.when(pl.col("play_direction") == "right").then(pl.col("x") - pl.col("absolute_yardline_number")).otherwise(pl.col("absolute_yardline_number") - pl.col("x")))
print("defense depth relative to absolute_yardline_number by direction:")
print(f1d.group_by("play_direction").agg(pl.col("rel").median(), pl.col("rel").quantile(.05).alias("p5"), pl.col("rel").quantile(.95).alias("p95")))
print("output players per play", o.group_by(key).agg(pl.col("nfl_id").n_unique()).group_by("nfl_id").len().sort("nfl_id").to_dicts())
print("player_to_predict rows by role at f1"); print(f1.group_by("player_role", "player_to_predict").len().sort("player_role"))

print("\n== nulls in input"); print(i.null_count())
print("nulls in supp (non-zero)"); nc = s.null_count(); print({c: nc[c][0] for c in nc.columns if nc[c][0]})

print("\n== supplementary labels (2023 plays with tracking)")
s23 = s.join(plays, on=key)
print("n", s23.height)
print(s23.group_by("team_coverage_type").len().sort("len", descending=True).with_columns(pct=pl.col("len") / s23.height * 100))
print(s23.group_by("team_coverage_man_zone").len().sort("len", descending=True))
print("supp all-season coverage type"); print(s.group_by("season", "team_coverage_type").len().pivot(on="season", index="team_coverage_type", values="len").sort("team_coverage_type"))
print(s23.group_by("pass_result").len().sort("len"))
print("nullified by penalty", s23.group_by("play_nullified_by_penalty").len())
dd = s23.with_columns(dist=pl.when(pl.col("yards_to_go") <= 3).then(pl.lit("1-3")).when(pl.col("yards_to_go") <= 6).then(pl.lit("4-6"))
                     .when(pl.col("yards_to_go") <= 10).then(pl.lit("7-10")).otherwise(pl.lit("11+")),)
tab = dd.group_by("down", "dist", "team_coverage_type").len()
tot = dd.group_by("down", "dist").len().rename({"len": "tot"})
tab = tab.join(tot, on=["down", "dist"]).with_columns(pct=(pl.col("len") / pl.col("tot") * 100).round(1))
print(tab.pivot(on="team_coverage_type", index=["down", "dist", "tot"], values="pct").sort("down", "dist"))
print("defenders_in_the_box dist", s23["defenders_in_the_box"].value_counts().sort("defenders_in_the_box").to_dicts())

print("\n== does n_cov imply rushers? compare 11-n_cov with participation number_of_pass_rushers and FTN n_pass_rushers")
pa = part().select(game_id="old_game_id", play_id="play_id", rushers="number_of_pass_rushers", part_n_def="n_defense", cov="defense_coverage_type")
ftn = pl.read_parquet(RAW / "ftn_charting.parquet").filter(pl.col("season") == 2023)
pbp = (pl.scan_parquet(RAW / "pbp.parquet").filter(pl.col("season") == 2023)
       .select(pl.col("old_game_id").cast(pl.Int64), pl.col("play_id").cast(pl.Int64), nflverse_game_id="game_id").collect())
ftn = ftn.join(pbp, left_on=["nflverse_game_id", "nflverse_play_id"], right_on=["nflverse_game_id", "play_id"], how="left") \
         .select(game_id="old_game_id", play_id="nflverse_play_id", ftn_rush="n_pass_rushers", ftn_blitz="n_blitzers")
ftn = ftn.with_columns(pl.col("play_id").cast(pl.Int64))
c = per.join(pa, on=key, how="left").join(ftn, on=key, how="left").with_columns(impl=11 - pl.col("n_cov"))
print("joined to participation", c["rushers"].is_not_null().sum(), "/", c.height, "; to FTN", c["ftn_rush"].is_not_null().sum())
for col in ["rushers", "ftn_rush"]:
    cc = c.filter(pl.col(col).is_not_null())
    print(col, "exact match 11-n_cov:", round((cc["impl"] == cc[col]).mean() * 100, 1), "%  within 1:", round(((cc["impl"] - cc[col]).abs() <= 1).mean() * 100, 1), "%")
    print(cc.group_by("impl", col).len().pivot(on=col, index="impl", values="len", sort_columns=True).sort("impl"))
print("n_defense in participation for these plays", c["part_n_def"].value_counts().sort("part_n_def").to_dicts())
