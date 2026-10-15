"""nflverse participation/FTN/pbp: coverage label distribution by season, taxonomy change, team-season sample sizes, grouped agreement with BDB."""
import sys; sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import *

key = ["game_id", "play_id"]
pa = part()
pbp = (pl.scan_parquet(RAW / "pbp.parquet")
       .select(game_id=pl.col("old_game_id").cast(pl.Int64), play_id=pl.col("play_id").cast(pl.Int64), nflverse_game_id="game_id",
               season="season", season_type="season_type", week="week", posteam="posteam", defteam="defteam", qb_dropback="qb_dropback", play_type="play_type", down="down")
       .collect())
print("pbp rows", pbp.height, "seasons", pbp["season"].unique().sort().to_list(), "season_type", pbp["season_type"].value_counts().to_dicts())
print("participation rows", pa.height, "null old_game_id", pa["old_game_id"].null_count(), "dup keys", pa.height - pa.select("old_game_id", "play_id").unique().height)
pa = pa.rename({"old_game_id": "game_id"})
j = pbp.join(pa.select(key + ["defense_coverage_type", "defense_man_zone_type", "number_of_pass_rushers", "defense_players", "n_defense", "time_to_throw", "was_pressure"]), on=key, how="left")
db = j.filter(pl.col("qb_dropback") == 1)
print("pbp dropbacks without a participation row", db["n_defense"].is_null().sum(), "of", db.height)
db = db.with_columns(cov=pl.col("defense_coverage_type").replace("", None), mz=pl.col("defense_man_zone_type").replace("", None))
print("\n== labeled dropbacks by season (REG+POST)")
print(db.group_by("season").agg(dropbacks=pl.len(), cov_lab=pl.col("cov").is_not_null().sum(), mz_lab=pl.col("mz").is_not_null().sum(),
                                  rushers=pl.col("number_of_pass_rushers").is_not_null().sum()).with_columns(cov_pct=(pl.col("cov_lab") / pl.col("dropbacks") * 100).round(1)).sort("season"))
print("REG only:"); print(db.filter(pl.col("season_type") == "REG").group_by("season").agg(cov_lab=pl.col("cov").is_not_null().sum()).sort("season").transpose(include_header=False))
print("\n== coverage type % by season (dropbacks with label)")
lab = db.filter(pl.col("cov").is_not_null())
t = lab.group_by("season", "cov").len().with_columns(pct=(pl.col("len") / pl.col("len").sum().over("season") * 100).round(1))
print(t.pivot(on="season", index="cov", values="pct", sort_columns=True).sort("cov"))
print(t.pivot(on="season", index="cov", values="len", sort_columns=True).sort("cov"))
print("man/zone % by season"); mz = db.filter(pl.col("mz").is_not_null()).group_by("season", "mz").len().with_columns(pct=(pl.col("len") / pl.col("len").sum().over("season") * 100).round(1))
print(mz.pivot(on="season", index="mz", values="pct", sort_columns=True))
print("coverage label w/o man-zone label or vice versa by season")
print(db.group_by("season").agg(cov_no_mz=(pl.col("cov").is_not_null() & pl.col("mz").is_null()).sum(), mz_no_cov=(pl.col("mz").is_not_null() & pl.col("cov").is_null()).sum()).sort("season"))
print("cov x mz 2022 vs 2023"); print(lab.filter(pl.col("season").is_in([2022, 2023])).group_by("season", "cov", "mz").len().sort("season", "cov", "mz"))

print("\n== team-season labeled dropbacks (defense)")
ts = lab.filter(pl.col("season_type") == "REG").group_by("season", "defteam").len()
print(ts.group_by("season").agg(teams=pl.len(), min=pl.col("len").min(), p10=pl.col("len").quantile(.1), median=pl.col("len").median(), max=pl.col("len").max()).sort("season"))
tsc = lab.filter(pl.col("season_type") == "REG").group_by("season", "defteam", "cov").len()
print("per team-season, labeled dropbacks of rarer classes (median across teams):")
print(tsc.group_by("season", "cov").agg(pl.col("len").median()).pivot(on="season", index="cov", values="len", sort_columns=True).sort("cov"))
# by down within team-season: 3rd down dropbacks per team-season
print("3rd/4th-down labeled dropbacks per team-season:", lab.filter(pl.col("season_type") == "REG", pl.col("down") >= 3).group_by("season", "defteam").len()
      .group_by("season").agg(min=pl.col("len").min(), median=pl.col("len").median(), max=pl.col("len").max()).sort("season").to_dicts())

print("\n== FTN charting")
ftn = pl.read_parquet(RAW / "ftn_charting.parquet")
print(ftn.group_by("season").agg(rows=pl.len(), weeks=pl.col("week").max(), games=pl.col("nflverse_game_id").n_unique(),
      blitz_nn=pl.col("n_blitzers").is_not_null().sum(), rush_nn=pl.col("n_pass_rushers").is_not_null().sum()).sort("season"))
fj = ftn.join(pbp.rename({"nflverse_game_id": "ngid"}), left_on=["nflverse_game_id", "nflverse_play_id"], right_on=["ngid", "play_id"], how="inner", suffix="_pbp")
fdb = fj.filter(pl.col("qb_dropback") == 1)
print("FTN dropbacks by season", fdb.group_by("season").len().sort("season").to_dicts())
print("n_blitzers dist on dropbacks (all seasons)", fdb["n_blitzers"].value_counts().sort("n_blitzers").to_dicts())
print("blitz rate (n_blitzers>0) by season", fdb.group_by("season").agg(pl.col("n_blitzers").gt(0).mean().round(3)).sort("season").to_dicts())
print("n_pass_rushers dist on dropbacks", fdb["n_pass_rushers"].value_counts().sort("n_pass_rushers").to_dicts())
print("n_blitzers vs n_pass_rushers-4 crosstab 2023"); x = fdb.filter(pl.col("season") == 2023)
print(x.group_by("n_pass_rushers", "n_blitzers").len().pivot(on="n_blitzers", index="n_pass_rushers", values="len", sort_columns=True).sort("n_pass_rushers"))
# participation 2023+ number_of_pass_rushers vs FTN n_pass_rushers identical?
c = fj.select(game_id="game_id", play_id=pl.col("nflverse_play_id").cast(pl.Int64), ftn_r="n_pass_rushers").join(pa.select(key + ["number_of_pass_rushers"]), on=key)
c = c.filter(pl.col("ftn_r").is_not_null() & pl.col("number_of_pass_rushers").is_not_null())
print("participation number_of_pass_rushers == FTN n_pass_rushers (2022+ overlap):", c.height, round((c["ftn_r"] == c["number_of_pass_rushers"]).mean() * 100, 2), "%")
c2 = c.join(pbp.select(key + ["season"]), on=key)
print(c2.group_by("season").agg(pl.len(), eq=(pl.col("ftn_r") == pl.col("number_of_pass_rushers")).mean().round(4)).sort("season").to_dicts())

print("\n== grouped agreement: BDB (NGS taxonomy) vs nflverse/FTN on 2023-25")
def grp_ngs(c):
    return (pl.when(c.is_in(["COVER_0_MAN"])).then(pl.lit("zero")).when(c.is_in(["COVER_1_MAN", "COVER_3_ZONE"])).then(pl.lit("single"))
            .when(c.is_in(["COVER_2_ZONE", "COVER_2_MAN", "COVER_4_ZONE", "COVER_6_ZONE"])).then(pl.lit("two")).otherwise(None))
def grp_ftn(c):
    return (pl.when(c == "COVER_0").then(pl.lit("zero")).when(c.is_in(["COVER_1", "COVER_3"])).then(pl.lit("single"))
            .when(c.is_in(["COVER_2", "2_MAN", "COVER_4", "COVER_6"])).then(pl.lit("two")).otherwise(None))
g = pl.read_csv(RAW / "bdb2027/games.csv")
p27 = bdb27_pp().group_by(key).agg(tc=pl.col("team_coverage_type").drop_nulls().first(), mz27=pl.col("team_coverage_man_zone").drop_nulls().first()).join(g.select("game_id", "season"), on="game_id")
s26 = bdb26_supp().select(key + ["season", pl.col("team_coverage_type").alias("tc"), pl.col("team_coverage_man_zone").alias("mz27")])
for name, src in [("bdb26", s26), ("bdb27", p27)]:
    x = src.join(pa.select(key + ["defense_coverage_type", "defense_man_zone_type"]), on=key).filter(pl.col("tc").is_not_null(), pl.col("defense_coverage_type").is_not_null())
    x = x.with_columns(sh_a=grp_ngs(pl.col("tc")), sh_b=grp_ftn(pl.col("defense_coverage_type")), mz_b=pl.col("defense_man_zone_type").replace("", None))
    y = x.filter(pl.col("sh_a").is_not_null() & pl.col("sh_b").is_not_null())
    z = x.filter(pl.col("mz27").is_not_null() & pl.col("mz_b").is_not_null())
    print(f"{name}: shell (0/1-high/2-high, excl COMBO/9/BLOWN/PREVENT) n={y.height} agree={(y['sh_a']==y['sh_b']).mean()*100:.2f}%; "
          f"man/zone n={z.height} agree={(z['mz27']==z['mz_b']).mean()*100:.2f}%; FTN COMBO/9/BLOWN/PREVENT share={x['sh_b'].is_null().mean()*100:.1f}%")
    print(y.group_by("sh_a", "sh_b").len().pivot(on="sh_b", index="sh_a", values="len", sort_columns=True).sort("sh_a"))
    print(z.group_by("mz27", "mz_b").len().pivot(on="mz_b", index="mz27", values="len", sort_columns=True))
    print("by season", y.group_by("season").agg(pl.len(), shell_agree=(pl.col("sh_a") == pl.col("sh_b")).mean().round(4)).sort("season").to_dicts())
