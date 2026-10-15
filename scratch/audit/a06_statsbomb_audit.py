"""StatsBomb audit: tracked players, DB/safety visibility before the snap, pre-snap duration, label joins, depth/blitz vs nflverse."""
import sys; sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import *

P = pl.read_parquet(CACHE / "sb_plays.parquet"); T = pl.read_parquet(CACHE / "sb_tracks.parquet")  # built by a05
key = ["nfl_old_game_id", "gsis_play_id"]
pp = pl.read_parquet(RAW / "players.parquet").select(pl.col("nfl_id").cast(pl.Int64, strict=False), nfl_pos="position")
T = T.join(pp, left_on="gsis_player_id", right_on="nfl_id", how="left").with_columns(
    is_def=pl.col("team_id") != pl.col("off_team"),
    pre_frac=pl.when(pl.col("exp_pre") > 0).then(pl.col("n_pre") / pl.col("exp_pre")),
    pre1_frac=pl.col("n_pre1") / 30)
T = T.with_columns(role=pl.when(~pl.col("is_def")).then(pl.lit("OFF"))
                   .when(pl.col("pos") == "DB").then(pl.when(pl.col("nfl_pos").is_in(["FS", "SS", "S", "SAF"])).then(pl.lit("DB-safety")).when(pl.col("nfl_pos") == "CB").then(pl.lit("DB-CB")).otherwise(pl.lit("DB-other")))
                   .otherwise(pl.col("pos")))
print("plays", P.height, "pre-snap plays (tss_min<0)", (P["tss_min"] < 0).sum(), "plays starting after snap", (P["tss_min"] >= 0).sum())
print("def tracks per play", T.filter("is_def").group_by(key).len()["len"].value_counts().sort("len").to_dicts())
print("pos code mismatch: SB position_code vs team side", T.group_by("is_def", "pos").len().sort("is_def", "len").to_dicts())
print("\n== visibility by role (tracks on plays with pre-snap data)")
V = T.join(P.filter(pl.col("tss_min") < -1).select(key), on=key)
print(V.group_by("role").agg(n=pl.len(), on_cam_med=pl.col("on_cam").median(), on_cam_lt90=(pl.col("on_cam") < .9).mean().round(3),
                             pre_frac_med=pl.col("pre_frac").median(), pre_frac_mean=pl.col("pre_frac").mean().round(3),
                             pre1_ge90=(pl.col("pre1_frac") >= .9).mean().round(3), at_snap=pl.col("at_snap").mean().round(3),
                             zero_pre=(pl.col("n_pre") == 0).mean().round(3)).sort("role"))
print("safety pre_frac deciles", V.filter(pl.col("role") == "DB-safety").select(pl.col("pre_frac").quantile(q).alias(str(q)) for q in [.05, .1, .25, .5, .75]).to_dicts())
# per play safety visibility
# steps are continuous (no time gaps) even off camera, so on_camera_ratio (whole track) is the only visibility signal
sv = V.filter(pl.col("role") == "DB-safety").group_by(key).agg(n_saf=pl.len(), all_snap=pl.col("at_snap").all(), min_pre1=pl.col("pre1_frac").min(), min_on=pl.col("on_cam").min())
dbv = V.filter(pl.col("pos") == "DB", pl.col("is_def")).group_by(key).agg(n_db=pl.len(), db_all_snap=pl.col("at_snap").all(), db_min_pre1=pl.col("pre1_frac").min(), db_min_on=pl.col("on_cam").min())
print("safety on_camera_ratio quantiles", V.filter(pl.col("role") == "DB-safety").select(pl.col("on_cam").quantile(q).alias(str(q)) for q in [.01, .05, .1, .25, .5]).to_dicts())
print("on_camera_ratio < 0.9 share by role and SB season"); print(V.join(P.select(key + ["season"]), on=key).filter(pl.col("role").is_in(["DB-CB", "DB-safety", "LB", "DL", "OFF"])).group_by("season", "role").agg(lt90=(pl.col("on_cam") < .9).mean().round(3)).pivot(on="role", index="season", values="lt90").sort("season"))
print("plays: min safety on_cam >=0.9:", (sv["min_on"] >= .9).sum(), "of", sv.height, "; min DB on_cam >=0.9:", (dbv["db_min_on"] >= .9).sum(), "of", dbv.height)
print("safeties per play", sv["n_saf"].value_counts().sort("n_saf").to_dicts())

print("\n== pre-snap duration (s) by season (tss_min)")
print(P.group_by("season").agg(med=-pl.col("tss_min").median(), p10=-pl.col("tss_min").quantile(.9), p90=-pl.col("tss_min").quantile(.1),
                               lt1s=(pl.col("tss_min") > -1).mean().round(3), calib_bad=(pl.col("calib") > .1).mean().round(3)).sort("season"))

print("\n== join to plays CSV and nflverse")
S = pl.concat([pl.read_csv(f, infer_schema_length=0) for f in sorted((RAW / "statsbomb/plays").glob("*.csv"))], how="diagonal")
S = S.with_columns(nfl_old_game_id=pl.col("gsis_old_game_id").cast(pl.Float64, strict=False).cast(pl.Int64, strict=False), gsis_play_id=pl.col("gsis_play_id").cast(pl.Float64, strict=False).cast(pl.Int64, strict=False))
print("plays CSV rows", S.height, "null game id", S["nfl_old_game_id"].null_count(), "null play id", S["gsis_play_id"].null_count(), "seasons", S["season"].value_counts().sort("season").to_dicts())
print("tracking plays that join plays CSV", P.join(S.select(key).drop_nulls(), on=key).height, "of", P.height)
pa = part().select(nfl_old_game_id="old_game_id", gsis_play_id="play_id", cov="defense_coverage_type", mz="defense_man_zone_type", rushers="number_of_pass_rushers", def_pos="defense_positions")
pbp = pl.scan_parquet(RAW / "pbp.parquet").select(nfl_old_game_id=pl.col("old_game_id").cast(pl.Int64), gsis_play_id=pl.col("play_id").cast(pl.Int64), qb_dropback="qb_dropback", defteam="defteam", nflverse_game_id="game_id").collect()
ftn = pl.read_parquet(RAW / "ftn_charting.parquet").select("nflverse_game_id", gsis_play_id=pl.col("nflverse_play_id").cast(pl.Int64), ftn_blitzers="n_blitzers", ftn_rushers="n_pass_rushers")
J = (P.join(pa, on=key, how="left").join(pbp, on=key, how="left").join(ftn, on=["nflverse_game_id", "gsis_play_id"], how="left")
     .join(S.select(key + ["play_defensive_back_depths", "play_blitz", "play_drop_eight", "defense_team_name", "play_type"]).drop_nulls(key).unique(key), on=key, how="left", suffix="_csv")
     .join(sv, on=key, how="left").join(dbv, on=key, how="left"))
lab = J.filter(pl.col("cov").is_not_null() & (pl.col("cov") != ""))
print("tracking plays with nflverse coverage label:", lab.height, "dropbacks:", lab.filter(pl.col("qb_dropback") == 1).height, "by season", lab.group_by("season").len().sort("season").to_dicts())
dbk = lab.filter(pl.col("qb_dropback") == 1, pl.col("tss_min") < -1)
good = dbk.filter(pl.col("all_snap") & (pl.col("min_on") >= .9) & (pl.col("n_saf") >= 1))
gooddb = dbk.filter(pl.col("db_all_snap") & (pl.col("db_min_on") >= .9))
print("labeled dropbacks with >=1 s pre-snap:", dbk.height, "| all roster-identified safeties on_camera_ratio>=0.9 & step at snap:", good.height, "| ALL DBs on_camera_ratio>=0.9:", gooddb.height)
print("good-safety labeled dropbacks by defense (Brady's team on D vs opponents):", good.with_columns(brady_def=pl.col("defteam").is_in(["NE", "TB"])).group_by("brady_def").len().to_dicts())
print("good-safety label dist", good["cov"].value_counts().sort("count", descending=True).to_dicts())

print("\n== play_defensive_back_depths vs coverage shell")
dd = lab.filter(pl.col("play_defensive_back_depths").is_not_null()).with_columns(
    depths=pl.col("play_defensive_back_depths").str.strip_chars("{}").str.split(",").cast(pl.List(pl.Float64), strict=False))
dd = dd.with_columns(n_deep=pl.col("depths").list.eval((pl.element() >= 10).sum()).list.first(), n_db=pl.col("depths").list.len())
dd = dd.with_columns(shell=pl.when(pl.col("cov") == "COVER_0").then(pl.lit("zero")).when(pl.col("cov").is_in(["COVER_1", "COVER_3"])).then(pl.lit("single"))
                     .when(pl.col("cov").is_in(["COVER_2", "2_MAN", "COVER_4", "COVER_6"])).then(pl.lit("two")).otherwise(pl.lit("other")))
print("DB count in depth array", dd["n_db"].value_counts().sort("n_db").to_dicts())
print(dd.group_by("shell", pl.col("n_deep").clip(0, 3)).len().pivot(on="n_deep", index="shell", values="len", sort_columns=True).sort("shell"))
for thr in [8, 10, 12]:
    z = dd.with_columns(nd=pl.col("depths").list.eval((pl.element() >= thr).sum()).list.first()).filter(pl.col("shell").is_in(["zero", "single", "two"]))
    pred = pl.when(pl.col("nd") == 0).then(pl.lit("zero")).when(pl.col("nd") == 1).then(pl.lit("single")).otherwise(pl.lit("two"))
    print(f"depth>={thr}yd rule: n={z.height} shell accuracy={(z.select(pred == pl.col('shell')).to_series().mean())*100:.1f}%  (majority-class baseline {z['shell'].value_counts()['count'].max()/z.height*100:.1f}%)")

print("\n== play_blitz vs nflverse/FTN")
b = J.filter(pl.col("qb_dropback") == 1)
print("play_blitz values on dropbacks", b["play_blitz"].value_counts().to_dicts())
print("play_blitz null by season", b.group_by("season").agg(null=pl.col("play_blitz").is_null().mean().round(3), n=pl.len()).sort("season").to_dicts())
print("raw play_blitz values by season (dropbacks)"); print(b.group_by("season", "play_blitz").len().sort("season", "play_blitz"))
# play_blitz is only ever True/TRUE or NA/null (no explicit False); treat True/TRUE as blitz and everything else as no-blitz,
# restricted to seasons where the column is populated (2017-2020; 2021-22 are mostly null)
b = b.with_columns(sb=pl.col("play_blitz").str.to_lowercase() == "true").with_columns(pl.col("sb").fill_null(False))
for nm, sub in [("2017-2020 SB seasons", b.filter(pl.col("season").is_in(["2017/2018", "2018/2019", "2019/2020", "2020/2021"]))), ("2021-2022 SB seasons", b.filter(pl.col("season").is_in(["2021/2022", "2022/2023"])))]:
    bb = sub.filter(pl.col("rushers").is_not_null()).with_columns(r5=pl.col("rushers") >= 5)
    print(nm, "SB blitz rate", round(sub["sb"].mean() * 100, 1), "% | vs participation rushers>=5: n", bb.height, "agree", round((bb["sb"] == bb["r5"]).mean() * 100, 1), "%", bb.group_by("sb", "r5").len().sort("sb", "r5").to_dicts())
bf = b.filter(pl.col("ftn_blitzers").is_not_null()).with_columns(fb=pl.col("ftn_blitzers") > 0)
print("SB play_blitz vs FTN n_blitzers>0 (2022): n", bf.height, "agree", round((bf["sb"] == bf["fb"]).mean() * 100, 1), "%", bf.group_by("sb", "fb").len().sort("sb", "fb").to_dicts())
bfr = bf.filter(pl.col("rushers").is_not_null()).with_columns(r5=pl.col("rushers") >= 5)
print("same 2022 plays: participation(NGS) rushers>=5 vs FTN n_blitzers>0 agree", round((bfr["r5"] == bfr["fb"]).mean() * 100, 1), "% n", bfr.height)
