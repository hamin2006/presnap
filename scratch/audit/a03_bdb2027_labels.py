"""BDB 2027 per-player labels: join to BDB 2026, coverage per defender, team_coverage_type agreement across sources."""
import sys; sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import *

key = ["game_id", "play_id"]
g = pl.read_csv(RAW / "bdb2027/games.csv")
p = bdb27_pp().join(g.select("game_id", "season", "week", "season_type"), on="game_id", how="left")
p = p.with_columns(is_def=pl.col("team_abbr") != pl.col("possession_team"))
print("season x season_type plays"); print(p.group_by("season", "season_type").agg(plays=pl.struct(key).n_unique()).sort("season", "season_type"))

print("\n== label presence by side / season")
print(p.group_by("season", "is_def").agg(rows=pl.len(), cov=pl.col("coverage_assignment").is_not_null().sum(),
      cov_snap=pl.col("coverage_assignment_at_snap").is_not_null().sum(), blitz_nn=pl.col("blitzing").is_not_null().sum(),
      blitz_true=pl.col("blitzing").fill_null(False).sum()).sort("season", "is_def"))
d = p.filter("is_def")
print("defender lined_up_position x coverage_assignment-not-null"); print(d.group_by("lined_up_position").agg(n=pl.len(), cov=pl.col("coverage_assignment").is_not_null().mean().round(3), blitz=pl.col("blitzing").mean().round(3)).sort("n", descending=True))
print("assignment changed between snap and final:", d.filter(pl.col("coverage_assignment").is_not_null() & pl.col("coverage_assignment_at_snap").is_not_null())
      .select((pl.col("coverage_assignment") != pl.col("coverage_assignment_at_snap")).mean()).item())
ch = d.filter(pl.col("coverage_assignment") != pl.col("coverage_assignment_at_snap")).group_by("coverage_assignment_at_snap", "coverage_assignment").len().sort("len", descending=True)
print("top snap->final changes", ch.head(12).to_dicts())
print("blitzing x coverage_assignment (defense)"); print(d.group_by("blitzing", pl.col("coverage_assignment").is_null().alias("cov_null")).len().sort("blitzing"))
print("coverage_assignment by team_coverage_type (top)")
print(d.filter(pl.col("coverage_assignment").is_not_null()).group_by("team_coverage_type", "coverage_assignment").len().sort("team_coverage_type", "len", descending=[False, True]).group_by("team_coverage_type", maintain_order=True).head(5))

print("\n== join to BDB 2026")
i = bdb26_input().filter(pl.col("frame_id") == 1).select(key + ["nfl_id", "player_role", "player_position", "player_side"])
plays26 = i.select(key).unique()
pl27 = p.select(key).unique()
j = plays26.join(pl27, on=key)
print("bdb26 plays", plays26.height, "with any bdb27 player_play row", j.height)
s = bdb26_supp()
print("bdb26 supp (incl. 2024) plays joined", s.select(key).join(pl27, on=key).height, "of", s.height)
cov26 = i.filter(pl.col("player_side") == "Defense")
jj = cov26.join(d.select(key + ["nfl_id", "coverage_assignment", "coverage_assignment_at_snap", "blitzing", "lined_up_position"]), on=key + ["nfl_id"], how="left")
per = jj.group_by(key).agg(n_cov=pl.len(), n_lab=pl.col("coverage_assignment").is_not_null().sum(), n_row=pl.col("lined_up_position").is_not_null().sum() + pl.col("blitzing").is_not_null().sum().clip(0, 0))
print("bdb26 coverage defenders:", cov26.height, "with a bdb27 row:", jj.filter(pl.col("lined_up_position").is_not_null() | pl.col("blitzing").is_not_null() | pl.col("coverage_assignment").is_not_null()).height,
      "with coverage_assignment:", jj["coverage_assignment"].is_not_null().sum())
print("labeled coverage defenders per bdb26 play", per["n_lab"].value_counts().sort("n_lab").to_dicts())
print("plays where ALL coverage defenders labeled", (per["n_lab"] == per["n_cov"]).sum())
print("assignment values for bdb26 coverage defenders"); print(jj["coverage_assignment"].value_counts().sort("count", descending=True))
print("bdb26 defenders labeled as blitzing=True:", jj["blitzing"].sum())
# defenders in bdb27 for bdb26 plays but NOT in bdb26 tracking (i.e. rushers)
notin = d.join(plays26, on=key).join(cov26.select(key + ["nfl_id"]), on=key + ["nfl_id"], how="anti")
print("bdb27 defender rows on bdb26 plays that are absent from bdb26 tracking:", notin.height, notin["lined_up_position"].value_counts().sort("count", descending=True).head(6).to_dicts(),
      "blitzing", notin["blitzing"].value_counts().to_dicts(), "cov", notin["coverage_assignment"].value_counts().sort("count", descending=True).head(4).to_dicts())

print("\n== team_coverage_type agreement")
tc27 = p.group_by(key).agg(tc27=pl.col("team_coverage_type").drop_nulls().first(), mz27=pl.col("team_coverage_man_zone").drop_nulls().first(),
                           n_tc=pl.col("team_coverage_type").drop_nulls().n_unique())
print("plays with >1 distinct team_coverage_type across rows (internal inconsistency):", (tc27["n_tc"] > 1).sum())
a = s.select(key + ["season", "team_coverage_type", "team_coverage_man_zone"]).join(tc27, on=key)
a = a.filter(pl.col("team_coverage_type").is_not_null() & pl.col("tc27").is_not_null())
print("bdb26 vs bdb27 both labeled:", a.height, "agree:", round((a["team_coverage_type"] == a["tc27"]).mean() * 100, 2), "%")
print(a.filter(pl.col("team_coverage_type") != pl.col("tc27")).group_by("team_coverage_type", "tc27").len().sort("len", descending=True).head(15).to_dicts())

pa = part().filter(pl.col("old_game_id") >= 2023000000).select(game_id="old_game_id", play_id="play_id", ftn_cov="defense_coverage_type", ftn_mz="defense_man_zone_type")
print("participation 2023+ coverage values", pa["ftn_cov"].value_counts().sort("count", descending=True).to_dicts())
print("participation 2023+ man/zone values", pa["ftn_mz"].value_counts().to_dicts())
for name, src, col in [("bdb26 (2023-24)", s.select(key + ["team_coverage_type", "season"]), "team_coverage_type"),
                       ("bdb27 (2023-25)", tc27.join(p.select(key + ["season"]).unique(), on=key).rename({"tc27": "team_coverage_type"}), "team_coverage_type")]:
    x = src.join(pa, on=key).filter(pl.col(col).is_not_null() & pl.col("ftn_cov").is_not_null())
    # crude taxonomy map: strip suffix, compare Cover N
    x = x.with_columns(ngs=pl.col(col).str.replace("COVER_", "COVER_").str.replace(r"_(MAN|ZONE)$", ""),
                       ftn=pl.col("ftn_cov").str.replace("2_MAN", "2"))
    x = x.with_columns(match=(pl.col(col) == "COVER_2_MAN") & (pl.col("ftn_cov") == "2_MAN") | (pl.col(col) != "COVER_2_MAN") & (pl.col("ngs") == pl.col("ftn_cov")))
    print(f"\n{name} vs nflverse/FTN: joined+labeled {x.height}; exact taxonomy match {x['match'].mean()*100:.2f}%")
    print(x.group_by("season").agg(pl.len(), agree=pl.col("match").mean().round(4)).sort("season"))
    print(x.group_by(col, "ftn_cov").len().pivot(on="ftn_cov", index=col, values="len", sort_columns=True).sort(col))
