"""Misc checks: BDB 2026 plays per week + label drift, StatsBomb coordinate ranges and off-camera step behaviour."""
import sys; sys.path.insert(0, __file__.rsplit("/", 1)[0])
import gzip, json
from common import *

s = bdb26_supp().filter(pl.col("season") == 2023)
w = s.group_by("week").agg(plays=pl.len(), games=pl.col("game_id").n_unique(), c3=(pl.col("team_coverage_type") == "COVER_3_ZONE").mean().round(3),
                           c1=(pl.col("team_coverage_type") == "COVER_1_MAN").mean().round(3), man=(pl.col("team_coverage_man_zone") == "MAN_COVERAGE").mean().round(3)).sort("week")
print(w.to_dicts())
print("games per team in bdb26:", pl.concat([s.select(t="home_team_abbr", g="game_id"), s.select(t="visitor_team_abbr", g="game_id")]).unique().group_by("t").len()["len"].value_counts().to_dicts())
print("plays per defensive team (2023 tracking):", s.group_by("defensive_team").len().select(pl.col("len").min().alias("min"), pl.col("len").median().alias("med"), pl.col("len").max().alias("max")).to_dicts())

files = sorted((RAW / "statsbomb/tracking").glob("*.json.gz"))[::12]
rows = []
for f in files:
    d = json.load(gzip.open(f))
    for p in d["plays"]:
        for t in p["tracks"]:
            st = t["steps"]
            xs = [s["x"] for s in st if s.get("x") is not None]; ys = [s["y"] for s in st if s.get("y") is not None]
            if len(xs) < 2: continue
            still = sum(1 for a, b in zip(st, st[1:]) if a.get("x") == b.get("x") and a.get("y") == b.get("y")) / (len(st) - 1)
            keys = set().union(*[s.keys() for s in st])
            rows.append(dict(on_cam=t["on_camera_ratio"], still=still, xmin=min(xs), xmax=max(xs), ymin=min(ys), ymax=max(ys), ltr=p["offense_left_to_right"],
                             keys=",".join(sorted(keys)), null_xy=sum(1 for s in st if s.get("x") is None)))
R = pl.DataFrame(rows)
print("SB x range", R["xmin"].min(), R["xmax"].max(), "y range", R["ymin"].min(), R["ymax"].max())
print("step keys", R["keys"].value_counts().to_dicts(), "steps with null x", R["null_xy"].sum())
print("share of consecutive identical positions by on_camera_ratio bucket:")
print(R.with_columns(b=pl.col("on_cam").cut([0.5, 0.8, 0.95, 0.999])).group_by("b").agg(pl.len(), pl.col("still").median()).sort("b"))
