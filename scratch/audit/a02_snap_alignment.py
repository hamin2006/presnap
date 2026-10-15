"""Is BDB 2026 frame 1 the snap? Match BDB 2026 frames to BDB 2027 tracking (which has ball_snap events) by game/play/nfl_id and position."""
import sys; sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import *

key = ["game_id", "play_id"]
t = bdb27_track()
print("bdb27 tracking rows", t.height, "plays", t.select(key).unique().height, "games", t["game_id"].n_unique())
print("events:", t["event"].value_counts().sort("count", descending=True).to_dicts())
t = t.with_columns(ts=pl.col("time").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f")).sort(key + ["nfl_id", "ts"])
t = t.with_columns(fr=pl.int_range(pl.len()).over(key + ["nfl_id"]) + 1)
pp = t.group_by(key).agg(nply=pl.col("nfl_id").n_unique())
print("players per play in bdb27 tracking", pp["nply"].value_counts().sort("nply").to_dicts())
dt = t.group_by(key + ["nfl_id"]).agg(d=pl.col("ts").diff().dt.total_milliseconds().drop_nulls().unique())
print("frame spacing ms values", dt.explode("d")["d"].value_counts().sort("count", descending=True).head(5).to_dicts())
snap = t.filter(pl.col("event") == "ball_snap").group_by(key + ["nfl_id"]).agg(snap_fr=pl.col("fr").first())
first = t.group_by(key + ["nfl_id"]).agg(n=pl.len())
pre = snap.join(first, on=key + ["nfl_id"])
print("plays w/ ball_snap", snap.select(key).unique().height)
print("frames before snap in bdb27 (snap_fr-1):", pre.select(pl.col("snap_fr").sub(1).quantile(q).alias(str(q)) for q in [0, .05, .5, .95, 1]))

i = bdb26_input().select(key + ["nfl_id", "frame_id", "x", "y", "s", "a", "player_role"])
ov = i.select(key + ["nfl_id"]).unique().join(t.select(key + ["nfl_id"]).unique(), on=key + ["nfl_id"])
print("\n(game,play,nfl_id) in both bdb26 input and bdb27 tracking:", ov.height, "plays", ov.select(key).unique().height)
# match each bdb26 frame (1..5) to the bdb27 frame with identical position
m = (i.filter(pl.col("frame_id") <= 5).join(ov, on=key + ["nfl_id"])
     .join(t.select(key + ["nfl_id", "fr", "x", "y", "event"]), on=key + ["nfl_id", "x", "y"], how="left")
     .join(snap, on=key + ["nfl_id"], how="left"))
print("bdb26 frames (<=5) with exact x,y match in bdb27:", m["fr"].is_not_null().mean())
m1 = m.filter(pl.col("frame_id") == 1, pl.col("fr").is_not_null()).with_columns(off=pl.col("fr") - pl.col("snap_fr"))
# a player standing still can match several bdb27 frames; keep the match closest to the snap
m1 = m1.with_columns(absoff=pl.col("off").abs()).sort("absoff").group_by(key + ["nfl_id"]).agg(pl.col("off").first(), pl.col("player_role").first(), n_match=pl.len(), s=pl.col("s").first())
print("offset of bdb26 frame 1 from bdb27 ball_snap frame (0 = frame 1 is the snap):")
print(m1["off"].value_counts().sort("count", descending=True).head(10).to_dicts())
print("unique-match only (moving player, n_match==1):", m1.filter(pl.col("n_match") == 1)["off"].value_counts().sort("count", descending=True).head(8).to_dicts())
m1.write_parquet(CACHE / "snap_offsets.parquet")

# does the last input frame equal the throw (pass_forward)?
last = i.join(ov, on=key + ["nfl_id"]).filter(pl.col("frame_id") == pl.col("frame_id").max().over(key))
pf = t.filter(pl.col("event").is_in(["pass_forward", "pass_shovel"])).group_by(key + ["nfl_id"]).agg(pf_fr=pl.col("fr").first())
ml = (last.join(t.select(key + ["nfl_id", "fr", "x", "y"]), on=key + ["nfl_id", "x", "y"]).join(pf, on=key + ["nfl_id"]).join(snap, on=key + ["nfl_id"])
      .with_columns(off=pl.col("fr") - pl.col("pf_fr"), absoff=(pl.col("fr") - pl.col("pf_fr")).abs()).sort("absoff").group_by(key + ["nfl_id"]).agg(pl.col("off").first(), pl.col("frame_id").first(), (pl.col("pf_fr") - pl.col("snap_fr")).first().alias("snap_to_pf")))
print("\nlast bdb26 input frame minus bdb27 pass_forward frame:", ml["off"].value_counts().sort("count", descending=True).head(6).to_dicts())
print("bdb26 n_frames - (pass_forward - snap) :", (ml["frame_id"] - ml["snap_to_pf"]).value_counts().sort("count", descending=True).head(5).to_dicts())
