"""StatsBomb tracking: per-track summary (position, on_camera_ratio, pre-snap coverage of steps). Writes cache/sb_tracks.parquet + cache/sb_plays.parquet."""
import sys; sys.path.insert(0, __file__.rsplit("/", 1)[0])
import gzip, json
from multiprocessing import Pool
from common import *

FPS = 30


def one(f):
    d = json.load(gzip.open(f))
    plays, tracks = [], []
    for p in d["plays"]:
        tss_all = [s["time_since_snap"] for t in p["tracks"] for s in t["steps"] if s.get("time_since_snap") is not None]
        tmin = min(tss_all) if tss_all else None
        tmax = max(tss_all) if tss_all else None
        plays.append(dict(file=f.name, nfl_old_game_id=d["nfl_old_game_id"], season=d["season_name"], gsis_play_id=p.get("gsis_play_id"),
                          play_type=p.get("play_type"), down=p.get("play_down"), off_team=p.get("play_offense_team_id"),
                          ltr=p.get("offense_left_to_right"), player_coverage_count=p.get("player_coverage_count"),
                          calib=p.get("calibration_fault_ratio"), n_tracks=len(p["tracks"]), tss_min=tmin, tss_max=tmax))
        for t in p["tracks"]:
            st = t["steps"]
            # off-camera steps exist but have x/y = null: count only steps with a position
            tss = [s["time_since_snap"] for s in st if s.get("time_since_snap") is not None and s.get("x") is not None]
            pre = sum(1 for x in tss if x < 0)
            exp_pre = round(-tmin * FPS) if (tmin is not None and tmin < 0) else 0
            # visible in the last 1 s before the snap
            pre1 = sum(1 for x in tss if -1.0 <= x < 0)
            tracks.append(dict(nfl_old_game_id=d["nfl_old_game_id"], gsis_play_id=p.get("gsis_play_id"), off_team=p.get("play_offense_team_id"),
                               team_id=t.get("team_id"), pos=(t.get("player") or {}).get("position_code"), gsis_player_id=(t.get("player") or {}).get("gsis_player_id"),
                               has_player=t.get("player") is not None, on_cam=t.get("on_camera_ratio"), n_steps=len(st), n_xy=sum(1 for s in st if s.get("x") is not None), n_pre=pre, exp_pre=exp_pre, n_pre1=pre1,
                               at_snap=any(abs(x) <= 0.5 / FPS + 1e-6 for x in tss),
                               n_calib=sum(1 for s in st if s.get("calibration_fault")),
                               ngs_ratio_x=(st[0]["ngs_x"] / st[0]["x"]) if st and st[0].get("x") and st[0].get("ngs_x") is not None else None))
    return plays, tracks


if __name__ == "__main__":
    files = sorted((RAW / "statsbomb/tracking").glob("*.json.gz"))
    with Pool(8) as pool:
        res = pool.map(one, files)
    P = pl.DataFrame([r for a, _ in res for r in a], infer_schema_length=None)
    T = pl.DataFrame([r for _, b in res for r in b], infer_schema_length=None)
    P.write_parquet(CACHE / "sb_plays.parquet"); T.write_parquet(CACHE / "sb_tracks.parquet")
    print("files", len(files), "plays", P.height, "tracks", T.height)
