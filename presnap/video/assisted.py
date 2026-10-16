"""Phase 2E: coach-assisted video. A coach clicks the 22 players on the snap frame; we track them +/-2 s and read
coverage. Simulated here on the 117 labelled All29 plays:

  clicks      = NGS positions at the snap projected into the frame (NGS-assisted registration) + 2 px click noise
  registration= line-based, labeled with the clicked players as the LOS anchor (deployable), carried frame to frame
  tracking    = pyramidal Lucas-Kanade on a patch above each player's feet at 60 fps, forward to +2 s and back to
                -2 s; variant "snap" also snaps each track to the nearest detector box every 0.5 s
  scoring     = per-player field error vs NGS over time (identities are known from the clicks), then the coverage
                models on those tracks vs on the NGS tracks of the same plays

Run on the PC: uv run --group train --group cv python -m presnap.video.assisted [--plays N]
"""
import argparse
import json

import cv2
import numpy as np
import polars as pl
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

from presnap import io, labels, seq
from presnap.video import detect as Dt, register as R
from presnap.video.pipeline import T_MAX, T_MIN, eval_plays, infer_roles, load_models, predict, to_tensors, MID
from presnap.video.pseudo import FPS, V, match, oracle_snap, project


SPAN = 2.0          # seconds each side of the snap
SNAP_EVERY = 30     # native frames between detector snaps (0.5 s)
LK = dict(winSize=(15, 15), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))


def read_clip(p):
    cap = cv2.VideoCapture(str(V / f"train/{p['game_play']}_All29.mp4"))
    t0 = (p["snap"] - p["start"]).total_seconds()
    k0, k1 = round((t0 - SPAN) * FPS), round((t0 + SPAN) * FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(k0, 0))
    frames = []
    for _ in range(k1 - max(k0, 0) + 1):
        ok, img = cap.read()
        if not ok:
            break
        frames.append(img)
    return frames, round(t0 * FPS) - max(k0, 0)  # frames, index of the snap frame


def lk_track(frames, k_snap, feet, hpx, boxes_at=None):
    """Track feet points from the snap frame in both directions. boxes_at: {frame index: boxes} for snapping."""
    n, P = len(frames), len(feet)
    out = np.full((n, P, 2), np.nan, np.float32)
    out[k_snap] = feet
    gray = {}

    def g(k):
        if k not in gray:
            gray[k] = cv2.cvtColor(frames[k], cv2.COLOR_BGR2GRAY)
        return gray[k]

    for step in (1, -1):
        cur = feet.copy()
        k = k_snap
        while 0 <= k + step < n:
            a, b = g(k), g(k + step)
            for i in range(P):
                h = hpx[i]
                xs = np.linspace(cur[i, 0] - .25 * h, cur[i, 0] + .25 * h, 3)
                ys = np.linspace(cur[i, 1] - .9 * h, cur[i, 1] - .2 * h, 5)
                pts = np.array([(x, y) for x in xs for y in ys], np.float32)[:, None]
                nxt, st, _ = cv2.calcOpticalFlowPyrLK(a, b, pts, None, **LK)
                ok = st[:, 0] == 1
                if ok.sum() >= 3:
                    cur[i] += np.median((nxt - pts)[ok, 0], axis=0)
            k += step
            if boxes_at is not None and k in boxes_at and len(boxes_at[k]):
                bf = Dt.feet(boxes_at[k])
                d = np.linalg.norm(cur[:, None] - bf[None], axis=-1)
                r, c = linear_sum_assignment(d)
                for i, j in zip(r, c):
                    if d[i, j] < 0.5 * hpx[i]:
                        cur[i] = bf[j]
            out[k] = cur
    return out


def register_clip(frames, k_snap, H0, orient):
    """Homography per 10 Hz step (native frame index), carried outward from the snap."""
    Hs = {0: H0}
    for direction in (1, -1):
        H = H0
        for s in range(direction, direction * (int(SPAN * 10) + 1), direction):
            k = k_snap + round(s / 10 * FPS)
            if not 0 <= k < len(frames):
                break
            lines = R.detect_lines(frames[k])
            if lines is not None:
                Hn = R.pick_consistent(lines, H, *orient, *frames[k].shape[:2])
                if Hn is not None:
                    H, _ = R.refine(Hn, frames[k])
            Hs[s] = H
    return Hs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plays", type=int, default=None)
    ap.add_argument("--models", default="sb_ft,sb_ft_robust")
    a = ap.parse_args()
    plays, held = eval_plays()
    if a.plays:
        plays = plays.head(a.plays)
    S_all = seq.situation(plays)
    tags = a.models.split(",")
    models = {t: load_models(t, n_sit=S_all.shape[1]) for t in tags}
    det = YOLO("yolo11x.pt")
    trk = pl.read_csv(V / "train_player_tracking.csv").filter(pl.col("step").is_between(-int(SPAN * 10), int(SPAN * 10)))
    rng = np.random.default_rng(0)
    out, errs = [], []
    for i, p in enumerate(plays.iter_rows(named=True)):
        t = trk.filter(pl.col("game_play") == p["game_play"])
        s0 = t.filter(pl.col("step") == 0).sort("nfl_player_id")
        ids = s0["nfl_player_id"].to_list()
        ngs0 = s0.select("x_position", "y_position").to_numpy()
        ol = s0.filter(pl.col("position").is_in(["C", "G", "T", "OG", "OT"]))
        dl = s0.filter(pl.col("position").is_in(["DE", "DT", "NT", "DL"]))
        if ol.height == 0 or dl.height == 0:
            continue
        los = (ol["x_position"].mean() + dl["x_position"].mean()) / 2
        true_dir = 1 if dl["x_position"].mean() > ol["x_position"].mean() else -1
        offense_home = ol["team"][0] == "home"
        side = np.array(["off" if (tm == "home") == offense_home else "def" for tm in s0["team"]], dtype=object)
        rec = dict(game_play=p["game_play"], coverage=p["coverage"],
                   y_cov=labels.COVERAGE7.index(p["coverage"]) if p["coverage"] in labels.COVERAGE7 else -1,
                   y_man=int(p["man_zone"] == "MAN"), y_shell=p["shell_played"], ok=False)

        frames, k_snap = read_clip(p)
        if not frames or k_snap >= len(frames):
            out.append(rec); continue
        img0 = frames[k_snap]
        lines = R.detect_lines(img0)
        if lines is None:
            out.append(rec); continue
        b0, _ = Dt.detect(det, img0, conf=0.08)
        f0 = Dt.feet(b0) if len(b0) else np.zeros((0, 2))
        orc = oracle_snap(lines, f0, ngs0)            # only to place the simulated clicks
        if orc is None or orc[0] < 6:
            rec["fail"] = "simulate"; out.append(rec); continue
        _, H_or, orient = orc
        H_or, _ = R.refine(H_or, img0)  # best estimate of the true registration, used only to place the clicks
        clicks = project(H_or, ngs0) + rng.normal(0, 2, (len(ngs0), 2)).astype(np.float32)
        H0 = R.pick_at_snap(lines, clicks, los, *orient)   # deployable: the clicks anchor the labeling
        if H0 is None:
            rec["fail"] = "register"; out.append(rec); continue
        H0, _ = R.refine(H0, img0)
        # pixel height per player from matched detections at the snap (fallback 30 px)
        m = match(clicks, Dt.feet(b0), 15) if len(b0) else np.zeros((0, 2), int)
        hmed = float(np.median(b0[m[:, 0], 3] - b0[m[:, 0], 1])) if len(m) else 30.0
        hpx = np.full(len(clicks), hmed, np.float32)

        snap_frames = {k_snap + d * SNAP_EVERY * j for d in (1, -1) for j in range(1, int(SPAN * 2) + 1)}
        boxes_at = {k: Dt.detect(det, frames[k], conf=0.08)[0] for k in sorted(snap_frames) if 0 <= k < len(frames)}
        tracks = {"lk": lk_track(frames, k_snap, clicks, hpx), "lk_snap": lk_track(frames, k_snap, clicks, hpx, boxes_at)}
        Hs = register_clip(frames, k_snap, H0, orient)

        ngs_traj = {pid: {s: (x, y) for s, x, y in g.select("step", "x_position", "y_position").iter_rows()}
                    for (pid,), g in t.group_by(["nfl_player_id"])}
        # NGS path
        Xn, Mn = to_tensors([(pos, sd, ngs_traj[pid]) for pid, pos, sd in zip(ids, s0["position"], side)], los, true_dir)
        for tag in tags:
            pn = predict(models[tag], Xn, Mn, S_all[i:i + 1])
            for c in (-10, 0, 10, 20):
                for h in ("cov", "man", "shell"):
                    rec[f"ngs_{tag}_{h}_{c}"] = int(pn[h][c - T_MIN].argmax())
        for kind, tr in tracks.items():
            field = {}
            for s, H in Hs.items():
                k = k_snap + round(s / 10 * FPS)
                if 0 <= k < len(frames) and not np.isnan(tr[k]).any():
                    field[s] = R.to_field(H, tr[k])
            for s, pf in field.items():
                truth = np.array([ngs_traj[pid].get(s, (np.nan, np.nan)) for pid in ids])
                e = np.linalg.norm(pf - truth, axis=1)
                errs += [dict(kind=kind, step=s, side=sd, err=float(v)) for v, sd in zip(e, side) if np.isfinite(v)]
            xr = np.array([(field[0][j][0] - los) * true_dir for j in range(len(ids))])
            yr = np.array([(field[0][j][1] - MID) * true_dir for j in range(len(ids))])
            roles = infer_roles(xr, yr, side)  # roles from the clicked alignment, not from NGS
            players = [(r, sd, {s: tuple(field[s][j]) for s in field}) for j, (r, sd) in enumerate(zip(roles, side))]
            Xv, Mv = to_tensors(players, los, true_dir)
            for tag in tags:
                pv = predict(models[tag], Xv, Mv, S_all[i:i + 1])
                for c in (-10, 0, 10, 20):
                    for h in ("cov", "man", "shell"):
                        rec[f"{kind}_{tag}_{h}_{c}"] = int(pv[h][c - T_MIN].argmax())
        rec["ok"] = True
        out.append(rec)
        e_snap = [x["err"] for x in errs[-200:] if x["kind"] == "lk_snap" and x["step"] == 20]
        print(i, p["game_play"], "median err +2s (lk_snap)", round(float(np.median(e_snap)), 2) if e_snap else None, flush=True)

    df = pl.DataFrame(out, infer_schema_length=None)
    E = pl.DataFrame(errs)
    df.write_parquet(io.INTERIM / "video/assisted_results.parquet"); E.write_parquet(io.INTERIM / "video/assisted_errors.parquet")
    ok = df.filter("ok")
    summ = {"plays": df.height, "ok": ok.height, "fail": df["fail"].value_counts().rows() if "fail" in df.columns else []}
    summ["position_error"] = E.group_by("kind", "step").agg(
        median=pl.col("err").median().round(3), within_05=(pl.col("err") <= 0.5).mean().round(3),
        within_1=(pl.col("err") <= 1).mean().round(3), lost_3yd=(pl.col("err") > 3).mean().round(3)).sort("kind", "step").rows()
    for kind in ("lk", "lk_snap"):
        for tag in tags:
            for c in (-10, 0, 10, 20):
                for h, y in (("man", "y_man"), ("cov", "y_cov"), ("shell", "y_shell")):
                    sub = ok.filter(pl.col(y) >= 0)
                    summ[f"{kind} {tag} {h}@{c / 10:+.1f}s"] = dict(
                        ngs=round(float((sub[f"ngs_{tag}_{h}_{c}"] == sub[y]).mean()), 3),
                        video=round(float((sub[f"{kind}_{tag}_{h}_{c}"] == sub[y]).mean()), 3), n=sub.height)
    (io.ROOT / "reports/phase2e_assisted.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
