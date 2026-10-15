"""Phase 2D: All29 video -> player tracks -> coverage model, compared with the same model on NGS tracks.

Evaluation plays: the All29 clips that are dropbacks with an NGS coverage label (nflverse participation, 2020-21).
For each play, frames -2.0 s .. +2.0 s at 10 Hz:
  video path: register (line-based; labeling anchored on the LOS with detected players), detect players, map to the
              field, link frame to frame, split officials / offense / defense by jersey colour + side of the LOS,
              infer roles from the alignment at the snap, build the model sequence, run the StatsBomb model ensemble
  NGS path:   the same plays' NGS tracking, real positions -> same sequence format -> same models
Stand-ins for facts known in practice: LOS x from NGS linemen (= play-by-play yard line) and field orientation
(= which sideline the camera is on), both taken from NGS.

Run on the PC: uv run --group train --group cv python -m presnap.video.pipeline [--detector PATH] [--plays N]
"""
import argparse
import json

import cv2
import numpy as np
import polars as pl
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from ultralytics import YOLO

from presnap import augment, build, io, labels, normalize, seq
from presnap.model import CoverageModel
from presnap.train import DEV, TARGETS, init_from
from presnap.video import detect as Dt, register as R
from presnap.video.pseudo import FPS, V, match, split_plays

STEPS = list(range(-20, 21))
T_MIN, T_MAX = -30, 20
MID = 53.3 / 2


def eval_plays():
    import nflreadpy
    meta = pl.read_csv(V / "train_video_metadata.csv").filter(pl.col("view") == "Endzone").with_columns(
        start=pl.col("start_time").str.to_datetime(time_zone="UTC"), snap=pl.col("snap_time").str.to_datetime(time_zone="UTC"))
    sch = nflreadpy.load_schedules([2020, 2021]).select(game_key=pl.col("gsis").cast(pl.Int64), game_id=pl.col("old_game_id").cast(pl.Int64))
    sit = build.situation()
    part = io.participation().select(game_id="old_game_id", play_id="play_id", coverage=labels.canon("defense_coverage_type"))
    m = (meta.join(sch, on="game_key").join(sit, on=["game_id", "play_id"]).join(part, on=["game_id", "play_id"])
         .filter((pl.col("qb_dropback") == 1) & pl.col("coverage").is_not_null()))
    return labels.with_label_columns(m), split_plays(meta)


# ---------------------------------------------------------------- video path
def link(frames, gate=1.5):
    """frames: [(step, pts[n,2])] -> tracks: list of {step: (x, y, det_idx)}"""
    tracks = []
    for step, pts in frames:
        live = [t for t in tracks if 0 < step - max(t) <= 3]
        if live and len(pts):
            pred = []
            for t in live:
                s1 = sorted(t)[-2:]
                v = (np.array(t[s1[-1]][:2]) - np.array(t[s1[0]][:2])) / max(s1[-1] - s1[0], 1) if len(s1) == 2 else 0
                pred.append(np.array(t[s1[-1]][:2]) + v * (step - s1[-1]))
            d = np.linalg.norm(np.array(pred)[:, None] - pts[None], axis=-1)
            r, c = linear_sum_assignment(d)
            used = set()
            for i, j in zip(r, c):
                if d[i, j] <= gate * (step - max(live[i])):
                    live[i][step] = (*pts[j], j); used.add(j)
        else:
            used = set()
        tracks += [{step: (*pts[j], j)} for j in range(len(pts)) if j not in used]
    return tracks


def torso_feature(img, box):
    x1, y1, x2, y2 = box
    h, w = y2 - y1, x2 - x1
    crop = img[int(y1 + .2 * h):int(y1 + .55 * h), int(x1 + .25 * w):int(x2 - .25 * w)]
    if crop.size == 0:
        return np.zeros(4)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).mean(0)
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(float)
    stripes = np.abs(np.diff(g, axis=1)).mean() / (g.mean() + 1)  # officials' vertical stripes
    return np.r_[lab, 40 * stripes]


def sides_and_roles(pos0, feats, los):
    """pos0 [n,2] field positions at the snap, feats [n,4] -> (side per track: 'off'/'def'/None, direction +1/-1)"""
    n = len(pos0)
    km = KMeans(3, n_init=10, random_state=0).fit((feats - feats.mean(0)) / (feats.std(0) + 1e-6))
    lab = km.labels_
    sizes = np.bincount(lab, minlength=3)
    stripe = np.array([feats[lab == k, 3].mean() if sizes[k] else -1 for k in range(3)])
    ref = int(np.argmax(np.where(sizes <= 8, stripe, -1))) if (sizes <= 8).any() else -1
    teams = [k for k in range(3) if k != ref]
    med = {k: np.median(pos0[lab == k, 0] - los) if sizes[k] else 0 for k in teams}
    off = min(teams, key=lambda k: abs(med[k]))
    dfn = [k for k in teams if k != off][0]
    direction = 1 if med[dfn] > med[off] else -1
    side = np.where(lab == off, "off", np.where(lab == dfn, "def", None))
    # cap each side at the 11 players closest to the ball's lateral centre and the LOS
    for s in ("off", "def"):
        idx = np.where(side == s)[0]
        if len(idx) > 11:
            score = np.abs(pos0[idx, 0] - los) + 0.3 * np.abs(pos0[idx, 1] - np.median(pos0[idx, 1]))
            side[idx[np.argsort(score)[11:]]] = None
    return side, direction


def infer_roles(xr, yr, side):
    """Roles from the alignment at the snap in the normalized frame (x_rel past the LOS, y_rel from mid-field)."""
    role = np.array(["OTHER"] * len(xr), dtype=object)
    o = np.where(side == "off")[0]
    if len(o):
        ball_y = np.median(yr[o][np.argsort(np.abs(xr[o]))[:5]])
        line = o[np.argsort(np.abs(yr[o] - ball_y) + 3 * (np.abs(xr[o]) > 2))[:5]]
        role[line] = "OL"
        rest = [i for i in o if i not in line]
        if rest:
            qb = min(rest, key=lambda i: abs(yr[i] - ball_y) + 0.3 * abs(xr[i] + 5))
            role[qb] = "QB"
            for i in rest:
                if i == qb:
                    continue
                inside = abs(yr[i] - ball_y) < 6
                role[i] = "RB" if (xr[i] < -3 and inside) else ("TE" if inside else "WR")
    d = np.where(side == "def")[0]
    if len(d):
        ball_y = np.median(yr[o][role[o] == "OL"]) if (role[o] == "OL").any() else 0
        for i in d:
            depth, lat = xr[i], abs(yr[i] - ball_y)
            role[i] = "DL" if (depth < 2 and lat < 7) else ("LB" if (depth < 8 and lat < 8) else None)
        unassigned = [i for i in d if role[i] is None]
        deep = sorted(unassigned, key=lambda i: -xr[i])
        for i in deep:
            role[i] = "S" if (xr[i] >= 8 and abs(yr[i]) < 18) else "CB"
    return role


def video_play(det, p, los, orient):
    cap = cv2.VideoCapture(str(V / f"train/{p['game_play']}_All29.mp4"))
    t0 = (p["snap"] - p["start"]).total_seconds()
    H, frames, imgs, boxes_by_step = None, [], {}, {}
    for s in STEPS:
        cap.set(cv2.CAP_PROP_POS_FRAMES, round((t0 + s / 10) * FPS))
        ok, img = cap.read()
        lines = R.detect_lines(img) if ok else None
        if lines is None:
            continue
        b, _ = Dt.detect(det, img, classes=None if det.names.get(0) == "player" else (0,))
        f = Dt.feet(b) if len(b) else np.zeros((0, 2))
        if H is None:
            H = R.pick_at_snap(lines, f, los, *orient)
            if H is None:
                continue
        else:
            Hn = R.pick_consistent(lines, H, *orient, *img.shape[:2])
            H = Hn if Hn is not None else H
        H, _ = R.refine(H, img)
        pf = R.to_field(H, f)
        keep = (pf[:, 0] > 0) & (pf[:, 0] < 120) & (pf[:, 1] > -1) & (pf[:, 1] < 54.3)
        frames.append((s, pf[keep]))
        boxes_by_step[s] = b[keep]
        if s == 0 or (0 not in imgs and abs(s) <= 2):
            imgs[s] = img
    if not frames:
        return None
    tracks = link(frames)
    s0 = min(imgs, key=abs) if imgs else None
    at0 = [t for t in tracks if s0 in t]
    if s0 is None or len(at0) < 12:
        return None
    pos0 = np.array([t[s0][:2] for t in at0])
    feats = np.array([torso_feature(imgs[s0], boxes_by_step[s0][int(t[s0][2])]) for t in at0])
    side, direction = sides_and_roles(pos0, feats, los)
    trajs = []
    for t, sd in zip(at0, side):
        if sd is None:
            continue
        steps = np.array(sorted(t))
        xs, ys = np.array([t[k][0] for k in steps]), np.array([t[k][1] for k in steps])
        full = np.arange(steps.min(), steps.max() + 1)  # bridge short gaps linearly
        trajs.append((sd, {int(k): (x, y) for k, x, y in zip(full, np.interp(full, steps, xs), np.interp(full, steps, ys))}))
    return trajs, direction, pos0, side


def choose_orient(det, p, los, ngs0):
    """Which way the camera sees the field (venue knowledge stand-in): the orientation whose snap-frame
    registration best matches NGS."""
    cap = cv2.VideoCapture(str(V / f"train/{p['game_play']}_All29.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, round((p["snap"] - p["start"]).total_seconds() * FPS))
    ok, img = cap.read()
    lines = R.detect_lines(img) if ok else None
    if lines is None:
        return None
    b, _ = Dt.detect(det, img, classes=None if det.names.get(0) == "player" else (0,))
    f = Dt.feet(b) if len(b) else np.zeros((0, 2))
    best = None
    for o in ((1, R.W), (1, 0.0), (-1, R.W), (-1, 0.0)):
        H = R.pick_at_snap(lines, f, los, *o)
        if H is not None:
            n = len(match(R.to_field(H, f), ngs0, 2.0))
            if best is None or n > best[0]:
                best = (n, o)
    return best[1] if best else None


# ---------------------------------------------------------------- shared: rows -> model tensors
def to_tensors(players, los, direction):
    """players: list of (role, side, {step: (x, y)}) in NGS field coords -> X [1,T,P,F], M [1,T,P]"""
    T = T_MAX - T_MIN + 1
    players = sorted(players, key=lambda r: r[1] != "def")  # defenders first, like seq.build
    X = np.zeros((1, T, len(players), seq.F), np.float32)
    M = np.zeros((1, T, len(players)), bool)
    for j, (role, side, traj) in enumerate(players):
        pg = normalize.POS_GROUP.get(role, "OTHER") if role not in seq.POS else role
        k = seq.POS.index(pg) if pg in seq.POS else len(seq.POS)
        for s, (x, y) in traj.items():
            i = s - T_MIN
            if 0 <= i < T:
                X[0, i, j, 0] = (x - los) * direction / 10
                X[0, i, j, 1] = (y - MID) * direction / 10
                X[0, i, j, 4] = side == "def"
                X[0, i, j, 5 + k] = 1
                M[0, i, j] = True
    return augment.smooth(X, M), M


def load_models(tag="sb_ft", n=5, n_sit=None):
    ms = []
    for s in range(n):
        m = CoverageModel(seq.F, n_sit, t_max=T_MAX - T_MIN + 1).to(DEV)
        init_from(m, io.INTERIM / f"runs/{tag}/seed{s}.pt")
        ms.append(m.eval())
    return ms


@torch.no_grad()
def predict(models, X, M, S):
    Xt, Mt, St = (torch.from_numpy(a).to(DEV) for a in (X, M, S))
    out = [m(Xt, Mt, St) for m in models]
    return {h: torch.stack([o[h].softmax(-1) for o in out]).mean(0)[0].cpu().numpy() for h in TARGETS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", default="yolo11x.pt")
    ap.add_argument("--plays", type=int, default=None)
    ap.add_argument("--models", default="sb_ft")
    a = ap.parse_args()
    plays, held_out = eval_plays()
    if a.plays:
        plays = plays.head(a.plays)
    S_all = seq.situation(plays)
    models = load_models(a.models, n_sit=S_all.shape[1])
    det = YOLO(a.detector)
    trk = pl.read_csv(V / "train_player_tracking.csv").filter(pl.col("step").is_between(T_MIN, T_MAX))
    out = []
    for i, p in enumerate(plays.iter_rows(named=True)):
        t = trk.filter(pl.col("game_play") == p["game_play"])
        s0 = t.filter(pl.col("step") == 0)
        ol = s0.filter(pl.col("position").is_in(["C", "G", "T", "OG", "OT"]))
        dl = s0.filter(pl.col("position").is_in(["DE", "DT", "NT", "DL"]))
        if ol.height == 0 or dl.height == 0:
            continue
        los = (ol["x_position"].mean() + dl["x_position"].mean()) / 2
        offense_home = (ol["team"][0] == "home")
        true_dir = 1 if dl["x_position"].mean() > ol["x_position"].mean() else -1
        # NGS path
        ngs_players = []
        for (pid,), g in t.group_by(["nfl_player_id"]):
            side = "off" if (g["team"][0] == "home") == offense_home else "def"
            ngs_players.append((g["position"][0], side, {s: (x, y) for s, x, y in g.select("step", "x_position", "y_position").iter_rows()}))
        Xn, Mn = to_tensors(ngs_players, los, true_dir)
        pn = predict(models, Xn, Mn, S_all[i:i + 1])
        rec = dict(game_play=p["game_play"], held_out=p["game_play"] in held_out, coverage=p["coverage"],
                   y_cov=labels.COVERAGE7.index(p["coverage"]) if p["coverage"] in labels.COVERAGE7 else -1,
                   y_man=int(p["man_zone"] == "MAN"), y_shell=p["shell_played"])
        for c in (-10, 0, 10, 20):
            for h in ("cov", "man", "shell"):
                rec[f"ngs_{h}_{c}"] = int(pn[h][c - T_MIN].argmax())
        ngs0 = s0.select("x_position", "y_position").to_numpy()
        orient = choose_orient(det, p, los, ngs0)
        res = video_play(det, p, los, orient) if orient else None
        if res is None:
            out.append(rec); continue
        trajs, vdir, pos0, side = res
        xs0 = [tr.get(0, tr[min(tr, key=abs)]) for _, tr in trajs]
        xr = np.array([(x - los) * vdir for x, _ in xs0])
        yr = np.array([(y - MID) * vdir for _, y in xs0])
        sd = np.array([s_ for s_, _ in trajs], dtype=object)
        roles = infer_roles(xr, yr, sd)
        Xv, Mv = to_tensors([(r, s_, tr) for r, (s_, tr) in zip(roles, trajs)], los, vdir)
        pv = predict(models, Xv, Mv, S_all[i:i + 1])
        mm = match(pos0, ngs0, 2.0)
        rec.update(video_ok=True, video_dir_correct=vdir == true_dir, n_video_players=int((sd != None).sum()),
                   snap_pos_err=float(np.median(np.linalg.norm(pos0[mm[:, 1]] - ngs0[mm[:, 0]], axis=1))) if len(mm) else None,
                   snap_recall_2yd=len(mm) / len(ngs0))
        for c in (-10, 0, 10, 20):
            for h in ("cov", "man", "shell"):
                rec[f"video_{h}_{c}"] = int(pv[h][c - T_MIN].argmax())
        out.append(rec)
        print(i, p["game_play"], {k: v for k, v in rec.items() if k in ("snap_pos_err", "snap_recall_2yd", "video_dir_correct")}, flush=True)

    df = pl.DataFrame(out, infer_schema_length=None)
    df.write_parquet(io.INTERIM / "video/pipeline_results.parquet")
    summ = {"plays": df.height, "video_ok": int(df["video_ok"].fill_null(False).sum()),
            "direction_correct": float(df["video_dir_correct"].fill_null(False).mean()),
            "snap_pos_err_median": float(df["snap_pos_err"].median()), "snap_recall_2yd": float(df["snap_recall_2yd"].mean())}
    ok = df.filter(pl.col("video_ok").fill_null(False))
    for c in (-10, 0, 10, 20):
        for h, y in (("man", "y_man"), ("cov", "y_cov"), ("shell", "y_shell")):
            sub = ok.filter(pl.col(y) >= 0)
            summ[f"{h}@{c / 10:+.1f}s"] = dict(ngs=round(float((sub[f"ngs_{h}_{c}"] == sub[y]).mean()), 3),
                                               video=round(float((sub[f"video_{h}_{c}"] == sub[y]).mean()), 3),
                                               agree=round(float((sub[f"ngs_{h}_{c}"] == sub[f"video_{h}_{c}"]).mean()), 3), n=sub.height)
    (io.ROOT / f"reports/phase2d_pipeline_{a.models}.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
