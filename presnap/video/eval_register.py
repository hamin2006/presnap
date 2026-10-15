"""Phase 2B/2C gate: from All29 video alone, how close do detected players land to their NGS positions?

Per play: frames at the snap and +0.5/1.0/1.5/2.0 s. Detect people (pretrained YOLO, COCO 'person'), take the
bottom-centre of each box as the feet, register the frame (presnap.video.register), map feet to the field, and
match them to the 22 NGS positions (Hungarian, yards).

Evaluation-only shortcuts, both stand-ins for facts known in practice:
  * LOS x = midpoint of the NGS linemen (in practice: the play-by-play yard line)
  * field orientation (which end/sideline is which) is picked once per play at the snap using NGS
    (in practice: fixed per venue / camera position)

Run on the PC: uv run --group cv python -m presnap.video.eval_register [--plays 40]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import polars as pl
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

from presnap import io
from presnap.video import register as R

V = io.ROOT / "data/raw/video/nfl-player-contact-detection"
OUT = io.ROOT / "data/review/register"
FPS = 60000 / 1001
STEPS = [0, 5, 10, 15, 20]


def frame(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, img = cap.read()
    return img if ok else None


def feet_of(model, img):
    r = model.predict(img, imgsz=1920, conf=0.2, classes=[0], verbose=False)[0]
    b = r.boxes.xyxy.cpu().numpy()
    return np.c_[(b[:, 0] + b[:, 2]) / 2, b[:, 3]], b


def match(pred, truth, max_d):
    if len(pred) == 0:
        return np.zeros((0, 2)), np.zeros(0, int)
    d = np.linalg.norm(truth[:, None] - pred[None], axis=-1)
    r, c = linear_sum_assignment(d)
    keep = d[r, c] <= max_d
    return (pred[c] - truth[r])[keep], r[keep]


def on_field(p):
    return p[(p[:, 0] > -2) & (p[:, 0] < 122) & (p[:, 1] > -2) & (p[:, 1] < R.W + 2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plays", type=int, default=40)
    a = ap.parse_args()
    meta = pl.read_csv(V / "train_video_metadata.csv").filter(pl.col("view") == "Endzone").with_columns(
        start=pl.col("start_time").str.to_datetime(time_zone="UTC"), snap=pl.col("snap_time").str.to_datetime(time_zone="UTC"))
    trk = pl.read_csv(V / "train_player_tracking.csv").filter(pl.col("step").is_between(0, 20))
    plays = meta.sample(a.plays, seed=0)
    model = YOLO("yolo11m.pt")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for k, p in enumerate(plays.iter_rows(named=True)):
        gp = p["game_play"]
        cap = cv2.VideoCapture(str(V / f"train/{gp}_All29.mp4"))
        t = trk.filter(pl.col("game_play") == gp)
        s0 = t.filter(pl.col("step") == 0)
        ol = s0.filter(pl.col("position").is_in(["C", "G", "T", "OG", "OT"]))["x_position"].mean()
        dl = s0.filter(pl.col("position").is_in(["DE", "DT", "NT", "DL"]))["x_position"].mean()
        los = (ol + dl) / 2 if ol is not None and dl is not None else (ol or dl or s0["x_position"].median())
        H, orient = None, None
        for s in STEPS:
            img = frame(cap, round(((p["snap"] - p["start"]).total_seconds() + s / 10) * FPS))
            truth = t.filter(pl.col("step") == s).select("x_position", "y_position").to_numpy()
            rec = dict(game_play=gp, step=s, registered=False)
            lines = R.detect_lines(img) if img is not None else None
            if lines is not None and len(truth):
                feet, boxes = feet_of(model, img)
                if s == 0:
                    best = None
                    for direction in (1, -1):
                        for y_far in (R.W, 0.0):
                            Hc = R.pick_at_snap(lines, feet, los, direction, y_far)
                            if Hc is None:
                                continue
                            err, _ = match(on_field(R.to_field(Hc, feet)), truth, 3.0)
                            score = (len(err), -np.median(np.linalg.norm(err, axis=1)) if len(err) else 0)
                            if best is None or score > best[0]:
                                best = (score, Hc, (direction, y_far))
                    if best:
                        _, H, orient = best
                elif H is not None:
                    Hn = R.pick_consistent(lines, H, *orient, *img.shape[:2])
                    H = Hn if Hn is not None else H
                if H is not None:
                    pf = on_field(R.to_field(H, feet))
                    err, _ = match(pf, truth, 3.0)
                    wide, _ = match(pf, truth, 7.0)
                    dist = np.linalg.norm(err, axis=1)
                    rec.update(registered=True, n_det=len(pf), n_truth=len(truth),
                               median_err=float(np.median(dist)) if len(dist) else None,
                               recall_1yd=float((dist <= 1).sum() / len(truth)), recall_2yd=float((dist <= 2).sum() / len(truth)),
                               alias=bool(len(wide) >= 5 and abs(np.median(wide[:, 0])) > 3))
                    if k < 6 and s in (0, 20):
                        vis = img.copy()
                        for x, y in cv2.perspectiveTransform(truth.astype(np.float32)[None], H)[0]:
                            cv2.circle(vis, (int(x), int(y)), 9, (0, 255, 255), 2)
                        for x1, y1, x2, y2 in boxes:
                            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 1)
                        cv2.imwrite(str(OUT / f"{gp}_s{s}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 75])
            rows.append(rec)
        print(gp, [(r["step"], round(r.get("median_err") or -1, 2), round(r.get("recall_2yd", 0), 2)) for r in rows[-len(STEPS):]], flush=True)

    df = pl.DataFrame(rows, infer_schema_length=None)
    reg = df.filter("registered")
    summ = dict(plays=a.plays, frames=df.height, registered=round(reg.height / df.height, 3),
                median_err_yd=round(float(reg["median_err"].median()), 3),
                p90_err_yd=round(float(reg["median_err"].quantile(0.9)), 3),
                recall_1yd=round(float(reg["recall_1yd"].mean()), 3), recall_2yd=round(float(reg["recall_2yd"].mean()), 3),
                alias_rate=round(float(reg["alias"].mean()), 3),
                by_step=reg.group_by("step").agg(pl.col("median_err").median().round(3), pl.col("recall_2yd").mean().round(3),
                                                 pl.col("alias").mean().round(3), pl.len()).sort("step").rows())
    (io.ROOT / "reports/phase2b_register.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
