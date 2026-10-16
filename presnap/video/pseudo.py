"""Phase 2B/2C: register every All29 play with an NGS-assisted labeling, measure registration accuracy, and write
pseudo-labelled detector training tiles.

Per play, frames from -1.0 s to +3.0 s every 0.5 s:
  1. detect lines (register.detect_lines); detect people with a pretrained COCO model on tiles
  2. at the snap: choose yard-line labeling + orientation that best matches NGS (allowed here: these are training
     labels; the deployable choice without NGS is evaluated separately), then refine on the painted lines
  3. later frames: labeling most consistent with the previous frame, refined
  4. registration accuracy = distance (yards) between detected feet mapped to the field and matched NGS players
  5. pseudo boxes: NGS feet projected into the image; box height from the matched detections' height per
     pixel-per-yard; written as YOLO tiles for plays outside the 40 evaluation plays

Run on the PC: uv run --group cv python -m presnap.video.pseudo
"""
import json

import cv2
import numpy as np
import polars as pl
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

from presnap import io
from presnap.video import detect as Dt, register as R

V = io.ROOT / "data/raw/video/nfl-player-contact-detection"
OUT = io.ROOT / "data/interim/video"
FPS = 60000 / 1001
STEPS = list(range(-10, 31, 5))
DET_STEPS = {-10, 0, 20}  # detection (labeling evidence, accuracy, box size) on these; registration carried between
EVAL_SEED, N_EVAL = 0, 40


def match(pred, truth, max_d):
    if len(pred) == 0 or len(truth) == 0:
        return np.zeros((0, 2), int)
    d = np.linalg.norm(truth[:, None] - pred[None], axis=-1)
    r, c = linear_sum_assignment(d)
    k = d[r, c] <= max_d
    return np.c_[r[k], c[k]]


def project(H, xy):
    return cv2.perspectiveTransform(np.asarray(xy, np.float32)[None], H)[0]


def px_per_yd(H, xy):
    a, b = project(H, xy), project(H, np.asarray(xy) + [1.0, 0.0])
    return np.linalg.norm(b - a, axis=1)


def oracle_snap(lines, feet_px, truth):
    best = None
    for direction in (1, -1):
        for y_far in (R.W, 0.0):
            for _, H in R.candidates(lines, direction, y_far):
                m = match(R.to_field(H, feet_px), truth, 1.5)
                if best is None or len(m) > best[0]:
                    best = (len(m), H, (direction, y_far))
    return best


def split_plays(meta):
    ev = meta.sample(N_EVAL, seed=EVAL_SEED)["game_play"].to_list()
    return set(ev)


def main(limit=None):
    meta = pl.read_csv(V / "train_video_metadata.csv").filter(pl.col("view") == "Endzone").with_columns(
        start=pl.col("start_time").str.to_datetime(time_zone="UTC"), snap=pl.col("snap_time").str.to_datetime(time_zone="UTC"))
    eval_plays = split_plays(meta)
    if limit:
        meta = meta.head(int(limit))
    trk = pl.read_csv(V / "train_player_tracking.csv").filter(pl.col("step").is_between(min(STEPS), max(STEPS)))
    det_model = YOLO("yolo11x.pt")
    (OUT / "tiles/images").mkdir(parents=True, exist_ok=True); (OUT / "tiles/labels").mkdir(parents=True, exist_ok=True)
    rows, sync = [], []
    for k, p in enumerate(meta.iter_rows(named=True)):
        gp = p["game_play"]
        cap = cv2.VideoCapture(str(V / f"train/{gp}_All29.mp4"))
        t = trk.filter(pl.col("game_play") == gp)
        t0 = (p["snap"] - p["start"]).total_seconds()
        H, orient, ratio = None, None, []
        for s in STEPS:
            cap.set(cv2.CAP_PROP_POS_FRAMES, round((t0 + s / 10) * FPS))
            ok, img = cap.read()
            truth_df = t.filter(pl.col("step") == s)
            truth = truth_df.select("x_position", "y_position").to_numpy()
            rec = dict(game_play=gp, step=s, eval=gp in eval_plays, registered=False)
            lines = R.detect_lines(img) if ok else None
            if lines is None or not len(truth) or (H is None and s not in DET_STEPS):
                rows.append(rec); continue  # labeling is only attempted on detection frames (caps GPU time per play)
            if s in DET_STEPS or H is None:
                boxes, _ = Dt.detect(det_model, img, conf=0.08)  # low threshold: NGS filters the false positives
            else:
                boxes = np.zeros((0, 4))
            fpx = Dt.feet(boxes) if len(boxes) else np.zeros((0, 2))
            if H is None:  # first usable frame: NGS-assisted labeling and orientation
                first = oracle_snap(lines, fpx, truth)
                if first is None or first[0] < 6:
                    rows.append(rec); continue
                _, H, orient = first
            else:
                Hn = R.pick_consistent(lines, H, *orient, *img.shape[:2])
                H = Hn if Hn is not None else H
            H, resid_px = R.refine(H, img)
            pf = R.to_field(H, fpx)
            m = match(pf, truth, 2.0)
            err = np.linalg.norm(pf[m[:, 1]] - truth[m[:, 0]], axis=1) if len(m) else np.zeros(0)
            if s == 0 and len(fpx):  # video/tracking sync: does a time shift explain the residual?
                sync.append(dict(game_play=gp, **{f"off_{o}": float(np.median(np.linalg.norm(
                    R.to_field(H, fpx)[mm[:, 1]] - t.filter(pl.col("step") == o).select("x_position", "y_position").to_numpy()[mm[:, 0]], axis=1)))
                    if len(mm := match(R.to_field(H, fpx), t.filter(pl.col("step") == o).select("x_position", "y_position").to_numpy(), 2.0)) else None
                    for o in (-3, -2, -1, 0, 1, 2, 3)}))
            ppy = px_per_yd(H, truth)
            if len(m):
                bh = boxes[m[:, 1], 3] - boxes[m[:, 1], 1]
                ratio += list(bh / ppy[m[:, 0]])
            rec.update(registered=True, resid_px=resid_px, n_det=len(fpx), n_match=len(m), detected=len(fpx) > 0,
                       median_err=float(np.median(err)) if len(err) else None,
                       within_1yd=float((err <= 1).sum() / len(truth)), within_05yd=float((err <= 0.5).sum() / len(truth)))
            rows.append(rec)
            if gp not in eval_plays and s in DET_STEPS and len(m):
                # labels = the detector's own (pixel-accurate) boxes that NGS confirms as players within 1 yd;
                # NGS players the detector missed are greyed out so they aren't taught as background
                conf = m[np.linalg.norm(pf[m[:, 1]] - truth[m[:, 0]], axis=1) <= 1.0]
                pos = boxes[conf[:, 1]]
                missed = np.setdiff1d(np.arange(len(truth)), conf[:, 0])
                hpx = (np.median(ratio) if ratio else 2.0) * ppy[missed]
                fx = project(H, truth[missed])
                img_l = img.copy()
                for (x, y), hh in zip(fx, hpx):
                    cv2.rectangle(img_l, (int(x - .35 * hh), int(y - 1.15 * hh)), (int(x + .35 * hh), int(y + .15 * hh)), (110, 110, 110), -1)
                for (ty, tx) in Dt.tiles(*img.shape[:2]):
                    inside = (pos[:, 2] > tx) & (pos[:, 0] < tx + 640) & (pos[:, 3] > ty) & (pos[:, 1] < ty + 640)
                    if not inside.any():
                        continue
                    b = pos[inside] - [tx, ty, tx, ty]
                    cl = np.clip(b, 0, 640)
                    keep = (cl[:, 2] - cl[:, 0]) * (cl[:, 3] - cl[:, 1]) >= 0.5 * (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
                    name = f"{gp}_s{s:+d}_{ty}_{tx}"
                    cv2.imwrite(str(OUT / f"tiles/images/{name}.jpg"), img_l[ty:ty + 640, tx:tx + 640], [cv2.IMWRITE_JPEG_QUALITY, 92])
                    with open(OUT / f"tiles/labels/{name}.txt", "w") as f:
                        for x1, y1, x2, y2 in cl[keep]:
                            f.write(f"0 {(x1 + x2) / 1280:.5f} {(y1 + y2) / 1280:.5f} {(x2 - x1) / 640:.5f} {(y2 - y1) / 640:.5f}\n")
        print(k, gp, [(r["step"], round(r.get("median_err") or -1, 2), round(r.get("within_1yd", 0), 2)) for r in rows[-len(STEPS):]], flush=True)

    df = pl.DataFrame(rows, infer_schema_length=None)
    df.write_parquet(OUT / "registration.parquet")
    pl.DataFrame(sync, infer_schema_length=None).write_parquet(OUT / "sync.parquet")
    reg = df.filter("registered")
    summ = dict(frames=df.height, registered=round(reg.height / df.height, 3),
                median_err_yd=round(float(reg["median_err"].drop_nulls().median()), 3), p90_err_yd=round(float(reg["median_err"].drop_nulls().quantile(.9)), 3),
                median_resid_px=round(float(reg["resid_px"].median()), 2),
                sync_median_err_by_offset={c: round(float(pl.DataFrame(sync, infer_schema_length=None)[c].median()), 3)
                                           for c in pl.DataFrame(sync, infer_schema_length=None).columns if c.startswith("off_")},
                by_step=reg.filter("detected").group_by("step").agg(pl.col("median_err").median().round(3), pl.col("n_match").mean().round(1), pl.len()).sort("step").rows())
    (io.ROOT / "reports/phase2b_oracle_registration.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])
