"""Score the hand-labelled snap frames (data/review/click_labels.json from the click tool).

Per frame:
  ground truth  H_gt from the 4 landmark clicks (yard line x sideline-edge points); orientation (which way NGS x/y
                run in the image) is the one whose mapped player clicks best match NGS (Hungarian)
  label check   player clicks mapped through H_gt vs NGS at the snap -> how good the labels are
  registration  our automatic line-based registration (LOS-anchored on the clicks) vs H_gt, in yards on the players
  tracking      LK tracks from the clicks, mapped with H_gt carried frame to frame (relabel), vs NGS at +/-1 s, +/-2 s

Run on the PC: uv run --group cv python -m presnap.video.click_eval
"""
import json

import cv2
import numpy as np
import polars as pl
from scipy.optimize import linear_sum_assignment

from presnap import io
from presnap.video import assisted as A, register as R
from presnap.video.click_tool import LINES

LABELS = io.ROOT / "data/review/click_labels.json"


def landmark_field(m, x_left_to_right, near_is_zero):
    x = 10 + 5 * LINES.index(m["line"])  # 10 (left goal line) .. 110 (right goal line), image left to right
    x = x if x_left_to_right else 120 - x
    near = 0.0 if near_is_zero else R.W
    return x, (near if m["edge"] == "near" else R.W - near)


def hungarian(a, b):
    d = np.linalg.norm(a[:, None] - b[None], axis=-1)
    r, c = linear_sum_assignment(d)
    return r, c, d[r, c]


def main():
    labels = json.loads(LABELS.read_text())
    plays, _ = A.eval_plays()
    trk = pl.read_csv(A.V / "train_player_tracking.csv").filter(pl.col("step").is_between(-20, 20))
    rows, track_rows = [], []
    for gp, lab in labels.items():
        if len(lab["landmarks"]) < 4 or len(lab["players"]) < 15:
            continue
        p = plays.filter(pl.col("game_play") == gp).row(0, named=True)
        t = trk.filter(pl.col("game_play") == gp)
        s0 = t.filter(pl.col("step") == 0).sort("nfl_player_id")
        ngs0 = s0.select("x_position", "y_position").to_numpy()
        clicks = np.array(lab["players"], np.float32)
        img_pts = np.array([(m["x"], m["y"]) for m in lab["landmarks"]], np.float32)
        best = None
        for lr in (True, False):
            for nz in (True, False):
                fld = np.array([landmark_field(m, lr, nz) for m in lab["landmarks"]], np.float32)
                H, _ = cv2.findHomography(fld, img_pts)
                if H is None:
                    continue
                r, c, d = hungarian(R.to_field(H, clicks), ngs0)
                if best is None or np.median(d) < best[0]:
                    best = (np.median(d), H, (lr, nz), r, c, d)
        _, H_gt, orient, r, c, d = best
        rec = dict(game_play=gp, n_clicks=len(clicks), label_err_median=float(np.median(d)), label_within_1yd=float((d <= 1).mean()))
        # our automatic registration, anchored on the clicked players (the coach-assisted setting)
        ol = s0.filter(pl.col("position").is_in(["C", "G", "T", "OG", "OT"]))["x_position"].mean()
        dl = s0.filter(pl.col("position").is_in(["DE", "DT", "NT", "DL"]))["x_position"].mean()
        los = (ol + dl) / 2
        frames, k0 = A.read_clip(p)
        lines = R.detect_lines(frames[k0]) if frames else None
        direction = 1 if orient[0] else -1
        y_far = R.W if orient[1] else 0.0
        H_auto = R.pick_at_snap(lines, clicks, los, direction, y_far) if lines else None
        if H_auto is not None:
            H_auto, _ = R.refine(H_auto, frames[k0])
            e = np.linalg.norm(R.to_field(H_auto, clicks) - R.to_field(H_gt, clicks), axis=1)
            rec.update(auto_reg_err_median=float(np.median(e)), auto_reg_err_p90=float(np.quantile(e, .9)))
        rows.append(rec)
        # tracking from the clicks, registration carried from H_gt
        if frames:
            ids = s0["nfl_player_id"].to_numpy()[c]          # NGS identity of each matched click
            tr = A.lk_track(frames, k0, clicks[r], np.full(len(r), 30.0, np.float32))
            Hs = A.register_clip(frames, k0, H_gt, (direction, y_far))
            for s in (-20, -10, 10, 20):
                k = k0 + round(s / 10 * A.FPS)
                if s not in Hs or not 0 <= k < len(frames):
                    continue
                truth = np.array([t.filter((pl.col("step") == s) & (pl.col("nfl_player_id") == i)).select("x_position", "y_position").row(0)
                                  for i in ids])
                e = np.linalg.norm(R.to_field(Hs[s], tr[k]) - truth, axis=1)
                track_rows += [dict(game_play=gp, step=s, err=float(v)) for v in e]
        print(rows[-1], flush=True)
    df, tk = pl.DataFrame(rows), pl.DataFrame(track_rows)
    summ = dict(frames=df.height,
                label_err_median=round(float(df["label_err_median"].median()), 3),
                label_within_1yd=round(float(df["label_within_1yd"].mean()), 3),
                auto_registration_err_median=round(float(df["auto_reg_err_median"].median()), 3) if "auto_reg_err_median" in df.columns else None,
                auto_registration_within_1yd_frames=round(float((df["auto_reg_err_median"] <= 1).mean()), 3) if "auto_reg_err_median" in df.columns else None,
                tracking=tk.group_by("step").agg(pl.col("err").median().round(3), (pl.col("err") <= 1).mean().round(3).alias("within_1yd"),
                                                 (pl.col("err") > 3).mean().round(3).alias("lost_3yd")).sort("step").rows() if tk.height else [])
    (io.ROOT / "reports/phase2_click_eval.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
