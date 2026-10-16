"""Detector choice: recall of on-field NGS players at the snap (within 1.5 yd after registration) for the
fine-tuned detector at several confidence thresholds vs the COCO detector. 20 held-out evaluation plays."""
import cv2
import numpy as np
import polars as pl
from ultralytics import YOLO

from presnap.video import detect as Dt, pipeline as P, register as R
from presnap.video.pseudo import match

plays, held = P.eval_plays()
plays = plays.filter(pl.col("game_play").is_in(list(held))).head(20)
trk = pl.read_csv(P.V / "train_player_tracking.csv").filter(pl.col("step") == 0)
ft, coco = YOLO("data/interim/video/det/player/weights/best.pt"), YOLO("yolo11x.pt")
configs = [("ft", 0.02), ("ft", 0.05), ("ft", 0.1), ("coco", 0.08), ("coco", 0.15)]
res = {c: [] for c in configs}
for p in plays.iter_rows(named=True):
    s0 = trk.filter(pl.col("game_play") == p["game_play"])
    ngs0 = s0.select("x_position", "y_position").to_numpy()
    los = (s0.filter(pl.col("position").is_in(["C", "G", "T", "OG", "OT"]))["x_position"].mean()
           + s0.filter(pl.col("position").is_in(["DE", "DT", "NT", "DL"]))["x_position"].mean()) / 2
    cap = cv2.VideoCapture(str(P.V / f"train/{p['game_play']}_All29.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, round((p["snap"] - p["start"]).total_seconds() * P.FPS))
    ok, img = cap.read()
    lines = R.detect_lines(img)
    if lines is None:
        continue
    for name, conf in configs:
        m = ft if name == "ft" else coco
        b, _ = Dt.detect(m, img, conf=conf, classes=None if name == "ft" else (0,))
        f = Dt.feet(b) if len(b) else np.zeros((0, 2))
        best = 0
        for o in ((1, R.W), (1, 0.0), (-1, R.W), (-1, 0.0)):
            H = R.pick_at_snap(lines, f, los, *o)
            if H is not None:
                best = max(best, len(match(R.to_field(H, f), ngs0, 1.5)))
        res[(name, conf)].append((len(b), best / len(ngs0)))
for c, v in res.items():
    a = np.array(v)
    print(c, "boxes", np.median(a[:, 0]), "recall@1.5yd", a[:, 1].mean().round(3), "plays", len(a))
