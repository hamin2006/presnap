"""Remaining never-registered plays: draw detected yard lines / sidelines and report the best NGS-assisted labeling."""
import cv2
import numpy as np
import polars as pl
from ultralytics import YOLO

from presnap import io
from presnap.video import detect as Dt, register as R
from presnap.video.pseudo import FPS, V, oracle_snap

r = pl.read_parquet(io.ROOT / "data/interim/video/registration.parquet")
bad = r.group_by("game_play").agg(pl.col("registered").sum().alias("n")).filter(pl.col("n") == 0)["game_play"].sort().to_list()
meta = pl.read_csv(V / "train_video_metadata.csv").filter(pl.col("view") == "Endzone").with_columns(
    start=pl.col("start_time").str.to_datetime(time_zone="UTC"), snap=pl.col("snap_time").str.to_datetime(time_zone="UTC"))
trk = pl.read_csv(V / "train_player_tracking.csv").filter(pl.col("step") == 0)
det = YOLO("yolo11x.pt")
for gp in bad[::8]:
    p = meta.filter(pl.col("game_play") == gp).row(0, named=True)
    cap = cv2.VideoCapture(str(V / f"train/{gp}_All29.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, round((p["snap"] - p["start"]).total_seconds() * FPS))
    ok, img = cap.read()
    lines = R.detect_lines(img)
    b, _ = Dt.detect(det, img)
    f = Dt.feet(b) if len(b) else np.zeros((0, 2))
    truth = trk.filter(pl.col("game_play") == gp).select("x_position", "y_position").to_numpy()
    best = oracle_snap(lines, f, truth) if lines else None
    print(gp, "dets", len(f), "yard", len(lines[0]) if lines else 0, "flat", len(lines[1]) if lines else 0,
          "pairs", len(list(R._pairs(lines))) if lines else 0, "best matches", best[0] if best else None)
    vis = img.copy()
    if lines:
        yard, flat, _, _ = lines
        for l, col in [(l, (0, 255, 0)) for l in yard] + [(l, (0, 0, 255)) for l in flat]:
            a, bb, c = l
            pts = [(int(x), int(-(a * x + c) / bb)) for x in (0, 1919)] if abs(bb) > abs(a) else [(int(-(bb * y + c) / a), int(y)) for y in (0, 1079)]
            cv2.line(vis, pts[0], pts[1], col, 2)
    for x1, y1, x2, y2 in b:
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 1)
    cv2.imwrite(f"/tmp/presnap_frames/fail2_{gp}.jpg", cv2.resize(vis, (960, 540)), [cv2.IMWRITE_JPEG_QUALITY, 80])
