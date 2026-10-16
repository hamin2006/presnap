"""Why do whole plays fail to register? Line-detector diagnostics on snap frames of failing plays."""
import cv2
import numpy as np
import polars as pl

from presnap import io
from presnap.video import register as R
from presnap.video.pseudo import FPS, V

r = pl.read_parquet(io.ROOT / "data/interim/video/registration.parquet")
bad = r.group_by("game_play").agg(pl.col("registered").sum().alias("n")).filter(pl.col("n") == 0)["game_play"].sort().to_list()
meta = pl.read_csv(V / "train_video_metadata.csv").filter(pl.col("view") == "Endzone").with_columns(
    start=pl.col("start_time").str.to_datetime(time_zone="UTC"), snap=pl.col("snap_time").str.to_datetime(time_zone="UTC"))
stats = []
for gp in bad[::6]:
    p = meta.filter(pl.col("game_play") == gp).row(0, named=True)
    cap = cv2.VideoCapture(str(V / f"train/{gp}_All29.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, round((p["snap"] - p["start"]).total_seconds() * FPS))
    ok, img = cap.read()
    segs = R._lines(img)
    h = img.shape[0]
    yard, flat = R._group(segs, True, h), R._group(segs, False, h)
    stats.append((gp, len(segs), len(yard), len(flat)))
    if len(stats) <= 4:
        cv2.imwrite(f"/tmp/presnap_frames/fail_{gp}.jpg", cv2.resize(img, (960, 540)), [cv2.IMWRITE_JPEG_QUALITY, 80])
        cv2.imwrite(f"/tmp/presnap_frames/fail_{gp}_white.jpg", cv2.resize(R.white_mask(img), (960, 540)))
for s in stats:
    print(s)
