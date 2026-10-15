"""Phase 2C exploration: does tiled, upscaled inference recover the small, packed players a full-frame
COCO detector misses? Counts detections whose feet project onto the field near NGS players."""
import sys

import cv2
import numpy as np
import polars as pl
import torch
from ultralytics import YOLO
from torchvision.ops import nms

from presnap import io

V = io.ROOT / "data/raw/video/nfl-player-contact-detection"
FPS = 60000 / 1001


def tiled(model, img, tile=640, overlap=160, imgsz=1280, conf=0.15):
    h, w = img.shape[:2]
    boxes, scores = [], []
    for y in range(0, h - overlap, tile - overlap):
        for x in range(0, w - overlap, tile - overlap):
            crop = img[y:y + tile, x:x + tile]
            r = model.predict(crop, imgsz=imgsz, conf=conf, classes=[0], verbose=False)[0]
            b = r.boxes.xyxy.cpu().numpy() + [x, y, x, y]
            boxes.append(b); scores.append(r.boxes.conf.cpu().numpy())
    b, s = np.concatenate(boxes), np.concatenate(scores)
    keep = nms(torch.tensor(b), torch.tensor(s), 0.5).numpy()
    return b[keep]


def main(gp="58177_004239"):
    meta = pl.read_csv(V / "train_video_metadata.csv").filter((pl.col("game_play") == gp) & (pl.col("view") == "Endzone"))
    t = (pl.Series([meta["snap_time"][0]]).str.to_datetime() - pl.Series([meta["start_time"][0]]).str.to_datetime())[0].total_seconds()
    cap = cv2.VideoCapture(str(V / f"train/{gp}_All29.mp4")); cap.set(cv2.CAP_PROP_POS_FRAMES, round(t * FPS)); _, img = cap.read()
    for name in ("yolo11m.pt", "yolo11x.pt"):
        m = YOLO(name)
        full = m.predict(img, imgsz=1920, conf=0.15, classes=[0], verbose=False)[0].boxes.xyxy.cpu().numpy()
        til = tiled(m, img)
        print(f"{name}: full-frame {len(full)} boxes, tiled {len(til)} boxes")
        vis = img.copy()
        for x1, y1, x2, y2 in til:
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 1)
        cv2.imwrite(f"/tmp/presnap_frames/{gp}_tiled_{name[:-3]}.jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 80])


if __name__ == "__main__":
    main(*sys.argv[1:])
