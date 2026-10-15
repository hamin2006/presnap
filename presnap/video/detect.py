"""Player detection on wide all-22 frames. Players are ~25-35 px tall at 1080p, so frames are cut into
overlapping tiles and upscaled before detection, then merged with NMS."""
import numpy as np
import torch
from torchvision.ops import nms


def tiles(h, w, tile=640, overlap=160):
    ys = list(range(0, max(h - tile, 0) + 1, tile - overlap)) + [max(h - tile, 0)]
    xs = list(range(0, max(w - tile, 0) + 1, tile - overlap)) + [max(w - tile, 0)]
    return sorted({(y, x) for y in ys for x in xs})


def detect(model, img, imgsz=1280, conf=0.15, classes=(0,), tile=640, overlap=160):
    """-> boxes [N,4] xyxy in frame pixels, scores [N]"""
    boxes, scores = [], []
    h, w = img.shape[:2]
    crops = [(y, x, img[y:y + tile, x:x + tile]) for y, x in tiles(h, w, tile, overlap)]
    for i in range(0, len(crops), 8):
        batch = crops[i:i + 8]
        for (y, x, _), r in zip(batch, model.predict([c for _, _, c in batch], imgsz=imgsz, conf=conf,
                                                     classes=list(classes) if classes else None, verbose=False)):
            boxes.append(r.boxes.xyxy.cpu().numpy() + [x, y, x, y])
            scores.append(r.boxes.conf.cpu().numpy())
    b, s = np.concatenate(boxes), np.concatenate(scores)
    if len(b) == 0:
        return b, s
    keep = nms(torch.tensor(b, dtype=torch.float32), torch.tensor(s, dtype=torch.float32), 0.5).numpy()
    return b[keep], s[keep]


def feet(boxes):
    return np.c_[(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]]
