"""Phase 2C: fine-tune a YOLO player detector on the NGS-projected pseudo-labels (presnap.video.pseudo).

Only on-field players are labelled, so sideline people, coaches and officials are learned as background.
Tiles are split by play (10% of the training plays for validation); the 40 evaluation plays never appear.

Run on the PC: uv run --group cv python -m presnap.video.train_det [epochs] [model]
"""
import random
import sys

from ultralytics import YOLO

from presnap import io

T = io.ROOT / "data/interim/video/tiles"


def main(epochs=20, base="yolo11s.pt"):
    imgs = sorted((T / "images").glob("*.jpg"))
    plays = sorted({p.name.split("_s")[0] for p in imgs})
    random.Random(0).shuffle(plays)
    val = set(plays[: max(1, len(plays) // 10)])
    (T / "train.txt").write_text("\n".join(str(p) for p in imgs if p.name.split("_s")[0] not in val))
    (T / "val.txt").write_text("\n".join(str(p) for p in imgs if p.name.split("_s")[0] in val))
    (T / "data.yaml").write_text(f"path: {T}\ntrain: train.txt\nval: val.txt\nnames:\n  0: player\n")
    print(f"{len(plays)} plays, {len(val)} for validation, {len(imgs)} tiles")
    YOLO(base).train(data=str(T / "data.yaml"), epochs=int(epochs), imgsz=960, batch=8, patience=5, workers=6,
                     project=str(io.ROOT / "data/interim/video/det"), name="player", exist_ok=True,
                     fliplr=0.5, mosaic=1.0, scale=0.3, degrees=0, hsv_h=0.02, close_mosaic=3, seed=0, plots=False)


if __name__ == "__main__":
    main(*sys.argv[1:])
