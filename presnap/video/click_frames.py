"""Extract snap frames for the click tool (20 labelled All29 plays) -> data/review/click/frames/."""
import json

import cv2
import polars as pl

from presnap import io
from presnap.video.pipeline import eval_plays
from presnap.video.pseudo import FPS, V

OUT = io.ROOT / "data/review/click"


def main(n=20):
    plays, held = eval_plays()
    plays = plays.sample(n, seed=7, shuffle=True)  # any labelled play: the detector split no longer matters
    (OUT / "frames").mkdir(parents=True, exist_ok=True)
    ids = []
    for p in plays.iter_rows(named=True):
        cap = cv2.VideoCapture(str(V / f"train/{p['game_play']}_All29.mp4"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, round((p["snap"] - p["start"]).total_seconds() * FPS))
        ok, img = cap.read()
        if ok:
            cv2.imwrite(str(OUT / f"frames/{p['game_play']}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            ids.append(p["game_play"])
    (OUT / "manifest.json").write_text(json.dumps(ids))
    print(len(ids), "frames")


if __name__ == "__main__":
    main()
