"""Is registration systematically off? For each play's snap frame: register (NGS-assisted labeling, refined), then
search a field-coordinate shift (dx, dy) of the NGS positions that best matches confident on-field detections."""
import numpy as np
import polars as pl
from ultralytics import YOLO

from presnap.video import assisted as A, detect as Dt, register as R
from presnap.video.pseudo import oracle_snap

plays, _ = A.eval_plays()
det = YOLO("yolo11x.pt")
trk = pl.read_csv(A.V / "train_player_tracking.csv").filter(pl.col("step") == 0)
rows = []
for p in plays.head(30).iter_rows(named=True):
    ngs0 = trk.filter(pl.col("game_play") == p["game_play"]).select("x_position", "y_position").to_numpy()
    frames, k0 = A.read_clip(p)
    if not frames:
        continue
    img = frames[k0]
    lines = R.detect_lines(img)
    if lines is None:
        continue
    b, sc = Dt.detect(det, img, conf=0.08)
    oc = oracle_snap(lines, Dt.feet(b), ngs0)
    if oc is None:
        continue
    H, _ = R.refine(oc[1], img)
    strong = Dt.feet(b[sc >= 0.3])
    pf = R.to_field(H, strong)
    pf = pf[(pf[:, 1] > 0) & (pf[:, 1] < 53.3) & (pf[:, 0] > 0) & (pf[:, 0] < 120)]
    best = (-1, 0, 0)
    for dx in np.arange(-7, 7.01, 0.25):
        for dy in np.arange(-4, 4.01, 0.25):
            d = np.linalg.norm((ngs0 + [dx, dy])[:, None] - pf[None], axis=-1).min(1)
            n = int((d < 0.75).sum())
            if n > best[0] or (n == best[0] and abs(dx) + abs(dy) < abs(best[1]) + abs(best[2])):
                best = (n, dx, dy)
    n0 = int((np.linalg.norm(ngs0[:, None] - pf[None], axis=-1).min(1) < 0.75).sum())
    rows.append(dict(game_play=p["game_play"], strong=len(pf), matched_at_zero=n0, best_matched=best[0], dx=best[1], dy=best[2]))
    print(rows[-1], flush=True)
df = pl.DataFrame(rows)
print(df.select(pl.col("dx").median(), pl.col("dy").median(), (pl.col("dx").abs() <= 1).mean().alias("dx_within_1yd"),
                pl.col("matched_at_zero").mean(), pl.col("best_matched").mean()))
