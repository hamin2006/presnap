"""Phase 2B exploration: register an All29 frame to the field from yard lines + sidelines, then overlay the
NGS player positions projected through the fitted homography. Writes overlay jpgs to /tmp/presnap_frames/."""
import sys
from pathlib import Path

import cv2
import numpy as np
import polars as pl

V = Path.home() / "presnap/data/raw/video/nfl-player-contact-detection"
OUT = Path("/tmp/presnap_frames"); OUT.mkdir(exist_ok=True)
FPS = 60000 / 1001
W_FIELD = 53.3


def frame_at(video, t_sec):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, round(t_sec * FPS))
    ok, img = cap.read()
    return img


def white_lines(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (30, 40, 40), (90, 255, 255))
    field = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    white = cv2.inRange(hsv, (0, 0, 170), (180, 60, 255)) & field
    segs = cv2.HoughLinesP(white, 1, np.pi / 360, threshold=80, minLineLength=80, maxLineGap=15)
    return white, (segs[:, 0] if segs is not None else np.zeros((0, 4)))


def fit_line(pts):
    vx, vy, x0, y0 = cv2.fitLine(np.asarray(pts, np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    return np.array([vy, -vx, vx * y0 - vy * x0])  # a x + b y + c = 0


def cluster_lines(segs, steep):
    """Group segments into lines: steep ones (yard lines) by x at mid-height, flat ones (sidelines) by y."""
    out = []
    for x1, y1, x2, y2 in segs:
        ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        is_steep = 25 < ang < 155
        if is_steep != steep:
            continue
        key = (x1 + (540 - y1) * (x2 - x1) / (y2 - y1 + 1e-9)) if steep else (y1 + y2) / 2
        out.append((key, (x1, y1), (x2, y2)))
    out.sort()
    groups, cur = [], []
    for k, a, b in out:
        if cur and k - cur[-1][0] > 25:
            groups.append(cur); cur = []
        cur.append((k, a, b))
    if cur:
        groups.append(cur)
    lines = []
    for g in groups:
        pts = [p for _, a, b in g for p in (a, b)]
        span = max(p[1] for p in pts) - min(p[1] for p in pts) if steep else max(p[0] for p in pts) - min(p[0] for p in pts)
        if span > (250 if steep else 600):
            lines.append((np.mean([k for k, _, _ in g]), fit_line(pts)))
    return lines


def intersect(l1, l2):
    p = np.cross(l1, l2)
    return p[:2] / p[2]


def main(game_play="58168_003392"):
    meta = pl.read_csv(V / "train_video_metadata.csv").filter((pl.col("game_play") == game_play) & (pl.col("view") == "Endzone"))
    start, snap = [pl.Series([meta[c][0]]).str.to_datetime(time_zone="UTC")[0] for c in ("start_time", "snap_time")]
    t_snap = (snap - start).total_seconds()
    img = frame_at(V / f"train/{game_play}_All29.mp4", t_snap)
    white, segs = white_lines(img)
    yard = cluster_lines(segs, steep=True)
    side = cluster_lines(segs, steep=False)
    print(f"{len(segs)} segments, {len(yard)} yard lines, {len(side)} sidelines")

    trk = pl.read_csv(V / "train_player_tracking.csv").filter((pl.col("game_play") == game_play) & (pl.col("step") == 0))
    # LOS anchor: NGS x of the ball isn't given; use the midpoint between offensive and defensive linemen
    ol = trk.filter(pl.col("position").is_in(["C", "G", "T"]))["x_position"].mean()
    dl = trk.filter(pl.col("position").is_in(["DE", "DT", "NT"]))["x_position"].mean()
    los = (ol + dl) / 2
    print(f"LOS (NGS x incl. end zone) ~ {los:.1f}")

    vis = img.copy()
    for _, l in yard + side:
        a, b, c = l
        pts = [(int(x), int(-(a * x + c) / b)) for x in (0, 1919)] if abs(b) > abs(a) else [(int(-(b * y + c) / a), int(y)) for y in (0, 1079)]
        cv2.line(vis, pts[0], pts[1], (0, 0, 255), 2)

    if len(yard) >= 3 and len(side) >= 2:
        top, bot = side[0][1], side[-1][1]
        # image-x of each yard line at the bottom sideline, left to right
        ys = sorted(yard, key=lambda r: intersect(r[1], bot)[0])
        bx = [intersect(l, bot)[0] for _, l in ys]
        # which yard line is just left of the LOS? linemen at the snap sit at the LOS in the image too:
        # take the median image x of the NGS-projected... (not available yet) -> use the yard line count heuristic:
        # try every alignment of 5-yd values and keep the one whose homography puts the linemen closest together
        best = None
        for first in np.arange(10, 115 - 5 * (len(ys) - 1) + 1, 5):
            vals = first + 5 * np.arange(len(ys))
            src = np.array([intersect(l, s) for (_, l) in ys for s in (top, bot)], np.float32)
            # NGS y: 0 at the far sideline? orientation is checked by the overlay; try both
            for flip in (False, True):
                dst = np.array([(v, (W_FIELD if (s is top) ^ flip else 0.0)) for v in vals for s in (top, bot)], np.float32)
                H, _ = cv2.findHomography(dst, src)
                if H is None:
                    continue
                pts = cv2.perspectiveTransform(trk.select("x_position", "y_position").to_numpy().astype(np.float32)[None], H)[0]
                inside = ((pts[:, 0] > 0) & (pts[:, 0] < 1920) & (pts[:, 1] > 0) & (pts[:, 1] < 1080)).mean()
                score = -abs(first + 5 * np.searchsorted(vals, los) - los)  # LOS must fall within the labelled lines
                cand = (inside, score, first, flip, H, pts)
                if best is None or cand[:2] > best[:2]:
                    best = cand
        inside, _, first, flip, H, pts = best
        print(f"chosen: first yard line x={first}, flip={flip}, NGS points inside frame {inside:.0%}")
        for (x, y), side_ in zip(pts, trk["team"]):
            cv2.circle(vis, (int(x), int(y)), 9, (255, 255, 0) if side_ == "home" else (0, 255, 255), 2)
    cv2.imwrite(str(OUT / f"{game_play}_reg.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 80])
    cv2.imwrite(str(OUT / f"{game_play}_white.jpg"), white)


if __name__ == "__main__":
    main(*sys.argv[1:])
