"""Phase 2B: register All29 (all-22 sideline) frames to field coordinates from the painted lines.

Field frame = NGS: x 0-120 yards including end zones (goal lines at 10 and 110), y 0-53.3.
Per frame: find the yard lines (white, steep in the image) and the two sidelines (long, flat), then fit a
homography from their intersections. Yard lines repeat every 5 yards, so which line is which is ambiguous;
it is resolved with the line of scrimmage (from play-by-play in practice): at the snap, the right labeling is
the one that puts the most players on the LOS. Later frames keep the labeling most consistent with the last.
"""
import cv2
import numpy as np

W = 53.3
YARD_LINES = np.arange(10, 111, 5)


def _lines(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    field = cv2.morphologyEx(cv2.inRange(hsv, (30, 40, 40), (90, 255, 255)), cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    white = cv2.inRange(hsv, (0, 0, 170), (180, 60, 255)) & field
    segs = cv2.HoughLinesP(white, 1, np.pi / 360, threshold=80, minLineLength=80, maxLineGap=15)
    return np.zeros((0, 4)) if segs is None else segs.reshape(-1, 4).astype(float)


def _fit(pts):
    vx, vy, x0, y0 = cv2.fitLine(np.asarray(pts, np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    return np.array([vy, -vx, vx * y0 - vy * x0])


def _group(segs, steep, h):
    keyed = []
    for x1, y1, x2, y2 in segs:
        ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if (25 < ang < 155) != steep:
            continue
        k = x1 + (h / 2 - y1) * (x2 - x1) / (y2 - y1) if steep else (y1 + y2) / 2
        keyed.append((k, (x1, y1), (x2, y2)))
    keyed.sort(key=lambda r: r[0])
    groups, cur = [], []
    for r in keyed:
        if cur and r[0] - cur[-1][0] > 25:
            groups.append(cur); cur = []
        cur.append(r)
    groups += [cur] if cur else []
    out = []
    for g in groups:
        pts = np.array([p for _, a, b in g for p in (a, b)])
        span = np.ptp(pts[:, 1 if steep else 0])
        if span > (0.23 * h if steep else 0.3 * h * 16 / 9):
            out.append(_fit(pts))
    return out


def _cross(l1, l2):
    p = np.cross(l1, l2)
    return p[:2] / p[2]


def detect_lines(img):
    """-> (yard lines sorted left to right along the near sideline, far sideline, near sideline) or None."""
    h = img.shape[0]
    segs = _lines(img)
    yard, flat = _group(segs, True, h), _group(segs, False, h)
    if len(yard) < 2 or len(flat) < 2:
        return None
    far, near = flat[0], flat[-1]
    yard.sort(key=lambda l: _cross(l, near)[0])
    return yard, far, near


def homography(lines, first_idx, direction, y_far):
    """Field->image homography for one labeling: yard line i gets YARD_LINES[first_idx + direction * i]."""
    yard, far, near = lines
    xs = YARD_LINES[first_idx + direction * np.arange(len(yard))] if direction > 0 else YARD_LINES[first_idx - np.arange(len(yard))]
    src, dst = [], []
    for x, l in zip(xs, yard):
        for s, y in ((far, y_far), (near, W - y_far)):
            src.append((x, y)); dst.append(_cross(l, s))
    H, _ = cv2.findHomography(np.array(src, np.float32), np.array(dst, np.float32), cv2.RANSAC, 6.0)
    return H


def candidates(lines, direction, y_far):
    n = len(lines[0])
    firsts = range(0, len(YARD_LINES) - n + 1) if direction > 0 else range(n - 1, len(YARD_LINES))
    for f in firsts:
        H = homography(lines, f, direction, y_far)
        if H is not None:
            yield f, H


def to_field(H, pts_img):
    if len(pts_img) == 0:
        return np.zeros((0, 2))
    return cv2.perspectiveTransform(np.asarray(pts_img, np.float32)[None], np.linalg.inv(H))[0]


def pick_at_snap(lines, feet, los_x, direction, y_far):
    """Choose the labeling that puts the most detected feet within 1.5 yd of the LOS and on the field."""
    best, best_n = None, -1
    for f, H in candidates(lines, direction, y_far):
        p = to_field(H, feet)
        n = int(((np.abs(p[:, 0] - los_x) < 1.5) & (p[:, 1] > 0) & (p[:, 1] < W)).sum())
        if n > best_n:
            best, best_n = H, n
    return best


def pick_consistent(lines, prev_H, direction, y_far, h, w):
    """Choose the labeling whose mapping of a grid of image points is closest to the previous frame's."""
    grid = np.array([(x, y) for x in np.linspace(0.2 * w, 0.8 * w, 5) for y in np.linspace(0.35 * h, 0.75 * h, 3)])
    ref = to_field(prev_H, grid)
    best, best_d = None, np.inf
    for _, H in candidates(lines, direction, y_far):
        d = np.median(np.linalg.norm(to_field(H, grid) - ref, axis=1))
        if d < best_d:
            best, best_d = H, d
    return best
