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
YARD_LINES = np.r_[0, np.arange(10, 111, 5), 120]  # end lines too: they show whenever an end zone is in view
BORDER = 1.0  # the detected "sideline" is the middle of the 2-yd white border outside the field: y = -1 / 54.3


def white_mask(img):
    """White paint on the playing surface. The turf hue is estimated per frame (grass ~35-60, Atlanta's grey-teal
    turf ~93 on OpenCV's 0-180 scale), so any field colour works."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    core = hsv[int(.4 * h):int(.8 * h), int(.2 * w):int(.8 * w)].reshape(-1, 3)
    core = core[core[:, 1] > 30]
    h0 = float(np.median(core[:, 0])) if len(core) else 45.0
    lo, hi = max(h0 - 20, 0), min(h0 + 20, 180)
    field = cv2.morphologyEx(cv2.inRange(hsv, (lo, 25, 40), (hi, 255, 255)), cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    # paint = clearly brighter than its immediate surroundings (top-hat), so low-contrast lines on bright turf count too
    v = hsv[..., 2]
    tophat = cv2.morphologyEx(v, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    paint = ((tophat > 18) & (hsv[..., 1] < np.median(core[:, 1]) if len(core) else tophat > 18)).astype(np.uint8) * 255
    return paint & field


def _lines(img):
    white = white_mask(img)
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
    """-> (yard lines sorted left to right, flat lines sorted top to bottom, image height, width) or None.
    Which flat lines are the sidelines is decided later: stands railings and coaching-box lines are flat too."""
    h = img.shape[0]
    segs = _lines(img)
    yard, flat = _group(segs, True, h), _group(segs, False, h)
    if len(yard) < 2 or len(flat) < 2:
        return None
    mid = np.array([1.0, 0.0, -img.shape[1] / 2])  # the vertical line x = w/2
    flat.sort(key=lambda l: _cross(l, mid)[1])
    yard.sort(key=lambda l: _cross(l, flat[-1])[0])
    return yard, flat, h, img.shape[1]


def _pairs(det):
    """(far, near) sideline candidates: any two flat lines at least a quarter of the frame apart."""
    _, flat, h, w = det
    mid = np.array([1.0, 0.0, -w / 2])  # the vertical line x = w/2
    ys = [_cross(l, mid)[1] for l in flat]
    for i in range(len(flat)):
        for j in range(i + 1, len(flat)):
            if ys[j] - ys[i] > 0.25 * h:
                yield flat[i], flat[j]


def homography(lines, first_idx, direction, y_far):
    """Field->image homography for one labeling: yard line i gets YARD_LINES[first_idx + direction * i]."""
    yard, far, near = lines
    xs = YARD_LINES[first_idx + direction * np.arange(len(yard))]
    yf, yn = (-BORDER, W + BORDER) if y_far == 0 else (W + BORDER, -BORDER)
    src, dst = [], []
    for x, l in zip(xs, yard):
        for s, y in ((far, yf), (near, yn)):
            src.append((x, y)); dst.append(_cross(l, s))
    H, _ = cv2.findHomography(np.array(src, np.float32), np.array(dst, np.float32), cv2.RANSAC, 6.0)
    return H


def candidates(det, direction, y_far):
    yard = det[0]
    n = len(yard)
    firsts = range(0, len(YARD_LINES) - n + 1) if direction > 0 else range(n - 1, len(YARD_LINES))
    for far, near in _pairs(det):
        for f in firsts:
            H = homography((yard, far, near), f, direction, y_far)
            if H is not None:
                yield f, H


def to_field(H, pts_img):
    if len(pts_img) == 0:
        return np.zeros((0, 2))
    return cv2.perspectiveTransform(np.asarray(pts_img, np.float32)[None], np.linalg.inv(H))[0]


def pick_at_snap(lines, feet, los_x, direction, y_far):
    """Choose the labeling that puts the most players on the LOS; ties (e.g. a wrong sideline pair that keeps x right)
    are broken by how many players land inside the field at all."""
    best, best_n = None, -1.0
    for f, H in candidates(lines, direction, y_far):
        p = to_field(H, feet)
        inside = (p[:, 1] > -1) & (p[:, 1] < W + 1) & (p[:, 0] > 0) & (p[:, 0] < 120)
        n = float(((np.abs(p[:, 0] - los_x) < 1.5) & inside).sum() + 0.5 * inside.sum())
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


def model_points(step=0.5):
    """Points on the painted lines we can see: yard lines and the middle of each sideline border."""
    pts = [(x, y) for x in YARD_LINES for y in np.arange(0, W + 1e-6, step)]  # includes the end lines
    pts += [(x, y) for x in np.arange(0, 120 + 1e-6, step) for y in (-BORDER, W + BORDER)]
    return np.array(pts, np.float32)


def refine(H, img, iters=2):
    """Nudge the field->image homography so the painted-line model sits on white pixels (robust least squares
    on the distance transform of the white mask). Returns the refined H and the median residual in pixels."""
    from scipy.optimize import least_squares
    white = white_mask(img)
    dt = cv2.distanceTransform(255 - white, cv2.DIST_L2, 3)
    h, w = dt.shape
    P = model_points()

    def project(p8):
        Hm = np.append(p8, 1.0).reshape(3, 3)
        return cv2.perspectiveTransform(P[None], Hm)[0]

    def resid(p8):
        q = project(p8)
        ok = (q[:, 0] > 1) & (q[:, 0] < w - 2) & (q[:, 1] > 1) & (q[:, 1] < h - 2)
        r = np.full(len(q), 15.0)
        if ok.any():
            r[ok] = cv2.remap(dt, q[ok, 0:1].astype(np.float32), q[ok, 1:2].astype(np.float32), cv2.INTER_LINEAR)[:, 0]
        return np.minimum(r, 15.0)

    p = (H / H[2, 2]).ravel()[:8]
    for _ in range(iters):
        p = least_squares(resid, p, loss="huber", f_scale=2.0, x_scale="jac", max_nfev=200).x
    r = resid(p)
    return np.append(p, 1.0).reshape(3, 3), float(np.median(r[r < 15])) if (r < 15).any() else np.inf
