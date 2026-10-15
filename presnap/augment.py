"""Simulated video-tracking errors and the smoothing a tracker applies (numpy only).
Shared by the noise-tolerance test (noise.py) and noise-augmented training (train.py --aug-noise)."""
import numpy as np

RHO = 0.8


def smooth(X, M, k=5):
    """Centred moving average of positions over k frames (what any tracker's filter does), then velocities."""
    X = X.copy()
    pad = k // 2
    p = np.pad(X[..., :2], ((0, 0), (pad, pad), (0, 0), (0, 0)), mode="edge")
    X[..., :2] = np.where(M[..., None], np.mean([p[:, i:i + X.shape[1]] for i in range(k)], 0), X[..., :2])
    return velocities(X, M)


def velocities(X, M):
    """Central differences of x,y (stored /10) -> vx,vy (stored /5), one-sided at the ends; masked frames ignored."""
    pos = X[..., :2] * 10
    v = np.zeros_like(pos)
    v[:, 1:-1] = (pos[:, 2:] - pos[:, :-2]) / 0.2
    v[:, 0] = (pos[:, 1] - pos[:, 0]) / 0.1
    v[:, -1] = (pos[:, -1] - pos[:, -2]) / 0.1
    X[..., 2:4] = np.where(M[..., None], v / 5, 0)
    return X


def drift(X, M, sigma, rng, rho=RHO, smoothed=False):
    """AR(1) position error with stationary std `sigma` yards; rho = 1 is a fixed per-player offset."""
    if sigma == 0:
        return X
    N, T, P, _ = X.shape
    e = np.zeros((N, T, P, 2), np.float32)
    e[:, 0] = rng.normal(0, sigma, (N, P, 2))
    for t in range(1, T):
        e[:, t] = rho * e[:, t - 1] + np.sqrt(1 - rho ** 2) * rng.normal(0, sigma, (N, P, 2))
    X = X.copy()
    X[..., :2] += np.where(M[..., None], e / 10, 0)
    return smooth(X, M) if smoothed else velocities(X, M)


def drop(M, p, rng):
    return M & (rng.random(M.shape) >= p)


def swap(X, M, share, rng):
    X = X.copy()
    for i in np.where(rng.random(len(X)) < share)[0]:
        live = np.where(M[i, 0])[0]
        if len(live) < 2:
            continue
        a = rng.choice(live)
        d = np.linalg.norm(X[i, 0, live, :2] - X[i, 0, a, :2], axis=-1)
        d[live == a] = np.inf
        b = live[d.argmin()]  # swaps happen between players who are close together
        k = rng.integers(0, max(1, M[i].any(-1).sum()))
        X[i, k:, [a, b], :2] = X[i, k:, [b, a], :2]
    return velocities(X, M)


def augment(X, M, rng, sigma_max):
    """Training-time corruption for one batch: random drift (sigma ~ U(0, sigma_max), rho ~ U(0.8, 1)) on most plays,
    a few missed detections, then the same smoothing used at inference."""
    X = X.copy()
    for i in range(len(X)):
        if rng.random() < 0.8:
            X[i:i + 1] = drift(X[i:i + 1], M[i:i + 1], rng.uniform(0, sigma_max), rng, rho=rng.uniform(0.8, 1.0))
    M = M & (rng.random(M.shape) >= rng.uniform(0, 0.1))
    return smooth(X, M), M
