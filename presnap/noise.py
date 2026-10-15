"""Phase 1F: how much tracking error can the coverage model take? The answer is the accuracy spec for video.

Corrupts the BDB 2026 test plays the way a video tracker would, re-derives velocities from the noisy positions,
and re-scores the 1C ensemble at the snap and at +2.0 s:
  * drift:  temporally correlated position error (AR(1), rho = 0.8 per 0.1 s), sigma in yards
  * drop:   each player missing from a frame with probability p (missed detections)
  * swap:   two nearby players exchange tracks from a random frame on (identity switch), in a share of plays

Gate (build plan 1F): the sigma at which man/zone loses <= 3 pts and shell <= 5 pts. Under 0.5 yd => narrow video scope.
Run on the PC: uv run --group train python -m presnap.noise
"""
import json

import numpy as np
import torch

from presnap import io
from presnap.model import CoverageModel
from presnap.train import DEV, TARGETS, init_from, load, predict

RHO = 0.8
CHECK = {"snap": 0, "+2.0 s": 20}


def velocities(X, M):
    """Central differences of x,y (stored /10) -> vx,vy (stored /5), one-sided at the ends; masked frames ignored."""
    pos = X[..., :2] * 10
    v = np.zeros_like(pos)
    v[:, 1:-1] = (pos[:, 2:] - pos[:, :-2]) / 0.2
    v[:, 0] = (pos[:, 1] - pos[:, 0]) / 0.1
    v[:, -1] = (pos[:, -1] - pos[:, -2]) / 0.1
    X[..., 2:4] = np.where(M[..., None], v / 5, 0)
    return X


def drift(X, M, sigma, rng):
    if sigma == 0:
        return X
    N, T, P, _ = X.shape
    e = np.zeros((N, T, P, 2), np.float32)
    e[:, 0] = rng.normal(0, sigma, (N, P, 2))
    for t in range(1, T):
        e[:, t] = RHO * e[:, t - 1] + np.sqrt(1 - RHO ** 2) * rng.normal(0, sigma, (N, P, 2))
    X = X.copy()
    X[..., :2] += np.where(M[..., None], e / 10, 0)
    return velocities(X, M)


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


def score(models, D, te, X, M):
    D2 = dict(D, X=X, M=M)
    probs = [predict(m, D2, te) for m in models]
    ens = {h: np.mean([p[h] for p in probs], 0) for h in TARGETS}
    out, last = {}, np.minimum(D["last"][te], D["X"].shape[1] - 1)
    for h in ("man", "cov", "shell"):
        y = D[TARGETS[h]][te]
        ok = y >= 0
        out[h] = {name: round(float((ens[h][np.arange(len(te)), np.minimum(f, last)].argmax(1) == y)[ok].mean()), 4)
                  for name, f in CHECK.items()}
    return out


def main():
    D = load("bdb26", None, 0, 40)
    te = np.where(D["split"] == "test")[0]
    X0, M0 = D["X"][te].copy(), D["M"][te].copy()
    Dt = {k: (v[te] if isinstance(v, np.ndarray) and v.ndim and len(v) == len(D["X"]) else v) for k, v in D.items()}
    idx = np.arange(len(te))
    models = []
    for s in range(5):
        m = CoverageModel(D["X"].shape[-1], D["S"].shape[-1], t_max=D["X"].shape[1]).to(DEV)
        init_from(m, io.INTERIM / f"runs/main/seed{s}.pt")
        models.append(m.eval())
    rng = np.random.default_rng(0)
    res = {"clean": score(models, Dt, idx, X0, M0)}
    for sigma in (0.25, 0.5, 1.0, 1.5, 2.0, 3.0):
        res[f"drift_{sigma}yd"] = score(models, Dt, idx, drift(X0, M0, sigma, rng), M0)
    for p in (0.05, 0.1, 0.2, 0.3):
        res[f"drop_{p}"] = score(models, Dt, idx, X0, drop(M0, p, rng))
    for share in (0.25, 0.5, 1.0):
        res[f"swap_{share}"] = score(models, Dt, idx, swap(X0, M0, share, rng), M0)
    res["combo_1yd_drop0.1_swap0.5"] = score(models, Dt, idx, swap(drift(X0, M0, 1.0, rng), M0, 0.5, rng), drop(M0, 0.1, rng))

    clean = res["clean"]
    ok = [k for k in res if k.startswith("drift_") and clean["man"]["snap"] - res[k]["man"]["snap"] <= 0.03
          and clean["shell"]["snap"] - res[k]["shell"]["snap"] <= 0.05]
    tol = max((float(k.split("_")[1][:-2]) for k in ok), default=0.0)
    res["tolerance_sigma_yd"] = tol
    (io.ROOT / "reports/phase1f_noise.json").write_text(json.dumps(res, indent=2))
    lines = ["# Phase 1F: noise tolerance", "",
             "1C ensemble on BDB 2026 test plays (weeks 16–18) with simulated video-tracking errors. Accuracy at the snap / +2.0 s. "
             "Drift = AR(1) position error (ρ = 0.8 per 0.1 s), σ in yards; drop = share of player-frames missing; swap = share of plays with one identity switch.", "",
             "| Condition | Man/zone snap | +2.0 s | 7-class snap | +2.0 s | Shell snap | +2.0 s |", "|---|---|---|---|---|---|---|"]
    for k, r in res.items():
        if isinstance(r, dict):
            lines.append(f"| {k} | {r['man']['snap']:.1%} | {r['man']['+2.0 s']:.1%} | {r['cov']['snap']:.1%} | {r['cov']['+2.0 s']:.1%} | "
                         f"{r['shell']['snap']:.1%} | {r['shell']['+2.0 s']:.1%} |")
    lines += ["", f"**Tolerance (man/zone ≤ 3 pts and shell ≤ 5 pts lost at the snap): σ = {tol} yd of drift.** "
              f"Gate: ≥ 0.5 yd → full video scope; < 0.5 yd → scope video to shell and man/zone, or add human QA."]
    (io.ROOT / "reports/phase1f_noise.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
