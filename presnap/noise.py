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
from presnap.augment import drift, drop, smooth, swap
from presnap.train import DEV, TARGETS, init_from, load, predict

CHECK = {"snap": 0, "+2.0 s": 20}


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


def main(tag="main"):
    D = load("bdb26", None, 0, 40)
    te = np.where(D["split"] == "test")[0]
    X0, M0 = D["X"][te].copy(), D["M"][te].copy()
    Dt = {k: (v[te] if isinstance(v, np.ndarray) and v.ndim and len(v) == len(D["X"]) else v) for k, v in D.items()}
    idx = np.arange(len(te))
    models = []
    for s in range(5):
        m = CoverageModel(D["X"].shape[-1], D["S"].shape[-1], t_max=D["X"].shape[1]).to(DEV)
        init_from(m, io.INTERIM / f"runs/{tag}/seed{s}.pt")
        models.append(m.eval())
    rng = np.random.default_rng(0)
    res = {"clean": score(models, Dt, idx, X0, M0)}
    res["clean_smoothed"] = score(models, Dt, idx, smooth(X0, M0), M0)
    for sigma in (0.25, 0.5, 1.0, 1.5, 2.0, 3.0):
        res[f"drift_{sigma}yd"] = score(models, Dt, idx, drift(X0, M0, sigma, rng), M0)
        res[f"drift_{sigma}yd_smoothed"] = score(models, Dt, idx, drift(X0, M0, sigma, rng, smoothed=True), M0)
        res[f"slow_{sigma}yd_smoothed"] = score(models, Dt, idx, drift(X0, M0, sigma, rng, rho=0.97, smoothed=True), M0)
        res[f"offset_{sigma}yd"] = score(models, Dt, idx, drift(X0, M0, sigma, rng, rho=1.0), M0)
    for p in (0.05, 0.1, 0.2, 0.3):
        res[f"drop_{p}"] = score(models, Dt, idx, X0, drop(M0, p, rng))
    for share in (0.25, 0.5, 1.0):
        res[f"swap_{share}"] = score(models, Dt, idx, swap(X0, M0, share, rng), M0)
    res["combo_1yd_drop0.1_swap0.5"] = score(models, Dt, idx, swap(drift(X0, M0, 1.0, rng), M0, 0.5, rng), drop(M0, 0.1, rng))

    clean = res["clean"]
    def tolerance(prefix, smoothed):
        ok = [k for k in res if k.startswith(prefix) and k.endswith("_smoothed") == smoothed
              and clean["man"]["snap"] - res[k]["man"]["snap"] <= 0.03 and clean["shell"]["snap"] - res[k]["shell"]["snap"] <= 0.05]
        return max((float(k.split("_")[1][:-2]) for k in ok), default=0.0)
    tol = {"fast_drift_raw": tolerance("drift_", False), "fast_drift_smoothed": tolerance("drift_", True),
           "slow_drift_smoothed": tolerance("slow_", True), "static_offset": tolerance("offset_", False)}
    res["tolerance_sigma_yd"] = tol
    (io.ROOT / f"reports/phase1f_noise_{tag}.json").write_text(json.dumps(res, indent=2))
    lines = ["# Phase 1F: noise tolerance", "",
             "1C ensemble on BDB 2026 test plays (weeks 16–18) with simulated video-tracking errors. Accuracy at the snap / +2.0 s. "
             "Drift = AR(1) position error (ρ = 0.8 per 0.1 s), σ in yards; drop = share of player-frames missing; swap = share of plays with one identity switch.", "",
             "| Condition | Man/zone snap | +2.0 s | 7-class snap | +2.0 s | Shell snap | +2.0 s |", "|---|---|---|---|---|---|---|"]
    for k, r in res.items():
        if isinstance(r, dict) and "man" in r:
            lines.append(f"| {k} | {r['man']['snap']:.1%} | {r['man']['+2.0 s']:.1%} | {r['cov']['snap']:.1%} | {r['cov']['+2.0 s']:.1%} | "
                         f"{r['shell']['snap']:.1%} | {r['shell']['+2.0 s']:.1%} |")
    lines += ["", f"**Tolerance (largest σ losing ≤ 3 pts man/zone and ≤ 5 pts shell at the snap), model `{tag}`:** {tol}. "
              "Gate: ≥ 0.5 yd → full video scope; < 0.5 yd → scope video to shell and man/zone, or add human QA."]
    (io.ROOT / f"reports/phase1f_noise_{tag}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])
