"""Phase 1C: train the coverage model, ensemble seeds, and compare accuracy-vs-time with GBMs.

Run on the PC (GPU):  uv run --group train python -m presnap.train --seeds 5
Ablation (no defender-count signal):  ... --max-def 5 --seeds 1 --tag maxdef5 --no-gbm
Writes reports/phase1c_<tag>.md/.json; per-seed test probabilities go to data/interim/runs/<tag>/.
"""
import argparse
import json
import time

import numpy as np
import polars as pl
import torch
import torch.nn.functional as Fn

from presnap import baselines, features, io, seq
from presnap.model import CoverageModel
from presnap.shell import KEY

CHECKPOINTS = [0, 5, 10, 15, 20, "throw"]
TARGETS = {"cov": "y_cov", "man": "y_man", "shell": "y_shell"}
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load(max_def):
    p = io.INTERIM / f"seq_bdb26_{max_def}.npz"
    if not p.exists():
        io.INTERIM.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, **{k: v for k, v in seq.build("bdb26", max_def).items() if v is not None})
    return dict(np.load(p, allow_pickle=True))


def batches(D, idx, bs, train):
    if train:
        idx = np.random.permutation(idx)
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        X = torch.from_numpy(D["X"][b]).to(DEV)
        if train:  # mirror the field left/right
            flip = torch.rand(len(b), device=DEV) < 0.5
            X[flip, ..., 1] *= -1
            X[flip, ..., 3] *= -1
        yield b, X, torch.from_numpy(D["M"][b]).to(DEV), torch.from_numpy(D["S"][b]).to(DEV)


def loss_fn(out, D, b, T, smooth):
    valid = torch.arange(T, device=DEV)[None] <= torch.from_numpy(np.minimum(D["last"][b], T - 1)).to(DEV)[:, None]
    total = 0.0
    for h, key in TARGETS.items():
        y = torch.from_numpy(D[key][b]).to(DEV)[:, None].expand(-1, T)
        m = valid & (y >= 0)
        if m.any():
            total = total + Fn.cross_entropy(out[h][m], y[m], label_smoothing=smooth)
    return total


@torch.no_grad()
def predict(model, D, idx):
    model.eval()
    probs = {h: [] for h in TARGETS}
    for b, X, M, S in batches(D, idx, 256, False):
        out = model(X, M, S)
        for h in TARGETS:
            probs[h].append(out[h].softmax(-1).cpu().numpy())
    return {h: np.concatenate(v) for h, v in probs.items()}


def train_seed(D, seed, epochs, patience=8):
    torch.manual_seed(seed); np.random.seed(seed)
    tr, va = np.where(D["split"] == "train")[0], np.where(D["split"] == "val")[0]
    T = D["X"].shape[1]
    model = CoverageModel(D["X"].shape[-1], D["S"].shape[-1], t_max=T).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, best_state, bad = 1e9, None, 0
    for ep in range(epochs):
        model.train(); t0 = time.time()
        for b, X, M, S in batches(D, tr, 64, True):
            opt.zero_grad()
            loss = loss_fn(model(X, M, S), D, b, T, 0.1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = np.mean([loss_fn(model(X, M, S), D, b, T, 0.0).item() for b, X, M, S in batches(D, va, 256, False)])
        print(f"seed {seed} epoch {ep} val {vl:.4f} ({time.time() - t0:.1f}s)", flush=True)
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model


def at(P, frames):
    """P [N,T,K] -> [N,K] at a per-play frame index."""
    return P[np.arange(len(P)), frames]


def evaluate(D, probs, te):
    res = {}
    last = np.minimum(D["last"][te], D["X"].shape[1] - 1)
    for h, key in TARGETS.items():
        y = D[key][te]
        ok = y >= 0
        res[h] = {}
        for c in CHECKPOINTS:
            fr = last if c == "throw" else np.minimum(c, last)
            pred = at(probs[h], fr).argmax(1)
            correct = (pred == y)[ok]
            lo, hi = baselines.boot_ci(D["game"][te][ok].tolist(), correct.tolist())
            r = dict(acc=round(float(correct.mean()), 4), ci=[round(lo, 4), round(hi, 4)], n=int(ok.sum()))
            if D.get("agree") is not None:
                r["acc_agree"] = round(float(correct[D["agree"][te][ok]].mean()), 4)
            res[h][str(c)] = r
    return res


def gbm_curve():
    plays = features.personnel_counts(pl.read_parquet(io.PROCESSED / "plays.parquet")).filter(source="bdb26")
    sit = features.SITUATION_NUM + features.SITUATION_CAT
    out = {h: {} for h in TARGETS}
    for c in CHECKPOINTS:
        frames = {"snap": 0} if c == 0 else {"snap": 0, "post": "last" if c == "throw" else ("min_last", c)}
        b = plays.join(features.build("bdb26", frames), on=KEY, how="left")
        cols = sit + [x for x in b.columns if x.split("_")[0] in frames]
        for h, t in (("cov", "cov7"), ("man", "man"), ("shell", "shell")):
            out[h][str(c)] = baselines.fit_eval(b, cols, t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--max-def", type=int, default=None)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--no-gbm", action="store_true")
    a = ap.parse_args()

    D = load(a.max_def)
    te = np.where(D["split"] == "test")[0]
    run = io.INTERIM / "runs" / a.tag
    run.mkdir(parents=True, exist_ok=True)
    seeds = []
    for s in range(a.seeds):
        m = train_seed(D, s, a.epochs)
        p = predict(m, D, te)
        np.savez_compressed(run / f"seed{s}.npz", **p)
        seeds.append(p)
        torch.save(m.state_dict(), run / f"seed{s}.pt")
    ens = {h: np.mean([p[h] for p in seeds], 0) for h in TARGETS}
    res = dict(n_params=sum(x.numel() for x in m.parameters()), seeds=a.seeds, max_def=a.max_def,
               transformer=evaluate(D, ens, te), single_seed=evaluate(D, seeds[0], te))
    if not a.no_gbm:
        res["gbm"] = gbm_curve()
    (io.ROOT / f"reports/phase1c_{a.tag}.json").write_text(json.dumps(res, indent=2))

    lab = lambda c: "snap" if c == "0" else ("throw" if c == "throw" else f"+{int(c) / 10:.1f} s")
    lines = [f"# Phase 1C coverage model: {a.tag}", "",
             f"Transformer ({res['n_params']:,} params, {a.seeds}-seed ensemble, max_def={a.max_def}) vs GBM trained at the same moment. "
             "Test = weeks 16–18 (2023); 95% game-clustered bootstrap CIs. `agree` = plays where NGS and FTN labels match.", ""]
    for h, name in (("man", "Man vs zone"), ("cov", "7-class coverage"), ("shell", "Played shell")):
        lines += [f"## {name}", "", "| Moment | Transformer | 95% CI | Agree subset | GBM | GBM 95% CI |", "|---|---|---|---|---|---|"]
        for c in map(str, CHECKPOINTS):
            r = res["transformer"][h][c]
            g = res.get("gbm", {}).get(h, {}).get(c)
            gtxt = f"{g['acc']:.1%} | {g['ci'][0]:.1%}–{g['ci'][1]:.1%}" if g else "– | –"
            lines.append(f"| {lab(c)} | **{r['acc']:.1%}** | {r['ci'][0]:.1%}–{r['ci'][1]:.1%} | {r.get('acc_agree', float('nan')):.1%} | {gtxt} |")
        lines.append("")
    (io.ROOT / f"reports/phase1c_{a.tag}.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
