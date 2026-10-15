"""Phase 1C/1E: train the coverage model, ensemble seeds, report accuracy vs time against GBMs.

Run on the PC (GPU):
  1C  uv run --group train python -m presnap.train --seeds 5
      ablation: ... --max-def 5 --seeds 1 --tag maxdef5 --no-gbm
  1E  ... --source statsbomb --t-min -30 --t-max 20 --init data/interim/runs/main/seed0.pt --tag sb_ft
Writes reports/phase1c_<tag>.md/.json; per-seed test probabilities go to data/interim/runs/<tag>/.
"""
import argparse
import json
import time

import numpy as np
import polars as pl
import torch
import torch.nn.functional as Fn
from sklearn.metrics import roc_auc_score

from presnap import augment, baselines, features, io, seq
from presnap.model import CoverageModel
from presnap.shell import KEY

TARGETS = {"cov": "y_cov", "man": "y_man", "shell": "y_shell", "blitz": "y_blitz"}
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load(source, max_def, t_min, t_max):
    p = io.INTERIM / f"seq_{source}_{max_def}_{t_min}_{t_max}.npz"
    if not p.exists():
        io.INTERIM.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, **{k: v for k, v in seq.build(source, max_def, t_max, t_min).items() if v is not None})
    return dict(np.load(p, allow_pickle=True))


AUG = {"sigma": 0.0, "rng": np.random.default_rng(0)}  # set by --aug-noise


def batches(D, idx, bs, train):
    if train:
        idx = np.random.permutation(idx)
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        Xn, Mn = D["X"][b], D["M"][b]
        if AUG["sigma"] > 0:  # smoothing is part of the pipeline whenever we train for video
            Xn, Mn = augment.augment(Xn, Mn, AUG["rng"], AUG["sigma"]) if train else (augment.smooth(Xn, Mn), Mn)
        X = torch.from_numpy(np.ascontiguousarray(Xn)).to(DEV)
        if train:  # mirror the field left/right
            flip = torch.rand(len(b), device=DEV) < 0.5
            X[flip, ..., 1] *= -1
            X[flip, ..., 3] *= -1
        yield b, X, torch.from_numpy(np.ascontiguousarray(Mn)).to(DEV), torch.from_numpy(D["S"][b]).to(DEV)


def loss_fn(out, D, b, smooth):
    present = torch.from_numpy(D["M"][b].any(-1)).to(DEV)  # frames with tracking (uses the uncorrupted mask)
    T = present.shape[1]
    total = 0.0
    for h, key in TARGETS.items():
        y = torch.from_numpy(D[key][b]).to(DEV)[:, None].expand(-1, T)
        m = present & (y >= 0)
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


def init_from(model, path):
    """Warm start: copy every weight whose name and shape match (time embedding / new heads stay fresh)."""
    src, dst = torch.load(path, map_location=DEV), model.state_dict()
    keep = {k: v for k, v in src.items() if k in dst and v.shape == dst[k].shape}
    dst.update(keep)
    model.load_state_dict(dst)
    print(f"init: copied {len(keep)}/{len(dst)} tensors from {path}", flush=True)


def train_seed(D, seed, epochs, lr, init=None, patience=8):
    torch.manual_seed(seed); np.random.seed(seed)
    tr, va = np.where(D["split"] == "train")[0], np.where(D["split"] == "val")[0]
    model = CoverageModel(D["X"].shape[-1], D["S"].shape[-1], t_max=D["X"].shape[1]).to(DEV)
    if init:
        init_from(model, init)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, best_state, bad = 1e9, None, 0
    for ep in range(epochs):
        model.train(); t0 = time.time()
        for b, X, M, S in batches(D, tr, 64, True):
            opt.zero_grad()
            loss = loss_fn(model(X, M, S), D, b, 0.1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = np.mean([loss_fn(model(X, M, S), D, b, 0.0).item() for b, X, M, S in batches(D, va, 256, False)])
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


def evaluate(D, probs, te, checkpoints):
    """Accuracy (and AUC for binary heads) at each moment; frame offsets are relative to the snap."""
    res, t0, T = {}, int(D["t0"]), D["X"].shape[1]
    last = np.minimum(D["last"][te], T - 1)
    for h, key in TARGETS.items():
        y = D[key][te]
        if (y >= 0).sum() == 0:
            continue
        res[h] = {}
        for c in checkpoints:
            fr = last if c == "throw" else np.minimum(t0 + c, last)
            has = D["M"][te, fr].any(-1) & (y >= 0)  # play has tracking at that moment and a label
            if has.sum() < 30:
                continue
            p = probs[h][np.arange(len(te)), fr][has]
            correct = p.argmax(1) == y[has]
            lo, hi = baselines.boot_ci(D["game"][te][has].tolist(), correct.tolist())
            r = dict(acc=round(float(correct.mean()), 4), ci=[round(lo, 4), round(hi, 4)], n=int(has.sum()))
            if p.shape[1] == 2:
                r["auc"] = round(float(roc_auc_score(y[has], p[:, 1])), 4)
            if "agree" in D:
                r["acc_agree"] = round(float(correct[D["agree"][te][has]].mean()), 4)
            res[h][str(c)] = r
    return res


def gbm_curve(source, checkpoints):
    plays = features.personnel_counts(pl.read_parquet(io.PROCESSED / "plays.parquet")).filter(source=source)
    if source == "statsbomb":
        plays = plays.filter(pl.col("sb_quality_ok"))
    plays = plays.with_columns(blitz=pl.col("n_pass_rushers") >= 5 if source == "statsbomb" else pl.lit(None, pl.Boolean))
    sit = [c for c in features.SITUATION_NUM + features.SITUATION_CAT if plays[c].null_count() < plays.height]
    out = {h: {} for h in TARGETS}
    for c in checkpoints:
        if source == "statsbomb" and c != 0:
            continue  # StatsBomb has no throw marker; compare at the snap only
        frames = {"snap": 0} if c == 0 else {"snap": 0, "post": "last" if c == "throw" else ("min_last", c)}
        b = plays.join(features.build(source, frames), on=KEY, how="left")
        cols = sit + [x for x in b.columns if x.split("_")[0] in frames]
        for h, t in (("cov", "cov7"), ("man", "man"), ("shell", "shell"), ("blitz", "blitz")):
            if t == "blitz" and source != "statsbomb":
                continue
            out[h][str(c)] = baselines.fit_eval(b, cols, t)
    return out


def label(c):
    if c == "throw":
        return "throw"
    return "snap" if c == "0" else f"{int(c) / 10:+.1f} s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="bdb26")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-def", type=int, default=None)
    ap.add_argument("--t-min", type=int, default=0)
    ap.add_argument("--t-max", type=int, default=seq.T_MAX)
    ap.add_argument("--init", default=None)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--no-gbm", action="store_true")
    ap.add_argument("--aug-noise", type=float, default=0.0, help="max drift sigma (yd) for noise-augmented training")
    a = ap.parse_args()
    AUG["sigma"] = a.aug_noise
    checkpoints = [-20, -10, -5, 0, 10, 20] if a.source == "statsbomb" else [0, 5, 10, 15, 20, "throw"]

    D = load(a.source, a.max_def, a.t_min, a.t_max)
    te = np.where(D["split"] == "test")[0]
    run = io.INTERIM / "runs" / a.tag
    run.mkdir(parents=True, exist_ok=True)
    seeds = []
    for s in range(a.seeds):
        m = train_seed(D, s, a.epochs, a.lr, a.init)
        p = predict(m, D, te)
        np.savez_compressed(run / f"seed{s}.npz", **p)
        seeds.append(p)
        torch.save(m.state_dict(), run / f"seed{s}.pt")
    ens = {h: np.mean([p[h] for p in seeds], 0) for h in TARGETS}
    res = dict(source=a.source, n_params=sum(x.numel() for x in m.parameters()), seeds=a.seeds, max_def=a.max_def,
               init=a.init, transformer=evaluate(D, ens, te, checkpoints), single_seed=evaluate(D, seeds[0], te, checkpoints))
    if not a.no_gbm:
        res["gbm"] = gbm_curve(a.source, checkpoints)
    (io.ROOT / f"reports/phase1c_{a.tag}.json").write_text(json.dumps(res, indent=2))

    lines = [f"# Coverage model: {a.tag}", "",
             f"Source {a.source}. Transformer ({res['n_params']:,} params, {a.seeds}-seed ensemble, max_def={a.max_def}, init={a.init}) "
             "vs GBM trained at the same moment. Test split; 95% game-clustered bootstrap CIs.", ""]
    names = {"man": "Man vs zone", "cov": "7-class coverage", "shell": "Played shell", "blitz": "Blitz (5+ rushers)"}
    for h, r_h in res["transformer"].items():
        lines += [f"## {names[h]}", "", "| Moment | n | Transformer | 95% CI | AUC | GBM | GBM AUC |", "|---|---|---|---|---|---|---|"]
        for c, r in r_h.items():
            g = res.get("gbm", {}).get(h, {}).get(c)
            gtxt = f"{g['acc']:.1%} | {g.get('auc', float('nan')):.3f}" if g else "– | –"
            lines.append(f"| {label(c)} | {r['n']} | **{r['acc']:.1%}** | {r['ci'][0]:.1%}–{r['ci'][1]:.1%} | {r.get('auc', float('nan')):.3f} | {gtxt} |")
        lines.append("")
    (io.ROOT / f"reports/phase1c_{a.tag}.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
