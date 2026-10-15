"""Phase 1B baseline scorecard.

Targets (NGS labels): man/zone, 7-class coverage, played shell (0/1/2).
Models: majority class; GBM on situation only; GBM on situation + tracking at the snap / +2.0 s / the throw.
Train on weeks 1-12, early-stop on 13-15, report weeks 16-18 with 95% CIs from a game-clustered bootstrap.
Clean check: the same snap model on StatsBomb (all 22 tracked, so no missing-rusher leak), split by game.

Run (on the PC): uv run python -m presnap.baselines
"""
import json

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.metrics import f1_score, log_loss

from presnap import features, io, labels
from presnap.shell import KEY

RNG = np.random.default_rng(0)
TARGETS = {
    "man": ("man_zone", ["ZONE", "MAN"]),
    "cov7": ("coverage", labels.COVERAGE7),
    "shell": ("shell_played", [0, 1, 2]),
}


def to_xy(df, cols, target):
    col, classes = TARGETS[target]
    df = df.filter(pl.col(col).is_in(classes))
    y = df[col].replace_strict({c: i for i, c in enumerate(classes)}, return_dtype=pl.Int32).to_numpy()
    X = df.select(cols).to_pandas()
    for c in features.SITUATION_CAT:
        if c in X:
            X[c] = X[c].astype("category")
    return df, X, y, len(classes)


def boot_ci(games, correct, n=1000):
    """95% CI for accuracy, resampling games (plays within a game are correlated)."""
    g = pl.DataFrame({"g": games, "c": correct}).group_by("g").agg(k=pl.col("c").sum(), n=pl.len())
    k, m = g["k"].to_numpy(), g["n"].to_numpy()
    idx = RNG.integers(0, len(k), (n, len(k)))
    acc = k[idx].sum(1) / m[idx].sum(1)
    return float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))


def fit_eval(data, cols, target, train="train", val="val", test="test"):
    tr, Xtr, ytr, k = to_xy(data.filter(split=train), cols, target)
    va, Xva, yva, _ = to_xy(data.filter(split=val), cols, target)
    te, Xte, yte, _ = to_xy(data.filter(split=test), cols, target)
    if not cols:  # majority class
        p = np.bincount(ytr, minlength=k) / len(ytr)
        P = np.tile(p, (len(yte), 1))
    else:
        m = xgb.XGBClassifier(n_estimators=3000, learning_rate=0.03, max_depth=5, subsample=0.8, colsample_bytree=0.8,
                              min_child_weight=5, enable_categorical=True, tree_method="hist", early_stopping_rounds=150,
                              eval_metric="mlogloss" if k > 2 else "logloss", random_state=0)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        P = m.predict_proba(Xte)
    pred = P.argmax(1)
    correct = pred == yte
    lo, hi = boot_ci(te["game_id"].to_list(), correct.tolist())
    res = dict(n=len(yte), acc=round(float(correct.mean()), 4), ci=[round(lo, 4), round(hi, 4)],
               logloss=round(float(log_loss(yte, P, labels=list(range(k)))), 4),
               macro_f1=round(float(f1_score(yte, pred, average="macro")), 4))
    if "labels_agree" in te.columns and te["labels_agree"].drop_nulls().len():
        a = te["labels_agree"].fill_null(False).to_numpy()
        res["acc_agree_subset"] = round(float(correct[a].mean()), 4)
    return res


def main():
    plays = pl.read_parquet(io.PROCESSED / "plays.parquet")
    sit = features.SITUATION_NUM + features.SITUATION_CAT

    b = plays.filter(source="bdb26").join(
        features.build("bdb26", {"snap": 0, "post": ("min_last", 20), "throw": "last"}), on=KEY, how="left")
    fcols = lambda tags: [c for c in b.columns if c.split("_")[0] in tags]
    models = {
        "majority": [],
        "situation": sit,
        "gbm_snap": sit + fcols(["snap"]),
        "gbm_post_2s": sit + fcols(["snap", "post"]),
        "gbm_throw": sit + fcols(["snap", "throw"]),
    }
    score = {t: {name: fit_eval(b, cols, t) for name, cols in models.items()} for t in TARGETS}

    s = plays.filter((pl.col("source") == "statsbomb") & pl.col("sb_quality_ok")).join(features.build("statsbomb", {"snap": 0}), on=KEY, how="left")
    sit_sb = [c for c in sit if s[c].null_count() < s.height]  # receiver_alignment / FTN motion don't exist for 2018-22
    clean = {t: {name: fit_eval(s, cols, t) for name, cols in
                 {"majority": [], "situation": sit_sb, "gbm_snap": sit_sb + [c for c in s.columns if c.startswith("snap_")]}.items()}
             for t in ("man", "cov7", "shell")}

    out = dict(bdb26_weeks16_18=score, statsbomb_clean=clean)
    (io.ROOT / "reports/phase1b_scorecard.json").write_text(json.dumps(out, indent=2))
    lines = ["# Phase 1B baseline scorecard", "",
             "BDB 2026 (2023 season, NGS labels): train weeks 1–12, early-stop on 13–15, **test on 16–18**. "
             "Accuracy with a 95% game-clustered bootstrap CI; `agree` = accuracy on test plays where NGS and FTN give the same label.", ""]
    for t in TARGETS:
        lines += [f"## {t}", "", "| Model | n | Accuracy | 95% CI | Log loss | Macro F1 | Acc (agree) |", "|---|---|---|---|---|---|---|"]
        for name, r in score[t].items():
            lines.append(f"| {name} | {r['n']} | **{r['acc']:.1%}** | {r['ci'][0]:.1%}–{r['ci'][1]:.1%} | {r['logloss']:.3f} | {r['macro_f1']:.3f} | {r.get('acc_agree_subset', float('nan')):.1%} |")
        lines.append("")
    lines += ["## StatsBomb clean check (all 22 tracked, 2018–22, split by game)", "",
              "| Target | Model | n | Accuracy | 95% CI | Log loss |", "|---|---|---|---|---|---|"]
    for t, rs in clean.items():
        for name, r in rs.items():
            lines.append(f"| {t} | {name} | {r['n']} | **{r['acc']:.1%}** | {r['ci'][0]:.1%}–{r['ci'][1]:.1%} | {r['logloss']:.3f} |")
    (io.ROOT / "reports/phase1b_scorecard.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
