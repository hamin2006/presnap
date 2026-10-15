"""Is the transformer's snap advantage just the defender count? Compare transformer vs GBM within groups of
plays that have the same number of tracked coverage defenders (the count carries no information inside a group)."""
import numpy as np
import polars as pl
import xgboost as xgb

from presnap import baselines, features, io
from presnap.shell import KEY

D = dict(np.load(io.INTERIM / "seq_bdb26_None.npz", allow_pickle=True))
te = np.where(D["split"] == "test")[0]
n_def = (D["M"][te, 0] & (D["X"][te, 0, :, 4] > 0)).sum(1)
ens = np.mean([np.load(io.INTERIM / f"runs/main/seed{s}.npz")["cov"] for s in range(5)], 0)
man_ens = np.mean([np.load(io.INTERIM / f"runs/main/seed{s}.npz")["man"] for s in range(5)], 0)

# GBM snap predictions for the same test plays, in the same order (plays sorted by KEY in seq.build)
plays = features.personnel_counts(pl.read_parquet(io.PROCESSED / "plays.parquet")).filter(source="bdb26").sort(KEY)
b = plays.join(features.build("bdb26", {"snap": 0}), on=KEY, how="left").sort(KEY)
cols = features.SITUATION_NUM + features.SITUATION_CAT + [c for c in b.columns if c.startswith("snap_")]
out = {}
for tgt, key, P in (("cov7", "y_cov", ens), ("man", "y_man", man_ens)):
    cats = {c: sorted(b[c].drop_nulls().unique().to_list()) for c in features.SITUATION_CAT}
    full = b.with_row_index("r")
    _, Xtr, ytr, k = baselines.to_xy(full.filter(split="train"), cols, tgt, cats)
    _, Xva, yva, _ = baselines.to_xy(full.filter(split="val"), cols, tgt, cats)
    m = xgb.XGBClassifier(n_estimators=3000, learning_rate=0.03, max_depth=5, subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                          enable_categorical=True, early_stopping_rounds=150, random_state=0).fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    Xte = full.filter(split="test").select(cols).with_columns(pl.col(pl.Boolean).cast(pl.Float32)).to_pandas()
    for c, v in cats.items():
        import pandas as pd; Xte[c] = pd.Categorical(Xte[c], categories=v)
    g_pred = m.predict_proba(Xte).argmax(1)
    y = D[key][te]; t_pred = P[:, 0].argmax(1)
    ok = y >= 0
    print(f"\n{tgt} at the snap, by number of tracked coverage defenders")
    print("n_def |   n | transformer | GBM | diff")
    for g in sorted(set(n_def[ok])):
        s = ok & (n_def == g)
        if s.sum() >= 30:
            ta, ga = (t_pred[s] == y[s]).mean(), (g_pred[s] == y[s]).mean()
            print(f"{g:5d} | {s.sum():4d} | {ta:.3f} | {ga:.3f} | {ta - ga:+.3f}")
    s = ok & (n_def == 7)
    print("all  |", ok.sum(), f"| {(t_pred[ok] == y[ok]).mean():.3f} | {(g_pred[ok] == y[ok]).mean():.3f}")
