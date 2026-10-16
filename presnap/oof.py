"""Out-of-fold coverage predictions for every 2023 BDB play (for team reports): 3 folds by week; each fold's model is
trained on the other two (15% of their games for early stopping), so no play is scored by a model that saw it.

Run on the PC: uv run --group train python -m presnap.oof  ->  data/processed/oof_predictions.parquet
"""
import numpy as np
import polars as pl
import torch

from presnap import io
from presnap.train import TARGETS, load, predict, train_seed

FOLDS = {0: range(1, 7), 1: range(7, 13), 2: range(13, 19)}
MOMENTS = {"snap": 0, "t05": 5, "t10": 10, "t20": 20}


def main():
    D = load("bdb26", None, 0, 40)
    plays = pl.read_parquet(io.PROCESSED / "plays.parquet").filter(source="bdb26").sort(["source", "game_id", "play_id"])
    week = plays["week"].to_numpy()
    games = D["game"]
    rng = np.random.default_rng(0)
    out = []
    for f, weeks in FOLDS.items():
        test = np.isin(week, list(weeks))
        g = np.unique(games[~test])
        val_games = set(rng.choice(g, int(0.15 * len(g)), replace=False))
        split = np.where(test, "test", np.where(np.isin(games, list(val_games)), "val", "train")).astype(object)
        Df = dict(D, split=split)
        model = train_seed(Df, seed=f, epochs=60, lr=1e-3)
        te = np.where(test)[0]
        P = predict(model, Df, te)
        last = np.minimum(D["last"][te], D["X"].shape[1] - 1)
        rec = {"game_id": games[te], "play_id": plays["play_id"].to_numpy()[te]}
        for name, k in list(MOMENTS.items()) + [("throw", None)]:
            fr = last if k is None else np.minimum(k, last)
            for h in ("cov", "man", "shell"):
                p = P[h][np.arange(len(te)), fr]
                rec[f"{h}_{name}"] = p.argmax(1)
                rec[f"{h}_{name}_conf"] = p.max(1)
        out.append(pl.DataFrame(rec))
        print(f"fold {f}: {len(te)} plays scored", flush=True)
        del model; torch.cuda.empty_cache()
    pl.concat(out).write_parquet(io.PROCESSED / "oof_predictions.parquet")


if __name__ == "__main__":
    main()
