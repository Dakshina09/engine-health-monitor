"""How much do the test metrics move with the random seed?

    python -m src.seed_check            # all subsets, 5 seeds

Refits only the final LightGBM (same features) with different seeds and reports
test RMSE mean and standard deviation, so results are not quoted from one lucky run.
"""
from __future__ import annotations

import argparse
import json

import lightgbm as lgb
import numpy as np

from .config import MODEL_DIR, N_REGIMES, RUL_CAP, SUBSETS
from .data import load_test, load_train
from .features import FeatureBuilder
from .train import LGB_PARAMS, rmse


def seed_spread(subset: str, seeds=(0, 1, 2, 3, 4)) -> dict:
    tr = load_train(subset)
    fb = FeatureBuilder(N_REGIMES[subset]).fit(tr)
    X = fb.transform(tr)
    last = fb.transform(load_test(subset)).groupby("unit").tail(1)
    true = np.minimum(last["RUL_true"].to_numpy(), RUL_CAP)
    scores = []
    for s in seeds:
        m = lgb.LGBMRegressor(**{**LGB_PARAMS, "random_state": s}).fit(X[fb.feature_names], X["RUL"])
        scores.append(rmse(np.clip(m.predict(last[fb.feature_names]), 0, RUL_CAP), true))
    return {"subset": subset, "seeds": list(seeds), "test_rmse": scores,
            "mean": float(np.mean(scores)), "std": float(np.std(scores, ddof=1))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", nargs="+", default=SUBSETS)
    res = [seed_spread(s) for s in ap.parse_args().subsets]
    for r in res:
        print(f"{r['subset']}: test RMSE {r['mean']:.2f} ± {r['std']:.2f} over {len(r['seeds'])} seeds")
    (MODEL_DIR / "seed_spread.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
