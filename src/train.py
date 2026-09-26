"""Train one RUL model per C-MAPSS subset.

    python -m src.train                 # all four subsets
    python -m src.train --subsets FD001 FD003

For every subset this saves models/<subset>.joblib containing the feature
builder, LightGBM regressor, uncertainty model and evaluation metrics.
"""
from __future__ import annotations

import argparse
import json
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .config import MODEL_DIR, N_REGIMES, RUL_CAP, SUBSETS
from .data import load_test, load_train
from .features import FeatureBuilder
from .risk import UncertaintyModel, assess

LGB_PARAMS = dict(
    objective="regression", learning_rate=0.03, n_estimators=900,
    num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.6, reg_lambda=1.0, verbose=-1, random_state=42,
)


def nasa_score(y_true, y_pred) -> float:
    """PHM08 asymmetric score: late predictions (over-estimating RUL) are punished harder."""
    d = np.asarray(y_pred) - np.asarray(y_true)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def _fit_fold(train_df: pd.DataFrame, n_regimes: int):
    """Fit feature builder + LightGBM on one set of engines."""
    fb = FeatureBuilder(n_regimes).fit(train_df)
    X = fb.transform(train_df)
    model = lgb.LGBMRegressor(**LGB_PARAMS).fit(X[fb.feature_names], X["RUL"])
    return fb, model


def oof_predictions(train: pd.DataFrame, n_regimes: int, n_folds: int = 5):
    """Out-of-fold RUL predictions with NO information from the validation engines.

    For every fold the whole preprocessing chain (operating-regime clustering,
    sensor selection, normalisation statistics, Health Index weights) is refitted
    on the fold's training engines only, then applied to the held-out engines.
    """
    units = train["unit"].to_numpy()
    oof = pd.Series(np.nan, index=train.index)
    for tr, va in GroupKFold(n_folds).split(train, groups=units):
        tr_df, va_df = train.iloc[tr], train.iloc[va]
        fb, model = _fit_fold(tr_df, n_regimes)
        Xv = fb.transform(va_df)                       # sorted by unit, cycle
        pv = np.clip(model.predict(Xv[fb.feature_names]), 0, RUL_CAP)
        key = va_df.sort_values(["unit", "cycle"]).index
        oof.loc[key] = pv
    assert not oof.isna().any()
    return oof.to_numpy()


def risk_recall(true: np.ndarray, tiers: list[str]) -> dict:
    """Share of engines that really were close to failure and got flagged."""
    tiers = np.asarray(tiers)
    out = {}
    for name, limit, flagged in [("critical_recall_rul_le_15", 15, {"Critical"}),
                                 ("high_or_critical_recall_rul_le_35", 35, {"Critical", "High"})]:
        m = true <= limit
        out[name] = float(np.isin(tiers[m], list(flagged)).mean()) if m.any() else None
        out[name.replace("recall", "n")] = int(m.sum())
    return out


def train_subset(subset: str, n_folds: int = 5) -> dict:
    t0 = time.time()
    train = load_train(subset).reset_index(drop=True)
    n_reg = N_REGIMES[subset]

    # ---- 1. clean out-of-fold predictions -> uncertainty model
    oof = oof_predictions(train, n_reg, n_folds)
    sorted_train = train.sort_values(["unit", "cycle"])
    oof_sorted = pd.Series(oof, index=train.index).loc[sorted_train.index].to_numpy()
    unc = UncertaintyModel().fit(oof_sorted, sorted_train["RUL_true"].to_numpy())

    # ---- 2. final feature builder + model on all training engines
    fb, model = _fit_fold(train, n_reg)
    feats = fb.feature_names

    # ---- 3. held-out NASA test set: evaluate on the last available cycle per engine
    test = load_test(subset)
    T = fb.transform(test)
    last = T.groupby("unit").tail(1)
    pred = np.clip(model.predict(last[feats]), 0, RUL_CAP)
    true = last["RUL_true"].to_numpy()

    assessments = [assess(p, unc) for p in pred]
    p10 = np.array([a.rul_p10 for a in assessments])
    p90 = np.array([a.rul_p90 for a in assessments])
    pf = np.array([a.prob_fail for a in assessments])

    metrics = {
        "subset": subset,
        "n_train_engines": int(train["unit"].nunique()),
        "n_test_engines": int(test["unit"].nunique()),
        "n_features": len(feats),
        "sensors_used": list(fb.sensors),
        "oof_rmse_all_cycles": rmse(oof, train["RUL"]),
        "test_rmse": rmse(pred, np.minimum(true, RUL_CAP)),
        "test_mae": float(np.mean(np.abs(pred - np.minimum(true, RUL_CAP)))),
        "test_nasa_score": nasa_score(true, pred),
        "interval80_coverage": float(np.mean((true >= p10) & (true <= p90))),
        "brier_fail30": float(np.mean((pf - (true <= 30)) ** 2)),
        **risk_recall(true, [a.risk for a in assessments]),
        # safety check: recommended inspection scheduled at or after the real failure
        "inspections_after_failure": int(sum(a.inspect_within >= t for a, t in zip(assessments, true))),
        "train_seconds": round(time.time() - t0, 1),
    }
    MODEL_DIR.mkdir(exist_ok=True, parents=True)
    # Only plain Python / numpy objects are saved (LightGBM as its text model),
    # so loading needs neither pyarrow nor matching pandas/scikit-learn versions.
    joblib.dump({"format": 2, "subset": subset, "fb": fb,
                 "model_str": model.booster_.model_to_string(), "features": list(feats),
                 "unc": unc, "metrics": metrics,
                 "test_eval": {"unit": last["unit"].astype(int).tolist(),
                               "pred": pred.astype(float).tolist(),
                               "true": true.astype(float).tolist()}},
                MODEL_DIR / f"{subset}.joblib")
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", nargs="+", default=SUBSETS)
    args = ap.parse_args()
    allm = []
    for s in args.subsets:
        m = train_subset(s)
        allm.append(m)
        print(f"{s}: test RMSE={m['test_rmse']:.2f}  NASA score={m['test_nasa_score']:.0f}  "
              f"80% PI coverage={m['interval80_coverage']:.2f}  OOF RMSE={m['oof_rmse_all_cycles']:.2f}  "
              f"crit recall={m['critical_recall_rul_le_15']}  ({m['train_seconds']}s)")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(allm, indent=2))


if __name__ == "__main__":
    main()
