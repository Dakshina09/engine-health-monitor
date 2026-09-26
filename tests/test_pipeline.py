"""Correctness checks for the pipeline.   Run:  python -m pytest -q

1. Features are causal, so the dashboard's RUL trajectory (computed on the full
   history) equals what a live, cycle-by-cycle system would have shown.
2. predict_raw() on an engine's history gives the same answer as the fleet view,
   and rejects bad input.
3. SHAP contributions add up exactly to the model output.
4. Saved models load without pyarrow.
5. OOF predictions never see validation engines during feature fitting.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_test, load_train  # noqa: E402
from src.features import FeatureBuilder  # noqa: E402
from src.service import get_model  # noqa: E402

SUBSET = "FD002"   # six flight conditions: exercises regime clustering too


@pytest.fixture(scope="module")
def m():
    return get_model(SUBSET)


def test_features_are_causal(m):
    raw = m.raw[m.raw["unit"] == m.raw["unit"].iloc[0]]
    full = m.fb.transform(raw)
    for t in (1, 5, 17, 40, len(raw) - 1):
        part = m.fb.transform(raw.iloc[:t])
        a = part[m.features].iloc[-1].to_numpy()
        b = full[m.features].iloc[t - 1].to_numpy()
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-9, err_msg=f"cycle {t}")


def test_trajectory_matches_streaming(m):
    unit = int(m.fleet["unit"].iloc[3])
    raw = m.raw[m.raw["unit"] == unit]
    hist = m.engine(unit)["history"]
    for t in (10, 50, len(raw)):
        cols = [c for c in raw.columns if c not in ("unit", "RUL", "RUL_true")]
        live = m.predict_raw(raw.iloc[:t][cols].to_dict("records"), unit=unit)
        assert abs(live["rul"] - round(hist["rul_pred"][t - 1], 1)) <= 0.051


def test_predict_raw_matches_fleet(m):
    row = m.fleet.iloc[0]
    raw = m.raw[m.raw["unit"] == row["unit"]].drop(columns=["unit", "RUL", "RUL_true"])
    out = m.predict_raw(raw.to_dict("records"), unit=int(row["unit"]))
    assert out["rul"] == row["rul"] and out["risk"] == row["risk"]
    assert out["inspect_within"] == row["inspect_within"]
    # order of submitted rows must not matter
    shuffled = raw.sample(frac=1, random_state=0).to_dict("records")
    assert m.predict_raw(shuffled, unit=int(row["unit"]))["rul"] == row["rul"]


@pytest.mark.parametrize("bad, msg", [
    ("gap", "consecutive"), ("dup", "duplicate"), ("nan", "NaN"), ("missing", "missing"),
])
def test_predict_raw_rejects_bad_input(m, bad, msg):
    raw = m.raw[m.raw["unit"] == 1].drop(columns=["unit", "RUL", "RUL_true"]).head(20)
    if bad == "gap":
        raw = raw.drop(raw.index[5])
    elif bad == "dup":
        raw.loc[raw.index[3], "cycle"] = raw["cycle"].iloc[2]
    elif bad == "nan":
        raw.loc[raw.index[4], "s11"] = np.nan
    elif bad == "missing":
        raw = raw.drop(columns=["s4"])
    with pytest.raises(ValueError, match=msg):
        m.predict_raw(raw.to_dict("records"))


def test_shap_adds_up(m):
    for unit in m.fleet["unit"].head(5):
        e = m.explain(int(unit), top=4)
        total = e["base_value"] + sum(d["shap_cycles"] for d in e["drivers"])
        assert abs(total - e["model_output"]) < 0.2
        assert e["prediction"] == pytest.approx(np.clip(e["model_output"], 0, 125), abs=0.11)


def test_models_load_without_pyarrow():
    code = ("import sys; sys.modules['pyarrow'] = None; import joblib\n"
            "for s in ['FD001','FD002','FD003','FD004']: joblib.load(f'models/{s}.joblib')\n"
            "print('ok')")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert r.stdout.strip() == "ok", r.stderr[-500:]


def test_oof_feature_fitting_excludes_validation_engines(monkeypatch):
    from src import train as T
    seen = []
    real_fit = FeatureBuilder.fit

    def spy(self, df):
        seen.append(set(df["unit"]))
        return real_fit(self, df)

    monkeypatch.setattr(FeatureBuilder, "fit", spy)
    tr = load_train("FD001")
    tr = tr[tr["unit"] <= 20].reset_index(drop=True)        # small + fast
    monkeypatch.setattr(T, "LGB_PARAMS", {**T.LGB_PARAMS, "n_estimators": 20})
    from sklearn.model_selection import GroupKFold
    folds = list(GroupKFold(5).split(tr, groups=tr["unit"]))
    T.oof_predictions(tr, 1, 5)
    assert len(seen) == 5
    for fitted_units, (_, va) in zip(seen, folds):
        assert fitted_units.isdisjoint(set(tr.iloc[va]["unit"]))
