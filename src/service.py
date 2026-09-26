"""Inference service shared by the FastAPI backend and the Streamlit dashboard.

Treats each subset's *test* file as the "live fleet": engines currently in
service whose sensor history stops at their latest flight cycle.
"""
from __future__ import annotations

from functools import lru_cache

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

from .config import MODEL_DIR, RAW_COLS, RUL_CAP, SENSOR_INFO, SUBSETS
from .data import load_test
from .risk import assess

RISK_ORDER = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}


def _group_of(feature: str) -> str:
    if feature.startswith("HI"):
        return "HI"
    if feature == "cycle":
        return "cycle"
    return feature.split("_")[0]


def _label(group: str) -> str:
    if group == "HI":
        return "Health Index (fused)"
    if group == "cycle":
        return "Engine age (cycles flown)"
    tag, desc = SENSOR_INFO[group]
    return f"{tag}: {desc}"


class SubsetModel:
    def __init__(self, subset: str):
        b = joblib.load(MODEL_DIR / f"{subset}.joblib")
        self.subset = subset
        if b.get("format") != 2:
            raise RuntimeError(f"{subset}.joblib was saved by an older version. "
                               "Retrain with: python -m src.train")
        self.fb, self.features = b["fb"], b["features"]
        self.model = lgb.Booster(model_str=b["model_str"])
        self.unc, self.metrics = b["unc"], b["metrics"]
        self.test_eval = pd.DataFrame(b["test_eval"])
        self._explainer = None

        raw = load_test(subset)
        self.raw = raw
        self.X = self.fb.transform(raw)
        self.X["RUL_pred"] = np.clip(self.model.predict(self.X[self.features]), 0, RUL_CAP)
        self.fleet = self._build_fleet()

    # ------------------------------------------------------------------ fleet
    def _build_fleet(self) -> pd.DataFrame:
        last = self.X.groupby("unit").tail(1)
        rows = []
        for _, r in last.iterrows():
            a = assess(r["RUL_pred"], self.unc, unit=int(r["unit"])).to_dict()
            a.update(unit=int(r["unit"]), cycles_flown=int(r["cycle"]),
                     health_index=round(float(r["HI"]), 3),
                     true_rul=float(r["RUL_true"]) if "RUL_true" in r else None)
            rows.append(a)
        f = pd.DataFrame(rows)
        f["_o"] = f["risk"].map(RISK_ORDER)
        return f.sort_values(["_o", "rul"]).drop(columns="_o").reset_index(drop=True)

    # ----------------------------------------------------------------- engine
    def engine(self, unit: int) -> dict:
        h = self.X[self.X["unit"] == unit]
        if h.empty:
            raise KeyError(unit)
        a = self.fleet[self.fleet["unit"] == unit].iloc[0].to_dict()
        # rolling uncertainty band for every past cycle (how the estimate evolved)
        band = [(self.unc.quantile(p, 0.1), self.unc.quantile(p, 0.9)) for p in h["RUL_pred"]]
        raw = self.raw[self.raw["unit"] == unit]
        history = {
            "cycle": h["cycle"].tolist(),
            "rul_pred": h["RUL_pred"].round(1).tolist(),
            "rul_p10": [round(b[0], 1) for b in band],
            "rul_p90": [round(b[1], 1) for b in band],
            "health_index": h["HI"].round(4).tolist(),
            "rul_true": h["RUL_true"].tolist() if "RUL_true" in h else None,
        }
        sensors = {s: raw[s].round(4).tolist() for s in self.fb.sensors}
        return {"subset": self.subset, **a, "history": history, "sensors": sensors}

    # ---------------------------------------------------------------- explain
    @property
    def explainer(self):
        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self.model)
        return self._explainer

    def explain(self, unit: int, top: int = 8) -> dict:
        row = self.X[self.X["unit"] == unit].tail(1)
        if row.empty:
            raise KeyError(unit)
        sv = self.explainer.shap_values(row[self.features])[0]
        base = float(np.ravel(self.explainer.expected_value)[0])
        contrib = pd.Series(sv, index=self.features)
        grouped = contrib.groupby(contrib.index.map(_group_of)).sum()
        grouped = grouped.reindex(grouped.abs().sort_values(ascending=False).index)
        rest = float(grouped.iloc[top:].sum())
        grouped = grouped.iloc[:top]
        drivers = []
        for g, v in grouped.items():
            d = {"sensor": g, "label": _label(g), "shap_cycles": round(float(v), 2)}
            if g in self.fb.sensors:
                drift = float(row[f"{g}_drift"].iloc[0])
                d["drift_sigma"] = round(drift, 2)
                d["note"] = (f"{'above' if drift > 0 else 'below'} its early-life baseline by "
                             f"{abs(drift):.1f}σ")
            d["effect"] = "shortens life" if v < 0 else "extends life"
            drivers.append(d)
        if abs(rest) >= 0.005:
            drivers.append({"sensor": "other", "label": "All remaining sensors combined",
                            "shap_cycles": round(rest, 2),
                            "effect": "shortens life" if rest < 0 else "extends life"})
        # SHAP explains the raw model output: base_value + sum(shap) == model_output.
        # The displayed prediction is that output clipped to [0, RUL_CAP].
        return {"unit": unit, "base_value": round(base, 1),
                "model_output": round(base + float(sv.sum()), 1),
                "prediction": round(float(row["RUL_pred"].iloc[0]), 1), "drivers": drivers}

    def global_importance(self, top: int = 12) -> list[dict]:
        sample = self.X[self.features].sample(min(2000, len(self.X)), random_state=0)
        sv = self.explainer.shap_values(sample)
        imp = pd.Series(np.abs(sv).mean(0), index=self.features)
        g = imp.groupby(imp.index.map(_group_of)).sum().sort_values(ascending=False)[:top]
        return [{"sensor": k, "label": _label(k), "mean_abs_shap": round(float(v), 2)} for k, v in g.items()]

    # ---------------------------------------------------- score a new engine
    def predict_raw(self, records: list[dict], unit: int | str = "new") -> dict:
        df = pd.DataFrame(records)
        missing = [c for c in RAW_COLS if c not in df and c != "unit"]
        if missing:
            raise ValueError(f"missing columns: {missing}")
        df = df.sort_values("cycle").reset_index(drop=True)
        cyc = df["cycle"].to_numpy()
        # rolling features are computed over rows, so rows must be one per flight cycle
        if len(np.unique(cyc)) != len(cyc):
            raise ValueError("duplicate cycle numbers in the history")
        if len(cyc) > 1 and not np.all(np.diff(cyc) == 1):
            raise ValueError("cycle history must be consecutive (no gaps); send every cycle "
                             "from the first recorded one to the latest")
        vals = df[[c for c in RAW_COLS if c not in ("unit", "cycle")]].to_numpy(dtype=float)
        if not np.isfinite(vals).all():
            raise ValueError("sensor or setting values contain NaN or infinity")
        df["unit"] = 0
        X = self.fb.transform(df[RAW_COLS])
        p = float(np.clip(self.model.predict(X[self.features].tail(1))[0], 0, RUL_CAP))
        out = assess(p, self.unc, unit=unit).to_dict()
        out.update(subset=self.subset, cycles_used=int(len(df)), latest_cycle=int(cyc[-1]))
        return out


@lru_cache(maxsize=None)
def get_model(subset: str) -> SubsetModel:
    if subset not in SUBSETS:
        raise KeyError(subset)
    return SubsetModel(subset)


def available_subsets() -> list[str]:
    return [s for s in SUBSETS if (MODEL_DIR / f"{s}.joblib").exists()]
