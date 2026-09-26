"""Health / degradation feature engineering.

Steps (all causal: every feature at cycle t only uses cycles <= t, so the same
code serves training and live, streaming inference):

1. Operating-regime detection   (KMeans on the 3 operational settings)
2. Regime normalisation         (z-score each sensor within its flight condition)
3. Sensor selection             (drop sensors that carry no information)
4. Time-series features         (rolling mean/std, trend slope, EWMA, drift from baseline)
5. Health Index (HI)            (linear sensor fusion: 1 = healthy, 0 = failure)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

from .config import ROLL_WINDOWS, SENSOR_COLS, SETTING_COLS, SLOPE_WINDOW

BASELINE_CYCLES = 10


class FeatureBuilder:
    """Fitted state is stored as plain Python lists and numpy arrays only.

    No pandas or scikit-learn objects are kept after fit(), so a saved model
    loads with any pandas/scikit-learn version and does not need pyarrow.
    """

    def __init__(self, n_regimes: int = 1, random_state: int = 42):
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.centers: np.ndarray | None = None       # (n_regimes, 3) KMeans centroids
        self.sensors: list[str] = []
        self.mean: np.ndarray | None = None          # (n_regimes, n_sensors)
        self.std: np.ndarray | None = None           # (n_regimes, n_sensors)
        self.hi_coef: np.ndarray | None = None       # (n_sensors,)
        self.hi_intercept: float = 0.0
        self.feature_names: list[str] = []

    # ------------------------------------------------------------------ fitting
    def _regimes(self, df: pd.DataFrame) -> np.ndarray:
        if self.n_regimes == 1:
            return np.zeros(len(df), dtype=int)
        x = df[SETTING_COLS].to_numpy(dtype=float)
        d = ((x[:, None, :] - self.centers[None, :, :]) ** 2).sum(-1)
        return d.argmin(1)                           # same rule as KMeans.predict

    def fit(self, train: pd.DataFrame) -> "FeatureBuilder":
        """Learn regimes, sensor selection, normalisation and the Health Index.

        `train` must be run-to-failure data with a `life_frac` column. life_frac
        needs the eventual failure cycle, so it exists only for training engines:
        it chooses which cycles count as "clearly healthy" / "clearly failed"
        when fitting the HI weights. transform() never uses it, so inference on
        an engine still in service needs sensor data only.
        """
        if self.n_regimes > 1:
            km = KMeans(self.n_regimes, n_init=10, random_state=self.random_state)
            km.fit(train[SETTING_COLS].to_numpy(dtype=float))
            self.centers = np.asarray(km.cluster_centers_, dtype=float)
        reg = self._regimes(train)

        # informative sensors: must vary *within* a flight regime
        keep = []
        for s in SENSOR_COLS:
            v = pd.Series(train[s].to_numpy())
            if v.groupby(reg).nunique().max() > 3 and v.groupby(reg).std().mean() > 1e-6:
                keep.append(s)
        self.sensors = keep

        vals = train[keep].to_numpy(dtype=float)
        self.mean = np.stack([vals[reg == r].mean(0) for r in range(self.n_regimes)])
        std = np.stack([vals[reg == r].std(0, ddof=1) for r in range(self.n_regimes)])
        self.std = np.where(std > 0, std, 1.0)

        # Health index: fit only on clearly-healthy (first 20% of life) and
        # clearly-failed (last 5%) cycles, then apply everywhere.
        z = self._normalise(train)
        life = train["life_frac"].to_numpy()
        mask = (life <= 0.20) | (life >= 0.95)
        y = (life[mask] <= 0.20).astype(float)
        lr = LinearRegression().fit(z.to_numpy()[mask], y)
        self.hi_coef = np.asarray(lr.coef_, dtype=float)
        self.hi_intercept = float(lr.intercept_)
        return self

    # ---------------------------------------------------------------- transform
    def _normalise(self, df: pd.DataFrame) -> pd.DataFrame:
        reg = self._regimes(df)
        z = (df[self.sensors].to_numpy(dtype=float) - self.mean[reg]) / self.std[reg]
        return pd.DataFrame(z, columns=self.sensors, index=df.index)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["unit", "cycle"]).reset_index(drop=True)
        z = self._normalise(df)
        out = {"unit": df["unit"], "cycle": df["cycle"]}
        g = z.groupby(df["unit"])
        t = df["cycle"].astype(float)

        for s in self.sensors:
            x = z[s]
            gx = g[s]
            out[f"{s}_z"] = x
            for w in ROLL_WINDOWS:
                out[f"{s}_mean{w}"] = gx.transform(lambda v, w=w: v.rolling(w, min_periods=1).mean())
            out[f"{s}_std15"] = gx.transform(lambda v: v.rolling(15, min_periods=2).std()).fillna(0)
            out[f"{s}_ewm"] = gx.transform(lambda v: v.ewm(alpha=0.1).mean())
            base = gx.transform(lambda v: v.expanding().mean().where(np.arange(len(v)) < BASELINE_CYCLES).ffill())
            out[f"{s}_drift"] = out[f"{s}_ewm"] - base
            out[f"{s}_slope"] = _rolling_slope(x, t, df["unit"], SLOPE_WINDOW)

        hi_raw = pd.Series(z[self.sensors].to_numpy() @ self.hi_coef + self.hi_intercept, index=df.index)
        out["HI_raw"] = hi_raw
        out["HI"] = hi_raw.groupby(df["unit"]).transform(lambda v: v.ewm(alpha=0.1).mean())
        out["HI_slope"] = _rolling_slope(out["HI"], t, df["unit"], SLOPE_WINDOW)
        out["HI_min"] = out["HI"].groupby(df["unit"]).cummin()

        feats = pd.DataFrame(out)
        self.feature_names = [c for c in feats.columns if c not in ("unit", "cycle", "HI_raw")]
        # carry labels / raw columns through if present
        for c in ("RUL", "RUL_true"):
            if c in df:
                feats[c] = df[c].to_numpy()
        return feats


def _rolling_slope(x: pd.Series, t: pd.Series, unit: pd.Series, w: int) -> pd.Series:
    """Least-squares slope of x vs t over the last w cycles (per unit, causal)."""
    frame = pd.DataFrame({"x": x, "t": t, "xt": x * t, "tt": t * t, "unit": unit})
    r = frame.groupby("unit")[["x", "t", "xt", "tt"]].transform(
        lambda v: v.rolling(w, min_periods=3).mean()
    )
    var = r["tt"] - r["t"] ** 2
    slope = (r["xt"] - r["x"] * r["t"]) / var.replace(0, np.nan)
    return slope.fillna(0.0)
