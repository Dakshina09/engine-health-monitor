"""From a point RUL estimate to failure risk and a maintenance recommendation.

The RUL model's out-of-fold errors on the training fleet give an empirical,
prediction-dependent error distribution (engines close to failure are predicted
more precisely than healthy ones). From it we derive:

* a conservative RUL (10th percentile) and an 80% interval
* P(failure within the next RISK_HORIZON cycles)
* a risk tier and an inspection deadline
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .config import (CONSERVATIVE_Q, LEAD_TIME, RISK_HORIZON, RISK_TIERS,
                     ROUTINE_INTERVAL)

BIN_EDGES = np.array([0, 15, 30, 45, 60, 80, 100, 115, np.inf])


class UncertaintyModel:
    """Empirical residual distribution conditioned on the predicted RUL."""

    def __init__(self, min_count: int = 200):
        self.min_count = min_count
        self.residuals: dict[int, np.ndarray] = {}

    def fit(self, pred: np.ndarray, true: np.ndarray) -> "UncertaintyModel":
        pred, true = np.asarray(pred), np.asarray(true)
        res = true - pred
        idx = np.digitize(pred, BIN_EDGES) - 1
        for b in range(len(BIN_EDGES) - 1):
            r = res[idx == b]
            if len(r) < self.min_count:          # borrow neighbours for sparse bins
                lo, hi = BIN_EDGES[max(b - 1, 0)], BIN_EDGES[min(b + 2, len(BIN_EDGES) - 1)]
                r = res[(pred >= lo) & (pred < hi)]
            # keep a compact quantile sketch instead of every residual
            self.residuals[b] = np.quantile(r, np.linspace(0.005, 0.995, 199))
        return self

    def _res(self, p: float) -> np.ndarray:
        b = int(np.clip(np.digitize(p, BIN_EDGES) - 1, 0, len(BIN_EDGES) - 2))
        return self.residuals[b]

    def quantile(self, p: float, q: float) -> float:
        return float(max(0.0, p + np.quantile(self._res(p), q)))

    def prob_fail_within(self, p: float, horizon: float) -> float:
        return float(np.mean(p + self._res(p) <= horizon))


@dataclass
class Assessment:
    rul: float
    rul_p10: float
    rul_p90: float
    prob_fail: float            # P(fail within RISK_HORIZON cycles)
    risk: str
    inspect_within: int
    action: str
    headline: str

    def to_dict(self):
        return asdict(self)


ACTIONS = {
    "Critical": "Remove from service / immediate borescope inspection of HPC & fan section.",
    "High":     "Schedule on-wing inspection and pre-order HPC/fan spares; limit to essential flights.",
    "Medium":   "Increase monitoring frequency; plan inspection at the next maintenance slot.",
    "Low":      "Continue normal operation; routine check at scheduled interval.",
}


def risk_tier(prob_fail: float, rul_p10: float) -> str:
    for name, p_min, p10_max in RISK_TIERS:
        if (p_min is not None and prob_fail >= p_min) or rul_p10 <= p10_max:
            return name
    return "Low"


def assess(rul_pred: float, unc: UncertaintyModel, unit: int | str | None = None) -> Assessment:
    rul_pred = float(max(0.0, rul_pred))
    p10 = unc.quantile(rul_pred, CONSERVATIVE_Q)
    p90 = unc.quantile(rul_pred, 1 - CONSERVATIVE_Q)
    pf = unc.prob_fail_within(rul_pred, RISK_HORIZON)
    tier = risk_tier(pf, p10)

    if tier == "Critical":
        within = 0
    else:
        # inspect before the conservative (P10) life runs out, leaving LEAD_TIME to act
        within = int(max(0, np.floor((p10 - LEAD_TIME) / 5) * 5))
        within = min(within, ROUTINE_INTERVAL)   # never later than the routine check

    name = f"Engine #{unit}" if unit is not None else "Engine"
    when = "immediately" if within == 0 else f"within next {within} cycles"
    headline = (f"{name}: estimated {rul_pred:.0f} cycles remaining. "
                f"Risk: {tier}. Recommended inspection: {when}.")
    return Assessment(round(rul_pred, 1), round(p10, 1), round(p90, 1), round(pf, 3),
                      tier, within, ACTIONS[tier], headline)
