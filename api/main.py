"""FastAPI backend for the Engine Health Monitor.

    uvicorn api.main:app --reload --port 8000
    docs: http://localhost:8000/docs
"""
from __future__ import annotations

from typing import List

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.config import LEAD_TIME, RISK_HORIZON, RISK_TIERS
from src.service import available_subsets, get_model

app = FastAPI(title="Engine Health Monitor API",
              description="C-MAPSS turbofan RUL prediction → failure risk → maintenance recommendation",
              version="1.0.0")


def _model(subset: str):
    if subset not in available_subsets():
        raise HTTPException(404, f"No trained model for {subset}. Run `python -m src.train`.")
    return get_model(subset)


@app.get("/health")
def health():
    return {"status": "ok", "subsets": available_subsets()}


@app.get("/policy")
def policy():
    return {"risk_horizon_cycles": RISK_HORIZON, "lead_time_cycles": LEAD_TIME,
            "tiers": [{"tier": t, "min_prob_fail": p, "max_conservative_rul": r} for t, p, r in RISK_TIERS]}


@app.get("/fleet/{subset}")
def fleet(subset: str, risk: str | None = Query(None, description="Filter: Critical/High/Medium/Low")):
    f = _model(subset).fleet
    if risk:
        f = f[f["risk"].str.lower() == risk.lower()]
    return {"subset": subset, "count": len(f), "engines": f.to_dict(orient="records")}


@app.get("/fleet/{subset}/summary")
def fleet_summary(subset: str):
    f = _model(subset).fleet
    return {"subset": subset, "engines": len(f),
            "by_risk": f["risk"].value_counts().to_dict(),
            "median_rul": float(f["rul"].median()),
            "due_within_20": int((f["inspect_within"] <= 20).sum())}


@app.get("/engine/{subset}/{unit}")
def engine(subset: str, unit: int):
    try:
        return _model(subset).engine(unit)
    except KeyError:
        raise HTTPException(404, f"Engine {unit} not found in {subset}")


@app.get("/engine/{subset}/{unit}/explain")
def explain(subset: str, unit: int, top: int = 8):
    try:
        return _model(subset).explain(unit, top)
    except KeyError:
        raise HTTPException(404, f"Engine {unit} not found in {subset}")


@app.get("/model/{subset}")
def model_info(subset: str):
    m = _model(subset)
    return {"metrics": m.metrics, "global_importance": m.global_importance(),
            "test_eval": m.test_eval.round(1).to_dict(orient="records")}


class Cycle(BaseModel):
    cycle: int
    setting1: float
    setting2: float
    setting3: float
    s1: float; s2: float; s3: float; s4: float; s5: float; s6: float; s7: float
    s8: float; s9: float; s10: float; s11: float; s12: float; s13: float; s14: float
    s15: float; s16: float; s17: float; s18: float; s19: float; s20: float; s21: float


class PredictRequest(BaseModel):
    subset: str = Field("FD001", description="Which fleet model to use (matches operating profile)")
    engine_id: str = "new"
    cycles: List[Cycle] = Field(..., min_length=1, description="Full sensor history, oldest first")


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        return _model(req.subset).predict_raw([c.model_dump() for c in req.cycles], unit=req.engine_id)
    except ValueError as e:
        raise HTTPException(422, str(e))
