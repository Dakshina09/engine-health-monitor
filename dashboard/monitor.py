"""Engine Health Monitor: main dashboard page (loaded by dashboard/app.py).

Talks to the FastAPI backend at $API_URL (default http://localhost:8000).
If the API is not running it falls back to calling the model in-process,
so the dashboard also works standalone.
"""
from __future__ import annotations

import os
import time
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import LEAD_TIME, RISK_HORIZON, SENSOR_INFO  # noqa: E402

API_URL = os.getenv("API_URL", "http://localhost:8000")

RISK_COLOR = {"Critical": "#d03b3b", "High": "#ec835a", "Medium": "#fab219", "Low": "#0ca30c"}
BLUE, INK, MUTED, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#e7e6e2"


# ----------------------------------------------------------------- data access
class Backend:
    """API client with automatic in-process fallback.

    Every call tries the FastAPI backend first. If it is not running (or stops
    while the dashboard is open) the call is answered by the model in-process
    instead, and the API is retried again after RETRY_SECONDS.
    """

    RETRY_SECONDS = 30

    def __init__(self):
        self._api_down_since: float | None = None

    @property
    def mode(self) -> str:
        return "api" if self._api_available() else "local"

    def _api_available(self) -> bool:
        if self._api_down_since is not None and time.time() - self._api_down_since < self.RETRY_SECONDS:
            return False
        try:
            requests.get(f"{API_URL}/health", timeout=(1.0, 5.0)).raise_for_status()
            self._api_down_since = None
            return True
        except requests.RequestException:
            self._api_down_since = time.time()
            return False

    def _call(self, path, local):
        if self._api_down_since is None or time.time() - self._api_down_since >= self.RETRY_SECONDS:
            try:
                r = requests.get(f"{API_URL}{path}", timeout=(1.0, 120.0))
                r.raise_for_status()
                self._api_down_since = None
                return r.json(), True
            except requests.RequestException:
                self._api_down_since = time.time()
        return local(), False

    def subsets(self):
        from src.service import available_subsets
        out, via_api = self._call("/health", lambda: {"subsets": available_subsets()})
        return out["subsets"]

    def fleet(self, s):
        from src.service import get_model
        out, via_api = self._call(f"/fleet/{s}", lambda: get_model(s).fleet.copy())
        return pd.DataFrame(out["engines"]) if via_api else out

    def engine(self, s, u):
        from src.service import get_model
        return self._call(f"/engine/{s}/{u}", lambda: get_model(s).engine(u))[0]

    def explain(self, s, u):
        from src.service import get_model
        return self._call(f"/engine/{s}/{u}/explain", lambda: get_model(s).explain(u))[0]

    def model(self, s):
        from src.service import get_model

        def local():
            m = get_model(s)
            return {"metrics": m.metrics, "global_importance": m.global_importance(),
                    "test_eval": m.test_eval.to_dict(orient="records")}
        return self._call(f"/model/{s}", local)[0]


@st.cache_resource
def backend():
    return Backend()


@st.cache_data(show_spinner="Scoring fleet…")
def fleet(s):
    return backend().fleet(s)


@st.cache_data
def engine(s, u):
    return backend().engine(s, u)


@st.cache_data(show_spinner="Computing SHAP explanation…")
def explain(s, u):
    return backend().explain(s, u)


@st.cache_data(show_spinner="Loading model report…")
def model_info(s):
    return backend().model(s)


def style(fig, h=320, legend=True):
    fig.update_layout(
        height=h, margin=dict(l=10, r=10, t=64 if legend else 40, b=10), plot_bgcolor="#fcfcfb", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK, size=12), title=dict(y=0.98, yanchor="top"), hovermode="x unified" if legend else "closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0, bgcolor="rgba(0,0,0,0)"), showlegend=legend,
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    return fig


def risk_badge(tier):
    c = RISK_COLOR[tier]
    return (f"<span style='display:inline-block;background:{c}1f;color:{INK};border:1px solid {c};"
            f"border-left:4px solid {c};padding:1px 8px;border-radius:3px;font-weight:600;"
            f"font-size:0.9rem'>{tier}</span>")


# ------------------------------------------------------------------------ UI
st.markdown("""
<style>
.block-container {padding-top: 1.6rem;}
.alert {border-left: 6px solid var(--c); background: #fcfcfb; border-radius: 10px;
        padding: 18px 22px; box-shadow: 0 1px 3px rgba(0,0,0,.08); margin-bottom: 8px;}
.alert h2 {margin: 0 0 6px 0; font-size: 1.55rem; color:#0b0b0b}
.alert .row {font-size: 1.15rem; margin: 4px 0; color:#0b0b0b}
.alert .muted {color:#52514e; font-size: .95rem}
</style>""", unsafe_allow_html=True)

be = backend()
subsets = be.subsets()
if not subsets:
    st.error("No trained models found. Run `python -m src.train` first.")
    st.stop()

with st.sidebar:
    st.title("Engine Health")
    st.caption(f"Backend: **{'FastAPI' if be.mode == 'api' else 'in-process'}**")
    subset = st.selectbox("Fleet (C-MAPSS subset)", subsets,
                          help="FD001/3: sea-level only · FD002/4: six flight conditions · "
                               "FD003/4: HPC + fan faults")
    F = fleet(subset)
    risk_filter = st.multiselect("Risk tiers", list(RISK_COLOR), default=list(RISK_COLOR))
    view = F[F["risk"].isin(risk_filter)]
    units = view["unit"].tolist() or F["unit"].tolist()
    unit = st.selectbox("Engine", units, format_func=lambda u: f"Engine #{u}  ·  "
                        f"{F.loc[F.unit == u, 'risk'].iloc[0]}  ·  {F.loc[F.unit == u, 'rul'].iloc[0]:.0f} cyc")
    st.divider()
    st.caption(f"**Policy.** Risk = P(failure within {RISK_HORIZON} cycles) and the conservative "
               f"(P10) RUL. Inspection deadline = P10 RUL minus a {LEAD_TIME}-cycle planning lead time.")

st.title("Turbofan Engine Health Monitor")
st.caption("NASA C-MAPSS · sensor data → health features → RUL → failure risk → maintenance action")
st.info("All engines shown are simulated test units from the NASA C-MAPSS dataset, not real aircraft. Predictions are for demonstration and must not be used for real maintenance decisions.")

counts = F["risk"].value_counts()
k = st.columns(5)
k[0].metric("Engines monitored", len(F))
k[1].metric("Critical", int(counts.get("Critical", 0)))
k[2].metric("High risk", int(counts.get("High", 0)))
k[3].metric("Inspection due ≤ 20 cycles", int((F["inspect_within"] <= 20).sum()))
k[4].metric("Median RUL (cycles)", f"{F['rul'].median():.0f}")

tab_engine, tab_fleet, tab_model = st.tabs(["Engine detail", "Fleet overview", "Model & explainability"])

# ================================================================ ENGINE TAB
with tab_engine:
    E = engine(subset, unit)
    X = explain(subset, unit)
    tier, c = E["risk"], RISK_COLOR[E["risk"]]
    when = "immediately" if E["inspect_within"] == 0 else f"within next {E['inspect_within']} cycles"
    st.markdown(f"""
    <div class="alert" style="--c:{c}">
      <h2>Engine #{unit}: estimated {E['rul']:.0f} cycles remaining</h2>
      <div class="row">Risk: {risk_badge(tier)}</div>
      <div class="row">Recommended inspection: <b>{when}</b></div>
      <div class="muted" style="margin-top:8px">{E['action']}</div>
    </div>""", unsafe_allow_html=True)

    m = st.columns(4)
    m[0].metric("RUL 80% interval", f"{E['rul_p10']:.0f} to {E['rul_p90']:.0f} cyc")
    m[1].metric(f"P(failure ≤ {RISK_HORIZON} cycles)", f"{E['prob_fail']:.0%}")
    m[2].metric("Health index", f"{E['health_index']:.2f}", help="1 = as-new, 0 = failure threshold")
    m[3].metric("Cycles flown", E["cycles_flown"])

    h = E["history"]
    left, right = st.columns([3, 2])
    with left:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=h["cycle"] + h["cycle"][::-1], y=h["rul_p90"] + h["rul_p10"][::-1],
                                 fill="toself", fillcolor="rgba(42,120,214,0.14)", line=dict(width=0),
                                 name="80% interval", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=h["cycle"], y=h["rul_pred"], name="Predicted RUL",
                                 line=dict(color=BLUE, width=2)))
        if h.get("rul_true"):
            fig.add_trace(go.Scatter(x=h["cycle"], y=[min(v, 125) for v in h["rul_true"]],
                                     name="Actual RUL (capped, NASA label)",
                                     line=dict(color=MUTED, width=1.5, dash="dot")))
        fig.add_hline(y=RISK_HORIZON, line=dict(color=RISK_COLOR["Critical"], width=1, dash="dash"),
                      annotation_text=f"{RISK_HORIZON}-cycle risk horizon", annotation_position="top left")
        fig.update_layout(title="Remaining useful life: how the estimate evolved")
        fig.update_xaxes(title="Flight cycle")
        fig.update_yaxes(title="RUL (cycles)", rangemode="tozero")
        st.plotly_chart(style(fig, 360), width="stretch")
    with right:
        fig = go.Figure(go.Scatter(x=h["cycle"], y=h["health_index"], name="Health index",
                                   line=dict(color=BLUE, width=2), fill="tozeroy",
                                   fillcolor="rgba(42,120,214,0.08)"))
        fig.update_layout(title="Health index (sensor fusion)")
        fig.update_xaxes(title="Flight cycle")
        fig.update_yaxes(title="HI", range=[min(-0.1, min(h["health_index"]) - 0.05), 1.15])
        st.plotly_chart(style(fig, 360, legend=False), width="stretch")

    left, right = st.columns([2, 3])
    with left:
        d = pd.DataFrame(X["drivers"]).iloc[::-1]
        colors = [RISK_COLOR["Critical"] if v < 0 else BLUE for v in d["shap_cycles"]]
        fig = go.Figure(go.Bar(
            x=d["shap_cycles"], y=d["sensor"].map(lambda s: SENSOR_INFO[s][0] if s in SENSOR_INFO else {"HI": "Health idx", "cycle": "Age", "other": "Other"}[s]),
            orientation="h", marker=dict(color=colors, cornerradius=4),
            customdata=d["label"], hovertemplate="%{customdata}<br>%{x:+.1f} cycles<extra></extra>"))
        fig.add_vline(x=0, line=dict(color=MUTED, width=1))
        fig.update_layout(title=f"SHAP drivers: fleet avg {X['base_value']:.0f} → {X['model_output']:.0f} cycles")
        fig.update_xaxes(title="Effect on predicted RUL (cycles). Red shortens life")
        st.plotly_chart(style(fig, 360, legend=False), width="stretch")
    with right:
        sens = [dr["sensor"] for dr in X["drivers"] if dr["sensor"] in SENSOR_INFO][:4]
        st.markdown("**Top driver sensors: raw readings**")
        cols = st.columns(2)
        for i, s in enumerate(sens):
            y = pd.Series(E["sensors"][s])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=h["cycle"], y=y, line=dict(color="#b9b8b2", width=1), name="raw"))
            fig.add_trace(go.Scatter(x=h["cycle"], y=y.rolling(10, min_periods=1).mean(),
                                     line=dict(color=BLUE, width=2), name="10-cycle mean"))
            tag, desc = SENSOR_INFO[s]
            fig.update_layout(title=dict(text=f"{tag} · {desc}", font=dict(size=12)))
            cols[i % 2].plotly_chart(style(fig, 180, legend=False), width="stretch")
        notes = [f"- **{SENSOR_INFO[dr['sensor']][0]}** ({dr['effect']}, {dr['shap_cycles']:+.1f} cyc): {dr['note']}"
                 for dr in X["drivers"] if dr.get("note")][:4]
        if notes:
            st.markdown("\n".join(notes))

# ================================================================= FLEET TAB
with tab_fleet:
    left, right = st.columns([3, 2])
    with left:
        fig = go.Figure()
        for t in RISK_COLOR:
            sub = view[view["risk"] == t]
            if sub.empty:
                continue
            fig.add_trace(go.Scatter(
                x=sub["rul"], y=sub["prob_fail"], mode="markers", name=t,
                marker=dict(color=RISK_COLOR[t], size=9, line=dict(color="#fcfcfb", width=2)),
                customdata=sub[["unit", "inspect_within"]],
                hovertemplate="Engine #%{customdata[0]}<br>RUL %{x:.0f} cycles<br>"
                              "P(fail ≤30) %{y:.0%}<br>Inspect within %{customdata[1]} cycles<extra>" + t + "</extra>"))
        fig.update_layout(title="Fleet risk map", hovermode="closest")
        fig.update_xaxes(title="Predicted RUL (cycles)")
        fig.update_yaxes(title=f"P(failure within {RISK_HORIZON} cycles)", tickformat=".0%")
        st.plotly_chart(style(fig, 380), width="stretch")
    with right:
        b = view["inspect_within"].clip(upper=50)
        bins = pd.cut(b, [-1, 0, 10, 20, 30, 40, 50], labels=["Now", "1-10", "11-20", "21-30", "31-40", "41-50+"])
        cnt = bins.value_counts().sort_index()
        fig = go.Figure(go.Bar(x=cnt.index.astype(str), y=cnt.values, marker=dict(color=BLUE, cornerradius=4),
                               hovertemplate="%{x} cycles: %{y} engines<extra></extra>"))
        fig.update_layout(title="Maintenance workload: inspections due")
        fig.update_xaxes(title="Inspection due (cycles from now)")
        fig.update_yaxes(title="Engines")
        st.plotly_chart(style(fig, 380, legend=False), width="stretch")

    table = view[["unit", "risk", "rul", "rul_p10", "rul_p90", "prob_fail", "inspect_within",
                  "health_index", "cycles_flown", "action"]].rename(columns={
        "unit": "Engine", "risk": "Risk", "rul": "RUL", "rul_p10": "RUL P10", "rul_p90": "RUL P90",
        "prob_fail": f"P(fail ≤{RISK_HORIZON})", "inspect_within": "Inspect within",
        "health_index": "Health idx", "cycles_flown": "Cycles flown", "action": "Recommended action"})
    st.dataframe(
        table, hide_index=True, width="stretch", height=420,
        column_config={
            f"P(fail ≤{RISK_HORIZON})": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1),
            "RUL": st.column_config.NumberColumn(format="%.0f"),
            "Inspect within": st.column_config.NumberColumn(format="%d cyc"),
        })
    st.download_button("Download maintenance plan (CSV)", view.to_csv(index=False),
                       file_name=f"maintenance_plan_{subset}.csv")

# ================================================================= MODEL TAB
with tab_model:
    M = model_info(subset)
    mt = M["metrics"]
    c = st.columns(4)
    c[0].metric("Test RMSE (cycles)", f"{mt['test_rmse']:.2f}")
    c[1].metric("Test MAE", f"{mt['test_mae']:.2f}")
    c[2].metric("NASA PHM08 score", f"{mt['test_nasa_score']:.0f}", help="Lower is better; late predictions penalised more")
    c[3].metric("80% interval coverage", f"{mt['interval80_coverage']:.0%}")
    left, right = st.columns(2)
    with left:
        te = pd.DataFrame(M["test_eval"])
        te["true_c"] = te["true"].clip(upper=125)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 130], y=[0, 130], mode="lines", line=dict(color=MUTED, dash="dot", width=1),
                                 name="perfect", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=te["true_c"], y=te["pred"], mode="markers", name="test engines",
                                 marker=dict(color=BLUE, size=8, line=dict(color="#fcfcfb", width=2)),
                                 customdata=te["unit"],
                                 hovertemplate="Engine #%{customdata}<br>actual %{x:.0f}<br>predicted %{y:.0f}<extra></extra>"))
        fig.update_layout(title="Predicted vs actual RUL (held-out NASA test set)", hovermode="closest")
        fig.update_xaxes(title="Actual RUL (capped at 125)")
        fig.update_yaxes(title="Predicted RUL")
        st.plotly_chart(style(fig, 400, legend=False), width="stretch")
    with right:
        gi = pd.DataFrame(M["global_importance"]).iloc[::-1]
        fig = go.Figure(go.Bar(x=gi["mean_abs_shap"], y=gi["label"], orientation="h",
                               marker=dict(color=BLUE, cornerradius=4),
                               hovertemplate="%{y}<br>mean |SHAP| %{x:.1f} cycles<extra></extra>"))
        fig.update_layout(title="Global feature importance (mean |SHAP|, grouped by sensor)")
        st.plotly_chart(style(fig, 400, legend=False), width="stretch")
    st.caption(f"LightGBM on {mt['n_features']} causal time-series features from sensors "
               f"{', '.join(SENSOR_INFO[s][0] for s in mt['sensors_used'])}. "
               f"{mt['n_train_engines']} run-to-failure training engines, {mt['n_test_engines']} test engines.")
