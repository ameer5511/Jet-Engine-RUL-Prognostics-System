"""Jet Engine RUL Prognostics System — Page 1: Engine View.

Shows how each model tracks true RUL for one validation engine.
Read-only: reads results/phase2_val_predictions.csv only.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import (
    render_footer,
    render_sidebar,
    render_disclaimer_strip,
    render_top_bar,
    render_kpi_band,
)
from dashboard_data import (
    get_val_engines,
    get_prediction_models,
    get_engine_predictions,
    load_predictions,
)
from styles.plotly_template import apply_jerps
from styles.tokens import (
    SERIES_TRUE_RUL,
    SERIES_CONSTANT,
    SERIES_RANDOM_FOREST,
    SERIES_GRADIENT_BOOSTING,
    SERIES_SAFE_SIDE,
    SERIES_P10_P90_BAND,
    BORDER_HAIRLINE,
    BG_PANEL,
    ACCENT,
    TEXT_MUTED,
    TEXT_PRIMARY,
)

st.set_page_config(
    page_title="Jet Engine RUL Prognostics System — Engine View",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject global CSS (shared)
with open(str(Path(__file__).resolve().parent.parent / "app.py"), "r") as _f:
    pass  # CSS is injected via app.py's inject_global_css when run as entry
# For multipage, inject minimal CSS here too
from app import inject_global_css
inject_global_css()

render_sidebar()
render_disclaimer_strip()

# ---------- Data ----------
preds = load_predictions()
if preds.empty:
    st.error(
        "Missing results/phase2_val_predictions.csv. Generate artifacts:\n"
        "  conda run -n ml python train.py"
    )
    st.stop()

engines = get_val_engines()
models = get_prediction_models()

# ---------- Header: engine + cycle selection ----------
col_engine, col_cycle = st.columns([3, 1])
with col_engine:
    engine_id = st.selectbox("Engine ID", engines, index=0)
with col_cycle:
    pass  # cycle slider below after we know engine

eng_df = get_engine_predictions(engine_id)
if eng_df.empty:
    st.warning(f"No data for engine {engine_id}.")
    st.stop()

min_cycle = int(eng_df["cycle"].min())
max_cycle = int(eng_df["cycle"].max())

cycle = st.slider(
    "Cycle",
    min_value=min_cycle,
    max_value=max_cycle,
    value=max_cycle,
    help="Select a flight cycle to inspect the predicted remaining life.",
)

row = eng_df[eng_df["cycle"] == cycle]
if row.empty:
    st.warning(f"No data for cycle {cycle} on engine {engine_id}.")
    st.stop()

r = row.iloc[0]
true_rul = float(r["RUL"])
preds_at_cycle = {m: float(r[m]) for m in models if m in r.index and pd.notna(r[m])}

# Header chips: predicted RUL at cycle
    gb_pred = preds_at_cycle.get("gradient_boosting", preds_at_cycle.get(models[0], 0))
    health = ("Critical" if gb_pred < 30 else "Warning" if gb_pred < 60 else "Healthy")
    health_color = {"Critical": "#f26464", "Warning": "#f2a93b", "Healthy": "#6fd08a"}[health]

    st.markdown(
        textwrap.dedent(f"""
        <div style="display:flex;align-items:center;gap:16px;margin-bottom:8px;">
        <div style="font-family:'IBM Plex Sans',sans-serif;font-size:11px;font-weight:600;
        text-transform:uppercase;letter-spacing:0.08em;color:#a2a9b0;">
        ENGINE {engine_id}
        </div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:24px;font-weight:500;color:#f2f4f8;">
        {gb_pred:.0f} flights left
        </div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:13px;color:#6f7782;">
        True: {true_rul:.0f} &nbsp;·&nbsp; Δ: {(gb_pred-true_rul):+.0f}
        </div>
        <span style="font-family:'IBM Plex Mono',monospace;font-size:11px;font-weight:700;
        color:{health_color};border:1px solid {health_color};padding:2px 8px;border-radius:2px;">
        {health}
        </span>
        </div>
        """).strip(),
        unsafe_allow_html=True,
    )

# ---------- KPI band ----------
per_engine = {}
for m in models:
    y_t = eng_df["RUL"].to_numpy(dtype=float)
    y_p = eng_df[m].to_numpy(dtype=float)
    err = y_p - y_t
    ae = np.abs(err)
    se = err ** 2
    per_engine[m] = {
        "mae": float(np.mean(ae)),
        "rmse": float(np.sqrt(np.mean(se))),
        "bias": float(np.mean(err)),
        "worst_mae": float(np.max(ae)),
        "too_opt": float(np.mean(err > 10.0)),
    }

kpi_items = [
    {"label": "MAE", "value": f"{per_engine[models[0]]['mae']:.1f}", "subtext": "mean abs error"},
    {"label": "RMSE", "value": f"{per_engine[models[0]]['rmse']:.1f}", "subtext": "root mean sq error"},
    {"label": "Bias", "value": f"{per_engine[models[0]]['bias']:+.1f}", "subtext": "+ optimistic", "sign": "pos" if per_engine[models[0]]['bias'] > 0 else "neg"},
    {"label": "Worst MAE", "value": f"{per_engine[models[0]]['worst_mae']:.1f}", "subtext": "max abs error"},
    {"label": "Too-optimistic", "value": f"{per_engine[models[0]]['too_opt']*100:.0f}%", "subtext": "engines > +10"},
]
render_kpi_band(kpi_items)

# ---------- Main chart ----------
fig = go.Figure()

# True RUL dotted
fig.add_trace(
    go.Scatter(
        x=eng_df["cycle"],
        y=eng_df["RUL"],
        mode="lines",
        name="True RUL",
        line=dict(color=SERIES_TRUE_RUL, width=1.5, dash="dot"),
        hovertemplate="Cycle %{x}<br>True RUL %{y:.1f}<extra></extra>",
    )
)

# Model lines
model_colors = {
    "constant": SERIES_CONSTANT,
    "random_forest": SERIES_RANDOM_FOREST,
    "gradient_boosting": SERIES_GRADIENT_BOOSTING,
    "safe_side": SERIES_SAFE_SIDE,
}
for m in models:
    if m not in model_colors:
        continue
    fig.add_trace(
        go.Scatter(
            x=eng_df["cycle"],
            y=eng_df[m],
            mode="lines",
            name=m.replace("_", " ").title(),
            line=dict(color=model_colors.get(m, ACCENT), width=1.5),
            visible=True if m == "gradient_boosting" else "legendonly",
            hovertemplate=f"Cycle %{{x}}<br>{m.replace('_', ' ').title()} %{{y:.1f}}<extra></extra>",
        )
    )

# P10-P90 band
if "quantile_10" in models and "quantile_90" in models:
    fig.add_trace(
        go.Scatter(
            x=eng_df["cycle"].tolist() + eng_df["cycle"][::-1].tolist(),
            y=eng_df["quantile_90"].tolist() + eng_df["quantile_10"][::-1].tolist(),
            fill="toself",
            fillcolor=SERIES_P10_P90_BAND,
            line=dict(color="rgba(0,0,0,0)"),
            name="P10–P90",
            hoverinfo="skip",
            showlegend=True,
        )
    )

# Vertical marker at selected cycle
fig.add_vline(
    x=cycle,
    line=dict(color=ACCENT, width=1, dash="dash"),
    annotation_text=f"Cycle {cycle}",
    annotation_position="bottom right",
    annotation_font=dict(size=10, color=TEXT_MUTED),
)

apply_jerps(fig)
fig.update_layout(
    xaxis_title="Cycle",
    yaxis_title="RUL (flights left)",
    height=420,
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ---------- Table: last 10-20 cycles ----------
    st.markdown(
        textwrap.dedent(f"""
        <div class="panel">
        <div class="panel-header">
        <div class="panel-title">Last 20 cycles — Engine {engine_id}</div>
        </div>
        </div>
        """).strip(),
        unsafe_allow_html=True,
    )

table_df = eng_df.tail(20).copy()
display_cols = ["cycle", "RUL"] + models
table_df = table_df[display_cols]
table_df["cycle"] = table_df["cycle"].astype(int)
table_df = table_df.round(1)
table_df.columns = ["Cycle", "True RUL"] + [m.replace("_", " ").title() for m in models]

st.dataframe(table_df, use_container_width=True, hide_index=True, height=420)

render_footer()
