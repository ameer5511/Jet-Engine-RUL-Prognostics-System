"""Jet Engine RUL Prognostics System — Page 2: Model Comparison.

Compares models overall, shows NASA score, cost of safety, and coverage.
Read-only: reads results/phase2_val_predictions.csv and results/phase2_metrics.json.
"""

from __future__ import annotations

import sys
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
    load_predictions,
    load_metrics,
    get_prediction_models,
    get_model_metrics_summary,
)
from styles.plotly_template import apply_jerps
from styles.tokens import (
    ACCENT,
    BG_PANEL,
    BORDER_HAIRLINE,
    NEGATIVE,
    POSITIVE,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    WARNING,
)

st.set_page_config(
    page_title="Jet Engine RUL Prognostics System — Model Comparison",
    layout="wide",
    initial_sidebar_state="expanded",
)

from app import inject_global_css
inject_global_css()

render_sidebar()
render_top_bar()
render_disclaimer_strip()

# ---------- Data ----------
preds = load_predictions()
if preds.empty:
    st.error("Missing results/phase2_val_predictions.csv. Run: conda run -n ml python train.py")
    st.stop()

summary_df = get_model_metrics_summary()
if summary_df.empty:
    st.warning("Could not build model summary from artifacts.")
    st.stop()

models = get_prediction_models()
metrics = load_metrics()

# ---------- KPI band: best model by MAE ----------
best_row = summary_df.loc[summary_df["engine_mean_mae"].idxmin()]
best_model = best_row["model"]
best_mae = float(best_row["engine_mean_mae"])

# NASA score per model
nasa_scores = {}
for m in models:
    y_true = preds["RUL"].to_numpy(dtype=float)
    y_pred = preds[m].to_numpy(dtype=float)
    err = y_true - y_pred
    s = np.where(err < 0, np.exp(-err / 13.0) - 1.0, np.exp(err / 10.0) - 1.0)
    nasa_scores[m] = float(np.sum(s))

# Cost of safety: safe_side MAE - GB MAE on val
ss_mae = float(np.mean(np.abs(preds["safe_side"] - preds["RUL"])))
gb_mae = float(np.mean(np.abs(preds["gradient_boosting"] - preds["RUL"])))
cost_of_safety = ss_mae - gb_mae

# Too-optimistic rate (pooled rows > +10)
best_too_opt = float(best_row["too_opt_engine_margin10"]) if pd.notna(best_row["too_opt_engine_margin10"]) else 0.0

kpi_items = [
    {"label": "Best model", "value": best_model, "subtext": "by engine-mean MAE"},
    {"label": "MAE", "value": f"{best_mae:.1f}", "subtext": f"{best_model}"},
    {"label": "NASA score", "value": f"{nasa_scores.get('gradient_boosting', 0):,.0f}", "subtext": "gradient boosting"},
    {"label": "Too-optimistic", "value": f"{best_too_opt*100:.0f}%", "subtext": "engines bias > +10"},
    {"label": "Cost of safety", "value": f"{cost_of_safety:+.1f}", "subtext": "safe − GB MAE", "sign": "pos" if cost_of_safety > 0 else "neg"},
]
render_kpi_band(kpi_items)

# ---------- Model scores table ----------
st.markdown(
    f"""
    <div class="panel">
      <div class="panel-header">
        <div class="panel-title">Model scores — val engines (uncapped RUL)</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

display_df = summary_df.copy()
for col in ["engine_mean_mae", "engine_mean_rmse", "engine_mean_bias", "engine_worst_mae",
            "pooled_mae", "pooled_rmse"]:
    display_df[col] = display_df[col].apply(lambda v: f"{v:.1f}" if pd.notna(v) else "—")
display_df["too_opt_engine_margin10"] = display_df["too_opt_engine_margin10"].apply(
    lambda v: f"{v*100:.0f}%" if pd.notna(v) else "—"
)
display_df["nasa_score"] = display_df["nasa_score"].apply(lambda v: f"{v:,.0f}" if pd.notna(v) else "—")

st.dataframe(
    display_df[["model", "engine_mean_mae", "engine_mean_rmse", "engine_mean_bias",
                "engine_worst_mae", "too_opt_engine_margin10", "nasa_score"]],
    use_container_width=True,
    hide_index=True,
    column_config={
        "model": "Model",
        "engine_mean_mae": "Engine-mean MAE",
        "engine_mean_rmse": "Engine-mean RMSE",
        "engine_mean_bias": "Engine-mean bias",
        "engine_worst_mae": "Worst MAE",
        "too_opt_engine_margin10": "Too-opt @10",
        "nasa_score": "NASA score",
    },
)

# ---------- Two columns: too-optimistic panel + coverage card ----------
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(
        f"""
        <div class="panel">
          <div class="panel-header">
            <div class="panel-title">Too-optimistic engines (margin = +10 flights)</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # Horizontal bars: fraction of engines with bias > 10 per model
    too_opt_data = []
    for _, row in summary_df.iterrows():
        too_opt_data.append((row["model"], float(row["too_opt_engine_margin10"]) if pd.notna(row["too_opt_engine_margin10"]) else 0.0))
    too_opt_df = pd.DataFrame(too_opt_data, columns=["model", "rate"])
    too_opt_df = too_opt_df.sort_values("rate")

    fig_bar = go.Figure()
    fig_bar.add_trace(
        go.Bar(
            x=too_opt_df["rate"],
            y=too_opt_df["model"],
            orientation="h",
            marker=dict(color=[NEGATIVE if r > 0.3 else POSITIVE for r in too_opt_df["rate"]]),
            hovertemplate="%{y}: %{x:.0%}<extra></extra>",
        )
    )
    apply_jerps(fig_bar)
    fig_bar.update_layout(
        xaxis=dict(tickformat=".0%", title="Fraction of engines"),
        yaxis=dict(title=""),
        height=260,
        showlegend=False,
        margin=dict(l=8, r=8, t=8, b=8),
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    st.caption("Red = more engines predicted too high (dangerous). Green = safer side.")

with col2:
    st.markdown(
        f"""
        <div class="panel">
          <div class="panel-header">
            <div class="panel-title">P10–P90 coverage vs nominal</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # Compute coverage from quantile predictions
    q10 = preds["quantile_10"].to_numpy(dtype=float)
    q90 = preds["quantile_90"].to_numpy(dtype=float)
    y_true = preds["RUL"].to_numpy(dtype=float)
    coverage = float(np.mean((y_true >= q10) & (y_true <= q90)))
    nominal = 0.80

    st.markdown(
        f"""
        <div style="font-family:'IBM Plex Mono',monospace;font-size:28px;font-weight:500;color:#f2f4f8;">
          {coverage*100:.1f}%
        </div>
        <div style="font-size:11px;color:#6f7782;margin-top:4px;">
          observed coverage · nominal {nominal*100:.0f}%
        </div>
        """,
        unsafe_allow_html=True,
    )
    if coverage < nominal:
        st.markdown(
            f'<div class="chip chip-warning">Below nominal 0.80</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="chip chip-info">Within nominal</div>',
            unsafe_allow_html=True,
        )
    st.caption(
        "Known limitation: P90 tends to over-predict (bias +40.42 on val). "
        "Coverage 0.746 vs nominal 0.80."
    )

# ---------- Metrics explained ----------
st.markdown(
    f"""
    <div class="panel">
      <div class="panel-header">
        <div class="panel-title">Metrics explained</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

explanations = [
    ("MAE", "Mean absolute error — average |prediction − truth| across cycles. Lower is better."),
    ("RMSE", "Root mean squared error — penalises large errors more than MAE."),
    ("Bias", "Mean signed error (prediction − truth). Positive = over-predict / optimistic / dangerous."),
    ("NASA score", "Asymmetric penalty per row: exp(−d/13)−1 if prediction too high; exp(d/10)−1 if too low. Lower is better."),
    ("Too-optimistic @10", "Fraction of engines whose mean bias exceeds +10 flights (predicts at least 10 flights too far)."),
    ("Cost of safety", "safe-side MAE − gradient-boosting MAE. Positive = cautious model pays extra error."),
]
for name, desc in explanations:
    st.markdown(
        f"""
        <div class="metric-row">
          <div class="metric-name mono">{name}</div>
          <div class="metric-explain">{desc}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

render_footer()
