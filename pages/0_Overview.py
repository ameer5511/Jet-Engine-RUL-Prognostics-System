"""Jet Engine RUL Prognostics System — Page 0: Overview / Landing page.

Read-only: reads phase2_metrics.json for headline result card.
Never trains models, never calls Nemotron.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from app import (
    render_footer,
    render_sidebar,
    render_disclaimer_strip,
    render_hero_banner,
    inject_global_css,
)
from dashboard_data import load_metrics, load_predictions
from styles.tokens import ACCENT, BG_PANEL, BORDER_HAIRLINE, TEXT_MUTED, TEXT_PRIMARY, WARNING

st.set_page_config(
    page_title="Jet Engine RUL Prognostics System — Overview",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_global_css()
render_sidebar()
render_hero_banner()
render_disclaimer_strip()

# ---------- Headline result card (from real metrics, not hardcoded) ----------
metrics = load_metrics()
preds = load_predictions()

headline = "—"
cost_of_safety = "—"
too_opt_gb = "—"
too_opt_ss = "—"
if metrics and not preds.empty:
    per_model = metrics.get("per_model", {})
    gb_agg = per_model.get("gradient_boosting", {}).get("aggregate", {})
    ss_agg = per_model.get("safe_side", {}).get("aggregate", {})

    def _mean(d, key):
        v = d.get(key)
        if isinstance(v, dict):
            return v.get("mean")
        return v

    gb_mae = _mean(gb_agg, "engine_mean_mae")
    ss_mae = _mean(ss_agg, "engine_mean_mae")
    gb_too = _mean(gb_agg, "too_opt_engine_margin10")
    ss_too = _mean(ss_agg, "too_opt_engine_margin10")

    if gb_mae is not None and ss_mae is not None:
        cost_of_safety = ss_mae - gb_mae
    if gb_too is not None:
        too_opt_gb = gb_too * 100
    if ss_too is not None:
        too_opt_ss = ss_too * 100

st.markdown(
    f"""
    <div class="panel">
      <div class="panel-header">
        <div class="panel-title">Headline result — val engines (uncapped RUL)</div>
      </div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:15px;color:#f2f4f8;line-height:1.5;">
        The safe-side model costs <span style="color:#{'08bdba' if (cost_of_safety if isinstance(cost_of_safety,float) else 0) < 0 else 'f26464'}">{cost_of_safety if isinstance(cost_of_safety, float) else '—':+.1f}</span> flights of MAE vs Gradient Boosting
        but cuts too-optimistic engines from {too_opt_gb:.0f}% to {too_opt_ss:.0f}%.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Page links to the three core pages ----------
st.markdown(
    f"""
    <div class="panel">
      <div class="panel-header">
        <div class="panel-title">Go to</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)
with col1:
    st.page_link(
        "1_Engine_View.py",
        label="Engine View",
        help="Inspect RUL predictions for one validation engine, cycle by cycle.",
    )
with col2:
    st.page_link(
        "2_Model_Comparison.py",
        label="Model Comparison",
        help="Compare Constant, Random Forest, Gradient Boosting, Safe Side and quantile models using NASA score.",
    )
with col3:
    st.page_link(
        "3_Stress_Lab.py",
        label="Stress Lab",
        help="Stress-test predictions under noise, drift and missing-reading faults.",
    )

# ---------- How it works ----------
st.markdown(
    f"""
    <div class="panel">
      <div class="panel-header">
        <div class="panel-title">How it works</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

how_it_works = [
    ("Dataset", "NASA C-MAPSS FD001 — simulated turbofan engine run-to-failure data. 80 training engines, 20 validation engines. Uncapped RUL targets used for training and metrics."),
    ("Models", "Six models: Constant, Random Forest, Gradient Boosting (point predictors), Safe-side quantile GB (alpha=0.30), P10 and P90 quantile models. No neural nets."),
    ("Score", "NASA C-MAPSS asymmetric scoring (Saxena & Goebel PHM 2008): exp(−d/13)−1 if prediction too high (dangerous); exp(d/10)−1 if too low. Lower is better."),
    ("Stress Lab", "Controlled sensor faults — noise, drift, missing readings — applied across magnitudes with 5 seeds each. Measures ΔMAE vs clean baseline and cost of safety."),
    ("Nemotron", "NVIDIA Nemotron-3.5-Lightning used as an optional fault-scenario generator; proposals are range-validated before use. Nemotron scenarios cached separately."),
    ("Dashboard", "Read-only from precomputed artifacts under results/. Dark theme, IBM Plex fonts, 2D charts only. Always shows the simulation disclaimer."),
]

for name, desc in how_it_works:
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
