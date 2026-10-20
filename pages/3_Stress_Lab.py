"""Jet Engine RUL Prognostics System — Page 3: Stress Lab.

Shows how predictions degrade under sensor faults and what safety costs.
Read-only: reads results/phase4_stress.json and results/phase4_stress.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
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
    load_stress_json,
    load_stress_csv,
    load_nemotron_settings,
    get_nemotron_count,
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
    page_title="Jet Engine RUL Prognostics System — Stress Lab",
    layout="wide",
    initial_sidebar_state="expanded",
)

from app import inject_global_css
inject_global_css()

render_sidebar()
render_top_bar()
render_disclaimer_strip()

# ---------- Data ----------
stress_json = load_stress_json()
stress_csv = load_stress_csv()

if stress_json.get("meta") is None and stress_csv.empty:
    st.error(
        "Missing results/phase4_stress.json or results/phase4_stress.csv. "
        "Generate artifacts:\n"
        "  conda run -n ml python stress_test.py"
    )
    st.stop()

meta = stress_json.get("meta", {})
aggregate = stress_json.get("aggregate", {})

n_scenarios = meta.get("n_scenarios", 0)
n_seeds = len(meta.get("seeds", []))
n_features = meta.get("n_features", 0)
n_val_engines = meta.get("n_val_engines", 0)

# Nemotron provenance
nemotron_count = get_nemotron_count()
grid_count = n_scenarios - 1  # minus clean
provenance_text = f"grid ({grid_count})"
if nemotron_count > 0:
    provenance_text += f" + nemotron ({nemotron_count})"
else:
    provenance_text += " · grid only"

# ---------- KPI band ----------
# Worst fault: max degradation across all models/scenarios from clean baseline
clean_mae = {}
if "clean" in aggregate:
    for model, vals in aggregate["clean"].items():
        clean_mae[model] = float(vals["engine_mean_mae"]["mean"])

worst_increase = 0.0
worst_scenario = "—"
for sname, svals in aggregate.items():
    if sname == "clean":
        continue
    for model, mvals in svals.items():
        inc = float(mvals["engine_mean_mae"]["mean"]) - clean_mae.get(model, 0)
        if inc > worst_increase:
            worst_increase = inc
            worst_scenario = sname

# Cost of safety at worst scenario
if worst_scenario != "—" and worst_scenario in aggregate:
    ss = float(aggregate[worst_scenario]["safe_side"]["engine_mean_mae"]["mean"])
    gb = float(aggregate[worst_scenario]["gradient_boosting"]["engine_mean_mae"]["mean"])
    cost_at_worst = ss - gb
else:
    cost_at_worst = 0.0

kpi_items = [
    {"label": "Scenarios", "value": str(n_scenarios), "subtext": provenance_text},
    {"label": "Seeds", "value": str(n_seeds), "subtext": "mean ± std reported"},
    {"label": "Worst fault", "value": worst_scenario.replace("_", " ").title(), "subtext": "largest MAE increase"},
    {"label": "Max MAE increase", "value": f"+{worst_increase:.1f}", "subtext": "vs clean baseline"},
    {"label": "Cost of safety", "value": f"{cost_at_worst:+.1f}", "subtext": "safe − GB at worst fault", "sign": "pos" if cost_at_worst > 0 else "neg"},
]
render_kpi_band(kpi_items)

# ---------- Provenance chips ----------
st.markdown(
    f"""
    <div class="panel">
      <div class="panel-header">
        <div class="panel-title">Fault provenance</div>
      </div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#a2a9b0;">
        {provenance_text} &nbsp;·&nbsp; Nemotron proposals are single-sensor extras, range-validated before use.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Controls ----------
st.markdown(
    f"""
    <div class="panel">
      <div class="panel-header">
        <div class="panel-title">Controls</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

col_kind, col_models = st.columns([1, 3])
with col_kind:
    fault_kind = st.segmented_control(
        "Fault kind",
        options=["Noise", "Drift", "Missing readings"],
        default="Noise",
        label_visibility="collapsed",
    )
with col_models:
    all_models = ["constant", "random_forest", "gradient_boosting", "safe_side"]
    selected_models = st.multiselect(
        "Models",
        options=all_models,
        default=["random_forest", "gradient_boosting", "safe_side"],
        format_func=lambda x: x.replace("_", " ").title(),
        label_visibility="collapsed",
    )

if not selected_models:
    st.info("Select at least one model.")
    st.stop()

# ---------- Chart A: Error increase vs fault size ----------
st.markdown(
    f"""
    <div class="panel">
      <div class="panel-header">
        <div class="panel-title">Chart A — Error increase vs fault size ({fault_kind})</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Build data for Chart A from aggregate
kind_map = {"Noise": "noise", "Drift": "drift", "Missing readings": "missing"}
kind_key = kind_map[fault_kind]

chart_data = []
for sname, svals in aggregate.items():
    if sname == "clean":
        continue
    # Determine kind and magnitude from scenario name
    parts = sname.split("_")
    if len(parts) >= 2:
        skind = parts[0]
        smag = parts[1].replace("k", "")
    else:
        continue
    if skind != kind_key:
        continue
    for model in selected_models:
        if model not in svals:
            continue
        inc = float(svals[model]["engine_mean_mae"]["mean"]) - clean_mae.get(model, 0)
        chart_data.append(
            {
                "scenario": sname,
                "kind": skind,
                "magnitude": float(smag),
                "model": model,
                "delta_mae": inc,
            }
        )

chart_df = pd.DataFrame(chart_data)
if not chart_df.empty:
    fig_a = go.Figure()
    model_colors = {
        "constant": "#6f7782",
        "random_forest": "#78a9ff",
        "gradient_boosting": "#4589ff",
        "safe_side": "#08bdba",
    }
    for model in selected_models:
        mdf = chart_df[chart_df["model"] == model].sort_values("magnitude")
        if mdf.empty:
            continue
        fig_a.add_trace(
            go.Scatter(
                x=mdf["magnitude"],
                y=mdf["delta_mae"],
                mode="lines+markers",
                name=model.replace("_", " ").title(),
                line=dict(color=model_colors.get(model, ACCENT), width=1.5),
                marker=dict(size=6),
                hovertemplate="%{x}<br>ΔMAE %{y:+.1f}<extra></extra>",
            )
        )
    fig_a.add_hline(y=0, line=dict(color=BORDER_HAIRLINE, width=1))
    apply_jerps(fig_a)
    fig_a.update_layout(
        xaxis_title="Fault magnitude",
        yaxis_title="Δ engine-mean MAE vs clean",
        height=300,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig_a, use_container_width=True)
else:
    st.info(f"No stress data for fault kind: {fault_kind}")

# ---------- Chart B: Cost of safety ----------
st.markdown(
    f"""
    <div class="panel">
      <div class="panel-header">
        <div class="panel-title">Chart B — Cost of safety ({fault_kind})</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

cost_data = []
for sname, svals in aggregate.items():
    if sname == "clean":
        continue
    parts = sname.split("_")
    if len(parts) >= 2:
        skind = parts[0]
        smag = parts[1].replace("k", "")
    else:
        continue
    if skind != kind_key:
        continue
    ss = float(svals["safe_side"]["engine_mean_mae"]["mean"])
    gb = float(svals["gradient_boosting"]["engine_mean_mae"]["mean"])
    cost_data.append(
        {
            "scenario": sname,
            "magnitude": float(smag),
            "cost": ss - gb,
        }
    )

cost_df = pd.DataFrame(cost_data)
if not cost_df.empty:
    cost_df = cost_df.sort_values("magnitude")
    fig_b = go.Figure()
    fig_b.add_trace(
        go.Bar(
            x=cost_df["magnitude"],
            y=cost_df["cost"],
            marker=dict(color=[NEGATIVE if c < 0 else POSITIVE for c in cost_df["cost"]]),
            hovertemplate="Magnitude %{x}<br>Cost %{y:+.1f}<extra></extra>",
        )
    )
    fig_b.add_hline(y=0, line=dict(color=BORDER_HAIRLINE, width=1))
    apply_jerps(fig_b)
    fig_b.update_layout(
        xaxis_title="Fault magnitude",
        yaxis_title="safe-side MAE − GB MAE",
        height=300,
        showlegend=False,
    )
    st.plotly_chart(fig_b, use_container_width=True)
    st.caption("Positive = safe side pays extra error. Negative = safe side wins.")

# ---------- Heatmap: fault grid ----------
st.markdown(
    f"""
    <div class="panel">
      <div class="panel-header">
        <div class="panel-title">Heatmap — MAE increase by fault kind × magnitude</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Build heatmap from aggregate: rows = kind, columns = magnitude, value = max GB delta
heatmap_rows = []
for sname, svals in aggregate.items():
    if sname == "clean":
        continue
    parts = sname.split("_")
    if len(parts) >= 2:
        skind = parts[0]
        smag = parts[1].replace("k", "")
    else:
        continue
    gb_delta = float(svals["gradient_boosting"]["engine_mean_mae"]["mean"]) - clean_mae.get("gradient_boosting", 0)
    heatmap_rows.append({"kind": skind, "magnitude": float(smag), "delta": gb_delta})

heatmap_df = pd.DataFrame(heatmap_rows)
if not heatmap_df.empty:
    pivot = heatmap_df.pivot(index="kind", columns="magnitude", values="delta")
    # Order rows and columns
    kind_order = ["noise", "drift", "missing"]
    mag_order = sorted(pivot.columns.tolist())
    pivot = pivot.reindex(index=[k for k in kind_order if k in pivot.index], columns=mag_order)

    fig_hm = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[f"k{m:g}" for m in pivot.columns],
            y=[k.title() for k in pivot.index],
            colorscale=[
                "#0f1b2d",
                "#1d3557",
                "#2a5db0",
                "#4589ff",
                "#a6c8ff",
            ],
            colorbar=dict(title="Δ MAE", titleside="right", tickfont=dict(size=9)),
            hovertemplate="Kind %{y}<br>Magnitude %{x}<br>ΔMAE %{z:+.1f}<extra></extra>",
            showscale=True,
        )
    )
    apply_jerps(fig_hm)
    fig_hm.update_layout(height=200, margin=dict(l=8, r=8, t=8, b=8))
    # Annotate cells with values
    for i, kind in enumerate(pivot.index):
        for j, mag in enumerate(pivot.columns):
            val = pivot.loc[kind, mag]
            if pd.notna(val):
                fig_hm.add_annotation(
                    x=j, y=i,
                    text=f"{val:+.1f}",
                    showarrow=False,
                    font=dict(size=9, color="white", family="IBM Plex Mono"),
                )
    st.plotly_chart(fig_hm, use_container_width=True)
else:
    st.info("No heatmap data available.")

# ---------- Stress results table ----------
st.markdown(
    f"""
    <div class="panel">
      <div class="panel-header">
        <div class="panel-title">Stress results — scenario × model × seed (mean ± std)</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not stress_csv.empty:
    # Filter by selected kind and models
    table_df = stress_csv.copy()
    table_df["kind"] = table_df["kind"].fillna("none")
    table_df = table_df[table_df["kind"] == kind_key]
    table_df = table_df[table_df["model"].isin(selected_models)]
    
    # Show aggregate: mean ± std per scenario × model
    agg_rows = []
    for (scenario, model), grp in table_df.groupby(["scenario", "model"]):
        mag = grp["magnitude"].iloc[0]
        src = grp["source"].iloc[0]
        mae_mean = grp["engine_mean_mae"].mean()
        mae_std = grp["engine_mean_mae"].std()
        agg_rows.append(
            {
                "Scenario": scenario.replace("_", " ").title(),
                "Kind": kind_key.title(),
                "Magnitude": f"{mag:g}",
                "Source": src.title(),
                "Model": model.replace("_", " ").title(),
                "Δ MAE": f"{mae_mean - clean_mae.get(model, 0):+.1f} ± {mae_std:.1f}",
            }
        )
    agg_df = pd.DataFrame(agg_rows)
    if not agg_df.empty:
        st.dataframe(agg_df, use_container_width=True, hide_index=True)
    else:
        st.info("No results for the selected filters.")
else:
    st.info("No stress CSV data available.")

render_footer()
