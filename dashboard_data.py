"""Read-only artifact loader for Jet Engine RUL Prognostics System.

The dashboard reads pre-computed artifacts only. It never trains, calls Nemotron,
or writes anything. Every number traces to a JSON / CSV file under results/.
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import streamlit as st

from styles.tokens import DISCLAIMER_TEXT

RESULTS_DIR = Path("results")

# Artifact filenames
PRED_CSV = RESULTS_DIR / "phase2_val_predictions.csv"
METRICS_JSON = RESULTS_DIR / "phase2_metrics.json"
CONFIG_JSON = RESULTS_DIR / "phase2_resolved_config.json"
STRESS_JSON = RESULTS_DIR / "phase4_stress.json"
STRESS_CSV = RESULTS_DIR / "phase4_stress.csv"
NEMOTRON_CACHE_JSON = RESULTS_DIR / "nemotron_scenarios.json"

# Disclaimer (shown on every page)
DISCLAIMER = DISCLAIMER_TEXT


def _ensure_results_dir() -> None:
    if not RESULTS_DIR.exists():
        raise FileNotFoundError(
            f"results/ directory not found at {RESULTS_DIR.resolve()}. "
            "Generate artifacts first:\n"
            "  conda run -n ml python train.py\n"
            "  conda run -n ml python stress_test.py"
        )


@st.cache_data
def load_predictions() -> pd.DataFrame:
    """Load per-cycle validation predictions: unit, cycle, RUL, <model> columns."""
    _ensure_results_dir()
    if not PRED_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(PRED_CSV)


@st.cache_data
def load_metrics() -> dict[str, Any]:
    """Load model metrics from phase2_metrics.json."""
    _ensure_results_dir()
    if not METRICS_JSON.exists():
        return {}
    with METRICS_JSON.open("r", encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data
def load_resolved_config() -> dict[str, Any]:
    """Load resolved config from phase2_resolved_config.json."""
    _ensure_results_dir()
    if not CONFIG_JSON.exists():
        return {}
    with CONFIG_JSON.open("r", encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data
def load_stress_json() -> dict[str, Any]:
    """Load aggregated stress-test results from phase4_stress.json."""
    _ensure_results_dir()
    if not STRESS_JSON.exists():
        return {}
    with STRESS_JSON.open("r", encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data
def load_stress_csv() -> pd.DataFrame:
    """Load flat stress-test table from phase4_stress.csv."""
    _ensure_results_dir()
    if not STRESS_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(STRESS_CSV)


@st.cache_data
def load_nemotron_settings() -> list[dict[str, Any]]:
    """Load Nemotron scenario cache if present; returns [] when missing."""
    if not NEMOTRON_CACHE_JSON.exists():
        return []
    try:
        with NEMOTRON_CACHE_JSON.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data.get("settings", [])
    except (OSError, ValueError):
        return []


@st.cache_resource
def load_hero_base64(image_path: str | Path = "docs/ui/assets/Hero-image.png") -> str | None:
    """Encode hero image as base64 data URI for embedding. Returns None if missing."""
    path = Path(image_path)
    if not path.exists():
        return None
    data = path.read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{encoded}"


# ---------------------------------------------------------------------------
# Accessors used by pages
# ---------------------------------------------------------------------------

@st.cache_data
def get_val_engines() -> list[int]:
    """Return sorted list of validation engine IDs."""
    df = load_predictions()
    if df.empty:
        return []
    return sorted(df["unit"].astype(int).unique().tolist())


@st.cache_data
def get_prediction_models() -> list[str]:
    """Return list of model column names in the predictions CSV (excluding base cols)."""
    df = load_predictions()
    if df.empty:
        return []
    base_cols = {"unit", "cycle", "RUL"}
    return [c for c in df.columns if c not in base_cols]


@st.cache_data
def get_engine_predictions(engine_id: int) -> pd.DataFrame:
    """Return all cycles for one validation engine, sorted by cycle."""
    df = load_predictions()
    if df.empty:
        return pd.DataFrame()
    eng = df[df["unit"] == engine_id].sort_values("cycle").reset_index(drop=True)
    return eng


@st.cache_data
def get_engine_per_engine_metrics(engine_id: int) -> dict[str, float] | None:
    """Compute per-engine metrics from the predictions CSV.

    Returns dict with engine_mean_mae, engine_mean_rmse, engine_mean_bias,
    engine_worst_mae, too_optimistic_engine_margin10, pooled_mae, pooled_rmse.
    """
    df = load_predictions()
    if df.empty:
        return None
    eng = df[df["unit"] == engine_id]
    if eng.empty:
        return None
    import numpy as np

    y_true = eng["RUL"].to_numpy(dtype=float)
    out: dict[str, float] = {}
    for model_col in get_prediction_models():
        y_pred = eng[model_col].to_numpy(dtype=float)
        err = y_pred - y_true
        ae = np.abs(err)
        se = err ** 2
        out[model_col] = {
            "mae": float(np.mean(ae)),
            "rmse": float(np.sqrt(np.mean(se))),
            "bias": float(np.mean(err)),
            "worst_mae": float(np.max(ae)),
            "too_opt": float(np.mean(err > 10.0)),
        }
    return out


@st.cache_data
def get_model_metrics_summary() -> pd.DataFrame:
    """Build a tidy model comparison table from phase2_metrics.json.

    Columns: model, engine_mean_mae, engine_mean_rmse, engine_mean_bias,
    engine_worst_mae, too_opt_engine_margin10, pooled_mae, pooled_rmse,
    nasa_score (computed from predictions if available).
    """
    metrics = load_metrics()
    df_preds = load_predictions()
    if not metrics or df_preds.empty:
        return pd.DataFrame()
    import numpy as np

    per_model = metrics.get("per_model", {})
    y_true = df_preds["RUL"].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    for model_col in get_prediction_models():
        agg = per_model.get(model_col, {}).get("aggregate", {})
        def _mean(key: str) -> float | None:
            v = agg.get(key)
            if isinstance(v, dict):
                return v.get("mean")
            return v
        y_pred = df_preds[model_col].to_numpy(dtype=float)
        # NASA score at last cycle per engine (d = pred − truth)
        d = y_pred - y_true
        s = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
        nasa = float(np.sum(s))
        rows.append(
            {
                "model": model_col.replace("_", " ").title(),
                "engine_mean_mae": _mean("engine_mean_mae"),
                "engine_mean_rmse": _mean("engine_mean_rmse"),
                "engine_mean_bias": _mean("engine_mean_bias"),
                "engine_worst_mae": _mean("engine_worst_mae"),
                "too_opt_engine_margin10": _mean("too_opt_engine_margin10"),
                "pooled_mae": _mean("pooled_mae"),
                "pooled_rmse": _mean("pooled_rmse"),
                "nasa_score": nasa,
            }
        )
    return pd.DataFrame(rows)


@st.cache_data
def get_stress_summary() -> dict[str, Any]:
    """Return the aggregate stress-test summary dict."""
    return load_stress_json()


@st.cache_data
def get_stress_table() -> pd.DataFrame:
    """Return the flat stress-test CSV table."""
    return load_stress_csv()


@st.cache_data
def get_nemotron_count() -> int:
    """Number of Nemotron-proposed scenarios (0 if cache missing)."""
    return len(load_nemotron_settings())


def artifact_status() -> dict[str, bool]:
    """Return presence status for each artifact, used by the sidebar."""
    return {
        "predictions_csv": PRED_CSV.exists(),
        "metrics_json": METRICS_JSON.exists(),
        "config_json": CONFIG_JSON.exists(),
        "stress_json": STRESS_JSON.exists(),
        "stress_csv": STRESS_CSV.exists(),
        "nemotron_cache": NEMOTRON_CACHE_JSON.exists(),
    }
