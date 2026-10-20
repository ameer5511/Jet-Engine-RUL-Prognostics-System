"""Evaluation metrics for AeroGuard RUL predictions.

Sign convention
---------------
``d = prediction - truth``.
For RUL, a *positive* d means we predicted **too many flights left**
(optimistic / dangerous). A negative d is conservative.

NASA C-MAPSS asymmetric score (Saxena & Goebel, PHM 2008):
    d < 0  (under-prediction, safe):   s = exp(-d/13) - 1   (less steep, rate 1/13)
    d >= 0 (over-prediction, dangerous): s = exp(d/10) - 1   (steeper, rate 1/10)

Aggregation (see MASTER_SPEC.md A6)
-----------------------------------
Primary metrics are per-engine: each engine is reduced to one value, then we
average over engines so a long test does not dominate a short one. A secondary
final-cycle view uses each engine's last recorded cycle. The NASA score is
computed at each engine's last cycle only (not pooled over all rows).
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def _as_float(values) -> np.ndarray:
    return np.asarray(values, dtype=float)


def mae(y_true, y_pred) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(_as_float(y_pred) - _as_float(y_true))))


def rmse(y_true, y_pred) -> float:
    """Root mean squared error."""
    diff = _as_float(y_pred) - _as_float(y_true)
    return float(np.sqrt(np.mean(diff ** 2)))


def per_engine_metrics(
    frame: pd.DataFrame,
    engine_col: str = "unit",
    true_col: str = "RUL",
    pred_col: str = "pred",
) -> pd.DataFrame:
    """Reduce each engine to bias / MAE / RMSE / max overestimate.

    Returns a frame with columns: engine, n, bias, mae, rmse, max_over.
    ``bias`` is the mean signed error (prediction - truth); positive is
    optimistic.
    """
    for col in (engine_col, true_col, pred_col):
        if col not in frame.columns:
            raise KeyError(f"missing column: {col}")
    errors = _as_float(frame[pred_col]) - _as_float(frame[true_col])
    work = pd.DataFrame(
        {
            "engine": frame[engine_col].to_numpy(),
            "err": errors,
            "ae": np.abs(errors),
            "se": errors ** 2,
        }
    )
    grouped = work.groupby("engine")
    out = pd.DataFrame(
        {
            "n": grouped.size(),
            "bias": grouped["err"].mean(),
            "mae": grouped["ae"].mean(),
            "rmse": np.sqrt(grouped["se"].mean()),
            "max_over": grouped["err"].max(),
        }
    )
    return out.reset_index()


def summarise_engine_metrics(engine_metrics: pd.DataFrame) -> dict:
    """Average the per-engine metrics across engines."""
    return {
        "n_engines": int(len(engine_metrics)),
        "engine_mean_mae": float(engine_metrics["mae"].mean()),
        "engine_mean_rmse": float(engine_metrics["rmse"].mean()),
        "engine_mean_bias": float(engine_metrics["bias"].mean()),
        "engine_worst_mae": float(engine_metrics["mae"].max()),
    }


def too_optimistic_rate(engine_metrics: pd.DataFrame, margin: float = 10.0) -> float:
    """Fraction of engines whose mean signed error exceeds ``margin`` flights."""
    return float((engine_metrics["bias"] > margin).mean())


def row_too_optimistic_rate(
    frame: pd.DataFrame, true_col: str = "RUL", pred_col: str = "pred", margin: float = 10.0
) -> float:
    """Fraction of rows whose signed error exceeds ``margin`` flights."""
    err = _as_float(frame[pred_col]) - _as_float(frame[true_col])
    return float(np.mean(err > margin))


def interval_coverage(y_true, low, high) -> float:
    """Fraction of truths within [low, high] (inclusive)."""
    y = _as_float(y_true)
    lo = _as_float(low)
    hi = _as_float(high)
    return float(np.mean((y >= lo) & (y <= hi)))


def nasa_score(d: np.ndarray) -> float:
    """NASA C-MAPSS asymmetric score on an array of errors d = pred − truth.

    Per-engine last-cycle errors only (caller slices to last cycle).
        d < 0  (under-prediction, safe):   s = exp(-d/13) - 1   (less steep)
        d >= 0 (over-prediction, dangerous): s = exp(d/10) - 1   (steeper)

    Returns sum. Lower is better.
    """
    d = _as_float(d)
    s = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(s))


def nasa_score_at_last_cycle(
    frame: pd.DataFrame,
    engine_col: str = "unit",
    cycle_col: str = "cycle",
    true_col: str = "RUL",
    pred_col: str = "pred",
) -> float:
    """NASA score summed over each engine's LAST recorded cycle only.

    Not pooled over all rows — one error per engine at the final cycle.
    """
    last = frame.groupby(engine_col)[cycle_col].transform("max")
    last_cycle = frame.loc[frame[cycle_col] == last].copy()
    d = _as_float(last_cycle[pred_col]) - _as_float(last_cycle[true_col])
    return nasa_score(d)


def nasa_score_per_engine(
    frame: pd.DataFrame,
    engine_col: str = "unit",
    cycle_col: str = "cycle",
    true_col: str = "RUL",
    pred_col: str = "pred",
) -> pd.DataFrame:
    """NASA score summed per engine at last cycle.

    Returns columns: engine, nasa_score.
    """
    for col in (engine_col, cycle_col, true_col, pred_col):
        if col not in frame.columns:
            raise KeyError(f"missing column: {col}")
    last = frame.groupby(engine_col)[cycle_col].transform("max")
    last_cycle = frame.loc[frame[cycle_col] == last].copy()
    d = _as_float(last_cycle[pred_col]) - _as_float(last_cycle[true_col])
    s = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    work = pd.DataFrame({"engine": last_cycle[engine_col].to_numpy(), "s": s})
    grouped = work.groupby("engine")
    out = pd.DataFrame({"nasa_score": grouped["s"].sum()})
    return out.reset_index()


def final_cycle_frame(frame: pd.DataFrame, cycle_col: str = "cycle", engine_col: str = "unit") -> pd.DataFrame:
    """Return each engine's last recorded cycle."""
    last = frame.groupby(engine_col)[cycle_col].transform("max")
    return frame.loc[frame[cycle_col] == last].copy()
