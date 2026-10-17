"""Past-only feature engineering for AeroGuard.

For each retained sensor we build three causal features per cycle:
- ``<sensor>_cur``   : the current reading.
- ``<col>_mean``     : mean over the last ``window`` cycles (available history only).
- ``{col}_trend``    : change across that window (current - first value in window).

Rules (see MASTER_SPEC.md, A4):
- Only past and current rows are used; future readings never leak in.
- The rolling window resets at each engine boundary.
- Partial windows at the start of an engine use the flights available so far;
  the trend is 0 when fewer than two flights are available.
- Sensors that never change are dropped (threshold configurable via ``atol``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import pandas as pd


@dataclass
class FeatureSet:
    """Result of :func:`add_features`."""

    frame: pd.DataFrame
    feature_cols: list
    used_sensors: list
    dropped_sensors: list


def constant_sensors(df, sensors, atol: float = 0.0):
    """Return sensors that never change in ``df``.

    A sensor is dropped if it has <= 1 unique value or its standard deviation is
    <= ``atol``. Default ``atol=0.0`` drops only strictly-constant sensors.
    """
    dropped = []
    for col in sensors:
        if col not in df.columns:
            raise KeyError(f"sensor column not found: {col}")
        if df[col].nunique(dropna=True) <= 1 or float(df[col].std(ddof=0)) <= atol:
            dropped.append(col)
    return dropped


def _validate_window(window: int) -> None:
    if not isinstance(window, int) or window < 1:
        raise ValueError(f"window must be a positive int, got {window!r}")


def add_features(
    df,
    sensors,
    window: int = 10,
    drop_constant: bool = True,
    constant: list | None = None,
    atol: float = 0.0,
) -> FeatureSet:
    """Build past-only features per sensor.

    Parameters
    ----------
    df : per-cycle rows with ``unit`` and ``cycle`` columns.
    sensors : sensor column names to featurise.
    window : rolling window length in cycles.
    drop_constant : whether to drop sensors that never change.
    constant : precomputed constant-sensor list (e.g. from the training split).
        If given, it is used as-is; otherwise it is computed on ``df``.
    atol : standard-deviation tolerance for constant-sensor detection.

    Returns
    -------
    FeatureSet with the augmented frame and bookkeeping lists.
    """
    _validate_window(window)
    frame = df.sort_values(["unit", "cycle"]).reset_index(drop=True).copy()
    for required in ("unit", "cycle"):
        if required not in frame.columns:
            raise KeyError(f"missing required column: {required}")

    if not drop_constant:
        dropped = []
    elif constant is not None:
        dropped = list(constant)
    else:
        dropped = constant_sensors(frame, sensors, atol=atol)

    dropped_set = set(dropped)
    used = [c for c in sensors if c not in dropped_set]

    grouped = frame.groupby("unit", sort=False)
    new_cols = {}
    feature_cols = []
    for col in used:
        series = frame[col].astype(float)
        mean = grouped[col].transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )
        trend = grouped[col].transform(
            lambda s: s.rolling(window, min_periods=1).apply(
                lambda w: float(w[-1] - w[0]), raw=True
            )
        )
        cur_name, mean_name, trend_name = f"{col}_cur", f"{col}_mean", f"{col}_trend"
        new_cols[cur_name] = series
        new_cols[mean_name] = mean.astype(float)
        new_cols[trend_name] = trend.astype(float)
        feature_cols += [cur_name, mean_name, trend_name]

    for name, values in new_cols.items():
        frame[name] = values

    return FeatureSet(
        frame=frame,
        feature_cols=feature_cols,
        used_sensors=used,
        dropped_sensors=dropped,
    )


def assert_no_nan(frame, feature_cols) -> None:
    """Raise if any feature column contains NaN."""
    if frame[feature_cols].isna().to_numpy().any():
        bad = [c for c in feature_cols if frame[c].isna().any()]
        raise ValueError(f"feature columns contain NaN: {bad}")
