"""Tests for evaluate.py: metric definitions, signs, and aggregation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import evaluate as ev


def test_mae_known():
    # errors 1, 0, -2 -> mean abs error 1.0
    assert ev.mae([1, 2, 3], [2, 2, 1]) == pytest.approx(1.0)


def test_rmse_known():
    # errors 1, 0, -2 -> sqrt((1 + 0 + 4) / 3)
    assert ev.rmse([1, 2, 3], [2, 2, 1]) == pytest.approx(np.sqrt(5.0 / 3.0))


def test_per_engine_metrics_groups_by_engine():
    df = pd.DataFrame({"unit": [1, 1, 2], "RUL": [10, 0, 5], "pred": [12, -2, 5]})
    em = ev.per_engine_metrics(df).set_index("engine")
    # engine 1 errors: +2, -2 -> bias 0, mae 2, rmse 2
    assert em.loc[1, "bias"] == pytest.approx(0.0)
    assert em.loc[1, "mae"] == pytest.approx(2.0)
    assert em.loc[1, "rmse"] == pytest.approx(2.0)
    # engine 2 error: 0
    assert em.loc[2, "mae"] == pytest.approx(0.0)


def test_summarise_engine_metrics_averages_across_engines():
    df = pd.DataFrame({"unit": [1, 2], "RUL": [0, 0], "pred": [10, 0]})
    em = ev.per_engine_metrics(df)
    summary = ev.summarise_engine_metrics(em)
    assert summary["n_engines"] == 2
    assert summary["engine_mean_mae"] == pytest.approx(5.0)


def test_too_optimistic_rate_margin_sweep():
    df = pd.DataFrame({"unit": [1, 2, 3], "RUL": [0, 0, 0], "pred": [20, 5, -5]})
    em = ev.per_engine_metrics(df)
    assert ev.too_optimistic_rate(em, 10.0) == pytest.approx(1 / 3)
    assert ev.too_optimistic_rate(em, 4.0) == pytest.approx(2 / 3)
    assert ev.too_optimistic_rate(em, 100.0) == pytest.approx(0.0)


def test_row_too_optimistic_rate():
    df = pd.DataFrame({"unit": [1, 1, 1, 1], "RUL": [0, 0, 0, 0], "pred": [20, 5, -5, 15]})
    assert ev.row_too_optimistic_rate(df, margin=10.0) == pytest.approx(0.5)


def test_interval_coverage_inclusive():
    assert ev.interval_coverage([1, 2, 3], [0, 2, 5], [2, 2, 5]) == pytest.approx(2 / 3)


def test_final_cycle_frame_picks_last_cycle_per_engine():
    df = pd.DataFrame({"unit": [1, 1, 2], "cycle": [1, 2, 7]})
    out = ev.final_cycle_frame(df)
    assert out[["unit", "cycle"]].values.tolist() == [[1, 2], [2, 7]]


def test_per_engine_metrics_missing_column_raises():
    with pytest.raises(KeyError):
        ev.per_engine_metrics(pd.DataFrame({"unit": [1]}))
