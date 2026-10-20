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


def test_nasa_score_zero_when_perfect():
    y_true = [10, 20, 30]
    y_pred = [10, 20, 30]
    d = np.array(y_pred) - np.array(y_true)
    assert ev.nasa_score(d) == pytest.approx(0.0)


def test_nasa_score_sign_and_magnitude():
    # d = pred - truth = +10 (pred too HIGH / optimistic / dangerous): exp(10/10)-1 (steeper)
    opt = ev.nasa_score(np.array([10]) - np.array([0]))
    assert opt == pytest.approx(np.exp(10 / 10.0) - 1.0)
    # d = -10 (pred too LOW / conservative / safe): exp(-(-10)/13)-1 = exp(10/13)-1 (less steep)
    cons = ev.nasa_score(np.array([0]) - np.array([10]))
    assert cons == pytest.approx(np.exp(10 / 13.0) - 1.0)
    # over-predict (dangerous) penalised more per unit than under-predict
    assert opt > cons


def test_nasa_score_is_sum_of_per_row():
    y_true = [0, 10]
    y_pred = [10, 0]
    d = np.array(y_pred) - np.array(y_true)
    total = ev.nasa_score(d)
    per = sum(ev.nasa_score(np.array([b]) - np.array([a])) for a, b in zip(y_true, y_pred))
    assert total == pytest.approx(per)


def test_nasa_score_per_engine():
    df = pd.DataFrame({
        "unit": [1, 1, 1, 2, 2],
        "cycle": [1, 2, 10, 1, 5],
        "RUL": [0, 10, 0, 0, 7],
        "pred": [10, 0, 5, 5, 0],
    })
    out = ev.nasa_score_per_engine(df).set_index("engine")
    assert set(out.index) == {1, 2}
    # engine 1 last cycle (cycle=10): d = 5 - 0 = +5 (over-predict): exp(5/10)-1
    assert out.loc[1, "nasa_score"] == pytest.approx(np.exp(5 / 10.0) - 1.0)
    # engine 2 last cycle (cycle=5): d = 0 - 7 = -7 (under-predict): exp(-(-7)/13)-1 = exp(7/13)-1
    assert out.loc[2, "nasa_score"] == pytest.approx(np.exp(7 / 13.0) - 1.0)
