"""Tests for dashboard_data.py: artifact loader for the dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest

import dashboard_data as dd


def test_artifact_status_returns_dict():
    status = dd.artifact_status()
    assert isinstance(status, dict)
    for key in ("predictions_csv", "metrics_json", "config_json",
                "stress_json", "stress_csv", "nemotron_cache"):
        assert key in status
        assert isinstance(status[key], bool)


def test_load_predictions_returns_dataframe():
    df = dd.load_predictions()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "unit" in df.columns
    assert "cycle" in df.columns
    assert "RUL" in df.columns


def test_get_val_engines():
    engines = dd.get_val_engines()
    assert isinstance(engines, list)
    assert len(engines) > 0
    assert all(isinstance(e, int) for e in engines)


def test_get_prediction_models():
    models = dd.get_prediction_models()
    assert isinstance(models, list)
    assert "constant" in models
    assert "random_forest" in models
    assert "gradient_boosting" in models
    assert "safe_side" in models


def test_get_engine_predictions():
    engines = dd.get_val_engines()
    eng_id = engines[0]
    df = dd.get_engine_predictions(eng_id)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert (df["unit"] == eng_id).all()
    assert list(df["cycle"].sort_values()) == list(df["cycle"])  # sorted


def test_get_model_metrics_summary():
    df = dd.get_model_metrics_summary()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "model" in df.columns
    assert "engine_mean_mae" in df.columns
    assert "nasa_score" in df.columns


def test_load_stress_json():
    data = dd.load_stress_json()
    assert isinstance(data, dict)
    assert "meta" in data
    assert "aggregate" in data
    assert "per_seed" in data


def test_load_stress_csv():
    df = dd.load_stress_csv()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_get_nemotron_count():
    count = dd.get_nemotron_count()
    assert isinstance(count, int)
    assert count >= 0


def test_disclaimer_constant():
    assert dd.DISCLAIMER == (
        "NASA C-MAPSS simulation data only. Not for real maintenance "
        "or airworthiness decisions."
    )


def test_get_engine_per_engine_metrics():
    engines = dd.get_val_engines()
    eng_id = engines[0]
    metrics = dd.get_engine_per_engine_metrics(eng_id)
    assert metrics is not None
    for model in dd.get_prediction_models():
        assert model in metrics
        m = metrics[model]
        for key in ("mae", "rmse", "bias", "worst_mae", "too_opt"):
            assert key in m
            assert isinstance(m[key], float)
