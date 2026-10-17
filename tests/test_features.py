"""Tests for features.py: past-only rolling features and constant-sensor handling."""
from __future__ import annotations

import numpy as np
import pytest

import data as data_mod
from data import load_raw
from features import add_features, assert_no_nan, constant_sensors


def test_constant_sensors_detects_constant(synthetic_train):
    dropped = constant_sensors(synthetic_train, ["s1", "s2"])
    assert dropped == ["s2"]


def test_feature_values_known(synthetic_train):
    fs = add_features(synthetic_train, ["s1"], window=3, drop_constant=False)
    unit1 = fs.frame[fs.frame["unit"] == 1].reset_index(drop=True)
    assert unit1["s1_cur"].tolist() == [1, 2, 3, 4, 5]
    assert unit1["s1_mean"].tolist() == [1, 1.5, 2, 3, 4]
    assert unit1["s1_trend"].tolist() == [0, 1, 2, 2, 2]


def test_trend_zero_on_first_row(synthetic_train):
    fs = add_features(synthetic_train, ["s1"], window=5, drop_constant=False)
    first_rows = fs.frame.groupby("unit", sort=False).head(1)
    assert (first_rows["s1_trend"] == 0).all()


def test_past_only_prefix_invariance(synthetic_train):
    # Features for a prefix must equal the same rows computed on the full frame:
    # proves no dependence on future rows or other engines.
    fs_full = add_features(synthetic_train, ["s1"], window=3, drop_constant=False)
    unit1_prefix = synthetic_train[synthetic_train["unit"] == 1].head(3)
    fs_prefix = add_features(unit1_prefix, ["s1"], window=3, drop_constant=False)
    np.testing.assert_allclose(
        fs_full.frame[fs_full.feature_cols].head(3).to_numpy(),
        fs_prefix.frame[fs_prefix.feature_cols].head(3).to_numpy(),
    )


def test_engine_boundary_reset(synthetic_train):
    fs = add_features(synthetic_train, ["s1"], window=5, drop_constant=False)
    u2_first = fs.frame[fs.frame["unit"] == 2].reset_index(drop=True).iloc[0]
    assert u2_first["s1_cur"] == 10
    assert u2_first["s1_mean"] == 10  # not contaminated by engine 1
    assert u2_first["s1_trend"] == 0


def test_constant_sensor_dropped_by_default(synthetic_train):
    fs = add_features(synthetic_train, ["s1", "s2"], window=3)
    assert "s2" in fs.dropped_sensors
    assert "s2" not in fs.used_sensors
    assert fs.feature_cols and all(c.startswith("s1_") for c in fs.feature_cols)


def test_precomputed_constant_list_used(synthetic_train):
    fs = add_features(synthetic_train, ["s1", "s2"], window=3, constant=["s2"])
    assert fs.dropped_sensors == ["s2"]
    assert "s2" not in fs.used_sensors


def test_window_validation(synthetic_train):
    with pytest.raises(ValueError):
        add_features(synthetic_train, ["s1"], window=0)


def test_missing_sensor_raises(synthetic_train):
    with pytest.raises(KeyError):
        add_features(synthetic_train, ["nope"], window=3, drop_constant=False)


def test_no_nan_features(synthetic_train):
    fs = add_features(synthetic_train, ["s1"], window=4, drop_constant=False)
    assert_no_nan(fs.frame, fs.feature_cols)


def test_real_data_feature_counts(have_real_data, raw_dir):
    if not have_real_data:
        pytest.skip("real FD001 data not present")
    train_df, _, _ = load_raw("FD001", raw_dir)
    fs = add_features(train_df, data_mod.SENSOR_COLS, window=10)
    assert fs.dropped_sensors == constant_sensors(train_df, data_mod.SENSOR_COLS)
    assert len(fs.used_sensors) == 21 - len(fs.dropped_sensors)
    assert len(fs.feature_cols) == 3 * len(fs.used_sensors)
    assert_no_nan(fs.frame, fs.feature_cols)


def test_s6_not_strictly_constant_but_near_constant(have_real_data, raw_dir):
    if not have_real_data:
        pytest.skip("real FD001 data not present")
    train_df, _, _ = load_raw("FD001", raw_dir)
    assert "s6" not in constant_sensors(train_df, ["s6"], atol=0.0)
    assert "s6" in constant_sensors(train_df, ["s6"], atol=0.01)
