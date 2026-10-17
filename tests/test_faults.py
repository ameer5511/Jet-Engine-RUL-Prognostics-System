"""Tests for faults.py: determinism, per-sensor isolation, label safety, ranges."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from faults import (
    ACCEPTABLE,
    FaultConfig,
    FaultSetting,
    apply_config,
    apply_settings,
    fixed_grid,
    training_sensor_std,
    validate_setting,
)


def _raw_frame() -> pd.DataFrame:
    rows = []
    for unit, base in ((1, 10.0), (2, 50.0)):
        for i in range(1, 7):
            rows.append({"unit": unit, "cycle": i, "s1": base + i, "s2": 3.0 + i * 0.5})
    return pd.DataFrame(rows)


def test_fixed_grid_has_12_configs():
    grid = fixed_grid(["s1"])
    assert len(grid) == 12
    kinds = [c.kind for c in grid]
    assert kinds.count("noise") == 4
    assert kinds.count("drift") == 4
    assert kinds.count("missing") == 4


def test_grid_magnitudes_match_spec():
    grid = fixed_grid(["s1"])
    assert [c.magnitude for c in grid[:4]] == [0.25, 0.50, 1.00, 2.00]
    assert [c.magnitude for c in grid[4:8]] == [0.50, 1.00, 2.00, 4.00]
    assert [c.magnitude for c in grid[8:12]] == [0.05, 0.10, 0.20, 0.40]


def test_grid_expands_per_sensor():
    cfg = FaultConfig("noise", 0.5)
    settings = cfg.expand(["s1", "s2"])
    assert len(settings) == 2
    assert {s.sensor for s in settings} == {"s1", "s2"}
    assert all(s.kind == "noise" and s.magnitude == 0.5 for s in settings)
    assert settings[0].seed != settings[1].seed


def test_grid_is_deterministic():
    assert fixed_grid(["s1"]) == fixed_grid(["s1"])
    assert FaultConfig("noise", 0.5).expand(["s1"]) == FaultConfig("noise", 0.5).expand(["s1"])


def test_training_sensor_std_uses_training_only():
    train = _raw_frame()
    std = training_sensor_std(train, ["s1", "s2"])
    assert std["s1"] > 0
    assert std["s2"] > 0
    assert set(std) == {"s1", "s2"}


def test_validate_setting_rejects_bad_input():
    used = ["s1", "s2"]
    with pytest.raises(ValueError):
        validate_setting("s9", "noise", 0.5, used)
    with pytest.raises(ValueError):
        validate_setting("s1", "bogus", 0.5, used)
    with pytest.raises(ValueError):
        validate_setting("s1", "noise", -0.5, used)
    with pytest.raises(ValueError):
        validate_setting("s1", "noise", 99.0, used)
    with pytest.raises(ValueError):
        validate_setting("s1", "missing", 1.5, used)
    # valid ones pass
    validate_setting("s1", "noise", 0.5, used)
    validate_setting("s2", "drift", 2.0, used)
    validate_setting("s1", "missing", 0.10, used)


def test_noise_shifts_by_approx_k_times_std():
    raw = _raw_frame()
    std = training_sensor_std(raw, ["s1"])
    cfg = FaultConfig("noise", 1.0)
    faulted = apply_config(raw, std, cfg, ["s1"])
    assert not faulted["s1"].equals(raw["s1"])
    assert faulted["s2"].equals(raw["s2"])  # untouched sensor
    diff = faulted["s1"] - raw["s1"]
    assert np.isfinite(diff).all()


def test_drift_shifts_across_engine():
    raw = _raw_frame()
    std = training_sensor_std(raw, ["s1"])
    cfg = FaultConfig("drift", 1.0)
    faulted = apply_config(raw, std, cfg, ["s1"])
    for unit in (1, 2):
        sub = faulted[faulted["unit"] == unit].sort_values("cycle")
        orig = raw[raw["unit"] == unit].sort_values("cycle")
        shift = (sub["s1"] - orig["s1"]).to_numpy(dtype=float)
        assert np.isclose(shift[0], 0.0, atol=1e-9)
        assert np.isclose(shift[-1], std["s1"], atol=1e-6)
        assert np.all(np.diff(shift) >= 0)


def test_missing_blanks_fraction_and_forward_fills():
    raw = _raw_frame()
    std = training_sensor_std(raw, ["s1"])
    cfg = FaultConfig("missing", 0.5)
    faulted = apply_config(raw, std, cfg, ["s1"])
    for unit in (1, 2):
        sub = faulted[faulted["unit"] == unit].sort_values("cycle")
        assert sub["s1"].notna().all()  # forward-fill removed NaNs
        orig = raw[raw["unit"] == unit].sort_values("cycle")
        changed = (sub["s1"].to_numpy() != orig["s1"].to_numpy()).any()
        assert changed


def test_missing_first_flight_keeps_original():
    raw = _raw_frame()
    std = training_sensor_std(raw, ["s1"])
    cfg = FaultConfig("missing", 1.0)
    faulted = apply_config(raw, std, cfg, ["s1"])
    for unit in (1, 2):
        first = faulted[faulted["unit"] == unit].sort_values("cycle").iloc[0]
        orig_first = raw[raw["unit"] == unit].sort_values("cycle").iloc[0]
        assert first["s1"] == orig_first["s1"]


def test_labels_never_touched():
    raw = _raw_frame().copy()
    raw["RUL"] = [100, 99, 98, 97, 96, 95, 90, 89, 88, 87, 86, 85]
    std = training_sensor_std(raw, ["s1"])
    cfg = FaultConfig("noise", 1.0)
    faulted = apply_config(raw, std, cfg, ["s1"])
    assert faulted["RUL"].equals(raw["RUL"])


def test_settings_applied_to_copy_not_original():
    raw = _raw_frame()
    std = training_sensor_std(raw, ["s1"])
    cfg = FaultConfig("noise", 1.0)
    before = raw["s1"].copy()
    apply_config(raw, std, cfg, ["s1"])
    assert raw["s1"].equals(before)


def test_unknown_sensor_raises_on_apply():
    raw = _raw_frame()
    std = training_sensor_std(raw, ["s1"])
    setting = FaultSetting("s9", "noise", 0.5, 0)
    with pytest.raises(KeyError):
        apply_settings(raw, std, [setting])


def test_multiple_settings_compound():
    raw = _raw_frame()
    std = training_sensor_std(raw, ["s1"])
    s1 = FaultSetting("s1", "noise", 0.5, 0)
    s2 = FaultSetting("s1", "drift", 1.0, 1)
    faulted = apply_settings(raw, std, [s1, s2])
    assert not faulted["s1"].equals(raw["s1"])
