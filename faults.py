"""Fault injection for the AeroGuard Robustness Lab.

Applies controlled sensor faults to *held-out* (never training) data:

- noise : add ``N(0, sigma)`` where ``sigma = k * std_train(sensor)``
- drift : add ``k * std_train(sensor) * (t / T_engine)`` to flight ``t``
- missing : randomly blank fraction ``f`` of an engine's flights, then forward-fill
            (first-flight gaps keep the original value)

Faults never touch true RUL labels. All magnitudes are in dataset-relative units
(spec A3). Everything is deterministic per seed — no global RNG leakage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

KINDS = ("noise", "drift", "missing")

NOISE_MAGNITUDES = [0.25, 0.50, 1.00, 2.00]  # k: sigma = k * std_train(sensor)
DRIFT_MAGNITUDES = [0.50, 1.00, 2.00, 4.00]  # k: total shift = k * std_train(sensor)
MISSING_MAGNITUDES = [0.05, 0.10, 0.20, 0.40]  # f: fraction of flights blanked

# Valid ranges for each fault kind (used when validating Nemotron suggestions).
# The fixed grid uses specific values within these ranges.
ACCEPTABLE = {
    "kinds": frozenset(KINDS),
    "noise": (0.0, 4.0),      # 0 < k <= 4  (sigma = k * std_train(sensor))
    "drift": (0.0, 8.0),      # 0 < k <= 8  (total shift = k * std_train(sensor))
    "missing": (0.0, 1.0),    # 0 < f <= 1  (fraction of flights blanked)
}


@dataclass(frozen=True)
class FaultSetting:
    """One concrete fault to apply to one sensor."""

    sensor: str
    kind: str
    magnitude: float
    seed: int


@dataclass(frozen=True)
class FaultConfig:
    """One grid entry: a kind + magnitude, expanded to one setting per sensor."""

    kind: str
    magnitude: float

    def expand(self, sensors: list[str], seed_base: int = 0) -> list[FaultSetting]:
        """One FaultSetting per sensor with deterministic per-sensor seeds."""
        return [
            FaultSetting(
                sensor=s,
                kind=self.kind,
                magnitude=self.magnitude,
                seed=seed_base + i,
            )
            for i, s in enumerate(sensors)
        ]


def fixed_grid(sensors: list[str], seed_base: int = 0) -> list[FaultConfig]:
    """Return the 12 fixed fault configs: 3 kinds x 4 magnitudes."""
    configs: list[FaultConfig] = []
    for k in NOISE_MAGNITUDES:
        configs.append(FaultConfig("noise", k))
    for k in DRIFT_MAGNITUDES:
        configs.append(FaultConfig("drift", k))
    for f in MISSING_MAGNITUDES:
        configs.append(FaultConfig("missing", f))
    return configs


def training_sensor_std(train_frame: pd.DataFrame, sensors: list[str]) -> dict[str, float]:
    """Std of each sensor computed on TRAINING engines only (no leakage)."""
    return {s: float(train_frame[s].std(ddof=0)) for s in sensors}


def validate_setting(
    sensor: str, kind: str, magnitude: float, used_sensors: list[str]
) -> None:
    """Strictly validate a single fault setting; raise ValueError if invalid."""
    if sensor not in used_sensors:
        raise ValueError(
            f"unknown sensor {sensor!r}; must be one of {used_sensors}"
        )
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}; must be one of {KINDS}")
    lo, hi = ACCEPTABLE[kind]
    if not (lo < magnitude <= hi):
        raise ValueError(
            f"{kind} magnitude {magnitude} not in ({lo}, {hi}]"
        )


def _apply_noise(df, sensor, std, magnitude, seed):
    sigma = magnitude * std
    rng = np.random.default_rng(seed)
    df[sensor] = df[sensor].to_numpy() + rng.normal(0.0, sigma, size=len(df))
    return df


def _apply_drift(df, sensor, std, magnitude, seed):
    """Linear drift across each engine's own flight sequence."""
    for unit, group in df.groupby("unit"):
        n = len(group)
        if n < 2:
            continue
        t = np.arange(n, dtype=float) / (n - 1)  # 0 .. 1
        shift = magnitude * std * t
        df.loc[group.index, sensor] = df.loc[group.index, sensor].to_numpy() + shift
    return df


def _apply_missing(df, sensor, magnitude, seed):
    """Blank fraction f of each engine's flights, forward-fill within engine."""
    f = magnitude
    rng = np.random.default_rng(seed)
    for unit, group in df.groupby("unit"):
        n = len(group)
        if n < 2:
            continue
        n_blank = int(round(f * n))
        n_blank = min(max(n_blank, 1), n - 1)  # at least one, never the first flight
        positions = rng.choice(n - 1, size=n_blank, replace=False) + 1  # skip index 0
        df.loc[group.index[positions], sensor] = np.nan
    df[sensor] = df.groupby("unit")[sensor].transform(lambda s: s.ffill())
    return df


def _apply_one(df, sensor_std, sensor, kind, magnitude, seed):
    std = sensor_std[sensor]
    # Ensure float dtype so fault values can be assigned (raw data may be int64).
    df[sensor] = df[sensor].astype(float)
    if kind == "noise":
        return _apply_noise(df, sensor, std, magnitude, seed)
    if kind == "drift":
        return _apply_drift(df, sensor, std, magnitude, seed)
    if kind == "missing":
        return _apply_missing(df, sensor, magnitude, seed)
    raise ValueError(f"unknown kind {kind!r}")


def apply_settings(
    raw: pd.DataFrame, sensor_std: dict[str, float], settings: list[FaultSetting]
) -> pd.DataFrame:
    """Apply multiple fault settings to raw held-out data. Returns a copy."""
    df = raw.copy()
    for st in settings:
        if st.sensor not in sensor_std:
            raise KeyError(f"sensor {st.sensor!r} not in sensor_std")
        df = _apply_one(df, sensor_std, st.sensor, st.kind, st.magnitude, st.seed)
    return df


def apply_config(
    raw: pd.DataFrame,
    sensor_std: dict[str, float],
    config: FaultConfig,
    sensors: list[str],
    seed_base: int = 0,
) -> pd.DataFrame:
    """Convenience: expand one config to all sensors and apply."""
    settings = config.expand(sensors, seed_base=seed_base)
    return apply_settings(raw, sensor_std, settings)
