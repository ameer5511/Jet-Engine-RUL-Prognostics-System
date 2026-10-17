"""Shared pytest fixtures and path setup for the AeroGuard test suite.

Keeps the repo root on ``sys.path`` so top-level modules (``data``, ``features``)
import cleanly, and provides a small synthetic frame for deterministic
feature-logic tests.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import COLUMNS, SENSOR_COLS, SETTING_COLS  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"


@pytest.fixture(scope="session")
def raw_dir() -> Path:
    return RAW_DIR


@pytest.fixture(scope="session")
def have_real_data() -> bool:
    needed = ["train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"]
    return all((RAW_DIR / name).exists() for name in needed)


@pytest.fixture
def synthetic_train() -> pd.DataFrame:
    """Two engines, five cycles each, with known sensor values.

    Engine 1 sensor s1 = [1, 2, 3, 4, 5]
    Engine 2 sensor s1 = [10, 20, 30, 40, 50]
    Sensor s2 is constant (7.0) for both engines.
    All operational settings are constant.
    """
    rows = []
    for unit, s1_values in ((1, [1, 2, 3, 4, 5]), (2, [10, 20, 30, 40, 50])):
        for i, s1 in enumerate(s1_values, start=1):
            row = {"unit": unit, "cycle": i, "s1": float(s1), "s2": 7.0}
            for setting in SETTING_COLS:
                row[setting] = 0.0
            for sensor in SENSOR_COLS:
                row.setdefault(sensor, 0.0)
            rows.append(row)
    frame = pd.DataFrame(rows)
    return frame[["unit", "cycle", *SETTING_COLS, *SENSOR_COLS]]
