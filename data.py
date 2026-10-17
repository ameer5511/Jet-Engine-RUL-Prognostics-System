"""Data loading, RUL labels, and engine-level splitting for NASA C-MAPSS FD001.

Verified against ``data/raw/readme.txt``:
- Files are whitespace-separated with no header row.
- 26 columns: unit, cycle, 3 operational settings, 21 sensor measurements.
- FD001 has 100 training engines (run to failure) and 100 test engines (truncated).

Design rules (see MASTER_SPEC.md):
- Training labels are capped (A1); metrics use the *uncapped* true RUL.
- The train/validation split is by whole engines, never by row.
- The official test set is only reachable through :func:`load_official_test`,
  intended for the final evaluation run and never for training.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

N_SETTINGS = 3
N_SENSORS = 21
SETTING_COLS: List[str] = [f"setting_{i}" for i in range(1, N_SETTINGS + 1)]
SENSOR_COLS: List[str] = [f"s{i}" for i in range(1, N_SENSORS + 1)]
COLUMNS: List[str] = ["unit", "cycle", *SETTING_COLS, *SENSOR_COLS]

_READ_KWARGS = dict(sep=r"\s+", header=None, names=COLUMNS)


def _paths(dataset: str, raw_dir: str | Path) -> Tuple[Path, Path, Path]:
    """Return (train, test, rul) file paths for one C-MAPSS subset."""
    raw = Path(raw_dir)
    return (
        raw / f"train_{dataset}.txt",
        raw / f"test_{dataset}.txt",
        raw / f"RUL_{dataset}.txt",
    )


def _read_frame(path: Path) -> pd.DataFrame:
    """Read one whitespace-separated C-MAPSS series file."""
    if not path.exists():
        raise FileNotFoundError(f"expected data file not found: {path}")
    frame = pd.read_csv(path, **_READ_KWARGS)
    # A trailing space on every line can produce an extra all-NaN column.
    frame = frame.dropna(axis=1, how="all")
    if frame.shape[1] != len(COLUMNS):
        raise ValueError(
            f"{path.name}: expected {len(COLUMNS)} columns, got {frame.shape[1]}"
        )
    return frame


def _read_rul(path: Path) -> pd.Series:
    """Read a RUL_<subset>.txt file as a Series indexed by unit (1..N).

    The RUL file lists one value per test engine, in ascending unit order.
    """
    if not path.exists():
        raise FileNotFoundError(f"expected data file not found: {path}")
    raw = pd.read_csv(path, sep=r"\s+", header=None)
    values = raw.to_numpy().reshape(-1).astype(float)
    if values.size == 0:
        raise ValueError(f"{path.name}: no RUL values found")
    return pd.Series(values, index=np.arange(1, values.size + 1), name="RUL")


def load_raw(
    dataset: str = "FD001", raw_dir: str | Path = "data/raw"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load train, test, and true-RUL for one C-MAPSS subset.

    Returns
    -------
    train_df : per-cycle rows for run-to-failure training engines.
    test_df  : per-cycle rows for the truncated test engines.
    test_rul : Series indexed by unit giving the true RUL *after* the engine's
               last recorded cycle (the official competition target).
    """
    train_path, test_path, rul_path = _paths(dataset, raw_dir)
    return _read_frame(train_path), _read_frame(test_path), _read_rul(rul_path)


def engine_ids(df: pd.DataFrame) -> np.ndarray:
    """Return the sorted unique engine ids in a frame."""
    return np.sort(df["unit"].unique())


def add_rul(df: pd.DataFrame, cap: float | None = None) -> pd.DataFrame:
    """Return a copy with RUL labels.

    ``RUL`` is the uncapped number of cycles remaining until failure
    (0 on an engine's last recorded cycle). If ``cap`` is given, an extra
    ``RUL_capped`` column = min(RUL, cap) is added for training.
    """
    frame = df.sort_values(["unit", "cycle"]).reset_index(drop=True).copy()
    max_cycle = frame.groupby("unit")["cycle"].transform("max")
    rul = (max_cycle - frame["cycle"]).astype(float)
    frame["RUL"] = rul
    if cap is not None:
        frame["RUL_capped"] = np.minimum(rul, float(cap))
    return frame


def split_engines(
    engine_ids_in: np.ndarray, val_fraction: float = 0.2, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Split engine ids into (train_ids, val_ids), deterministic by seed."""
    ids = np.sort(np.asarray(engine_ids_in))
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    n_val = int(round(ids.size * val_fraction))
    if not 1 <= n_val < ids.size:
        raise ValueError(
            f"val_fraction={val_fraction} yields {n_val} val engines from {ids.size}"
        )
    shuffled = np.random.RandomState(seed).permutation(ids)
    val_ids = np.sort(shuffled[:n_val])
    train_ids = np.sort(shuffled[n_val:])
    return train_ids, val_ids


def assert_disjoint(train_ids, val_ids) -> None:
    """Raise if any engine appears in both splits (leakage guard)."""
    overlap = set(np.asarray(train_ids).tolist()) & set(np.asarray(val_ids).tolist())
    if overlap:
        raise ValueError(f"engine overlap between splits: {sorted(overlap)}")


def build_train_val(
    dataset: str = "FD001",
    raw_dir: str | Path = "data/raw",
    train_cap: float | None = None,
    val_fraction: float = 0.2,
    seed: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Build labelled train/validation splits from the training engines only.

    Training targets are uncapped RUL by default so predictions span the full
    true range (matching metrics which score vs uncapped RUL). Pass
    ``train_cap`` to cap training labels for experiments.

    The official test engines are never loaded here.
    """
    train_df, _, _ = load_raw(dataset, raw_dir)
    labelled = add_rul(train_df, cap=train_cap)
    train_ids, val_ids = split_engines(engine_ids(labelled), val_fraction, seed)
    assert_disjoint(train_ids, val_ids)

    train_split = labelled[labelled["unit"].isin(train_ids)].reset_index(drop=True)
    val_split = labelled[labelled["unit"].isin(val_ids)].reset_index(drop=True)

    meta: Dict = {
        "dataset": dataset,
        "train_cap": train_cap,
        "val_fraction": val_fraction,
        "seed": seed,
        "n_train_engines": int(train_ids.size),
        "n_val_engines": int(val_ids.size),
        "train_ids": train_ids.tolist(),
        "val_ids": val_ids.tolist(),
    }
    return train_split, val_split, meta


def load_official_test(
    dataset: str = "FD001", raw_dir: str | Path = "data/raw"
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load the held-out test engines.

    The returned frame has an ``RUL`` column populated only on each engine's
    *last* recorded cycle (the official scoring row); other rows are NaN.
    Call this only for the final evaluation run (A5 guardrail).
    """
    _, test_df, test_rul = load_raw(dataset, raw_dir)
    frame = test_df.sort_values(["unit", "cycle"]).reset_index(drop=True).copy()
    is_last = frame["cycle"] == frame.groupby("unit")["cycle"].transform("max")
    frame["RUL"] = np.nan
    frame.loc[is_last, "RUL"] = frame.loc[is_last, "unit"].map(test_rul)
    if int(frame["RUL"].notna().sum()) != int(test_rul.size):
        raise ValueError("RUL mapping did not cover every test engine")
    return frame, test_rul
