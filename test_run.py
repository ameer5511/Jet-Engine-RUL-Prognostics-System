"""Phase 5: Official test set run (A5 guardrail) - Gradient Boosting only.

Trains Gradient Boosting on the FULL training set (no holdout), then scores on
the official C-MAPSS FD001 test engines at last cycle only. Produces a single
locked result artifact: results/phase3_test_results.json.

Never touches the test set during training or development.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import data as data_mod
import evaluate as evaluate_mod
from config import load_config
from features import add_features
from models import build_model


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main() -> None:
    cfg = load_config()
    raw_dir = cfg.get("data", {}).get("raw_dir", "data/raw")
    out_dir = Path(cfg.get("outputs", {}).get("dir", "results"))
    window = int(cfg.get("features", {}).get("window", 10))
    seed = 0  # locked seed

    # ---------- Train Gradient Boosting on FULL training set ----------
    train_df, _, _ = data_mod.load_raw("FD001", raw_dir)
    labelled = data_mod.add_rul(train_df, cap=None)  # uncapped targets

    fs_train = add_features(labelled, data_mod.SENSOR_COLS, window=window)
    feature_cols = fs_train.feature_cols
    X_train = fs_train.frame[feature_cols].to_numpy(dtype=float)
    y_train = fs_train.frame["RUL"].to_numpy(dtype=float)

    # ---------- Load official test set ----------
    test_df, test_rul = data_mod.load_official_test("FD001", raw_dir)
    fs_test = add_features(
        test_df,
        data_mod.SENSOR_COLS,
        window=window,
        constant=fs_train.dropped_sensors,
    )
    X_test = fs_test.frame[feature_cols].to_numpy(dtype=float)
    test_units = fs_test.frame["unit"].to_numpy()
    test_cycles = fs_test.frame["cycle"].to_numpy()

    # ---------- Predict on test set ----------
    estimator = build_model("gradient_boosting", seed=seed)
    estimator.fit(X_train, y_train)
    preds = np.asarray(estimator.predict(X_test), dtype=float)

    # Build frame with last-cycle RUL per engine
    frame = pd.DataFrame(
        {
            "unit": test_units,
            "cycle": test_cycles,
            "RUL": fs_test.frame["RUL"].to_numpy(),  # NaN except last cycle
            "pred": preds,
        }
    )

    # NASA score and MAE at last cycle only (Saxena & Goebel PHM 2008)
    last_cycle = evaluate_mod.final_cycle_frame(frame)
    d = last_cycle["pred"].to_numpy(dtype=float) - last_cycle["RUL"].to_numpy(dtype=float)
    nasa = evaluate_mod.nasa_score(d)
    mae = float(np.mean(np.abs(last_cycle["pred"].to_numpy(dtype=float) - last_cycle["RUL"].to_numpy(dtype=float))))

    results = {
        "meta": {
            "dataset": "FD001",
            "n_train_engines": int(fs_train.frame["unit"].nunique()),
            "n_test_engines": int(test_rul.size),
            "window": window,
            "n_features": len(feature_cols),
            "split_seed": seed,
            "dropped_sensors": list(fs_train.dropped_sensors),
            "used_sensors": list(fs_train.used_sensors),
            "note": "Locked official test set run. Gradient Boosting trained on full training data, scored on test engines at last cycle only.",
        },
        "per_model": {
            "gradient_boosting": {
                "nasa_score_at_last_cycle": nasa,
                "mae": mae,
            }
        },
        "nasa_scores": {"gradient_boosting": nasa},
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "phase3_test_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(_jsonable(results), fh, indent=2)

    print(f"Official test results written to {out_path}")
    print(f"Gradient Boosting: NASA score = {nasa:.1f}, MAE = {mae:.1f}")


if __name__ == "__main__":
    main()
