"""Phase 2 orchestration: fit models on the train split, evaluate on validation.

Nothing here touches the official test set (A5). All heavy work writes files to
``results/`` so the dashboard can stay read-only (see MASTER_SPEC.md).

Run:
    conda run -n ml python train.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import data as data_mod
import evaluate as evaluate_mod
from config import load_config
from features import add_features
from models import MODEL_NAMES, build_model

RANGE_LOW, RANGE_HIGH = "quantile_10", "quantile_90"
PREDICTED_MODELS = MODEL_NAMES


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _jsonable(obj):
    """Recursively convert numpy/pandas scalars so json.dump always works."""
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


def _aggregate(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Mean/std across seeds for each scalar key."""
    if not rows:
        return {}
    keys = sorted(rows[0].keys())
    out: Dict[str, Dict[str, float]] = {}
    for key in keys:
        values = np.asarray([row[key] for row in rows], dtype=float)
        out[key] = {"mean": float(values.mean()), "std": float(values.std(ddof=0))}
    return out


def _prediction_metrics(frame: pd.DataFrame) -> Dict[str, float]:
    """All scalar metrics for one model's validation predictions."""
    engine_metrics = evaluate_mod.per_engine_metrics(frame)
    summary = evaluate_mod.summarise_engine_metrics(engine_metrics)
    final_metrics = evaluate_mod.per_engine_metrics(evaluate_mod.final_cycle_frame(frame))
    return {
        "pooled_mae": evaluate_mod.mae(frame["RUL"], frame["pred"]),
        "pooled_rmse": evaluate_mod.rmse(frame["RUL"], frame["pred"]),
        **summary,
        "too_opt_engine_margin5": evaluate_mod.too_optimistic_rate(engine_metrics, 5.0),
        "too_opt_engine_margin10": evaluate_mod.too_optimistic_rate(engine_metrics, 10.0),
        "too_opt_engine_margin20": evaluate_mod.too_optimistic_rate(engine_metrics, 20.0),
        "too_opt_row_margin10": evaluate_mod.row_too_optimistic_rate(frame, margin=10.0),
        "finalcycle_mae": float(final_metrics["mae"].mean()),
        "finalcycle_bias": float(final_metrics["bias"].mean()),
    }


# --------------------------------------------------------------------------- #
# data preparation
# --------------------------------------------------------------------------- #
def prepare(cfg: dict, raw_dir: str | Path):
    """Load the train/val split and build features for both sides.

    The constant-sensor decision is made on the train split only and reused for
    validation, which keeps the choice leakage-free.
    """
    split_cfg = cfg.get("split", {})
    seed = int(split_cfg.get("seed", 0))
    val_fraction = float(split_cfg.get("val_fraction", 0.2))
    train_cap = cfg.get("labels", {}).get("train_cap", None)
    window = int(cfg.get("features", {}).get("window", 10))
    dataset = cfg.get("data", {}).get("dataset", "FD001")

    train_split, val_split, meta = data_mod.build_train_val(
        dataset, raw_dir, train_cap=train_cap, val_fraction=val_fraction, seed=seed
    )
    fs_train = add_features(train_split, data_mod.SENSOR_COLS, window=window)
    fs_val = add_features(
        val_split,
        data_mod.SENSOR_COLS,
        window=window,
        constant=fs_train.dropped_sensors,
    )
    meta.update(
        {
            "window": window,
            "n_features": len(fs_train.feature_cols),
            "dropped_sensors": list(fs_train.dropped_sensors),
            "used_sensors": list(fs_train.used_sensors),
        }
    )
    return fs_train, fs_val, meta


# --------------------------------------------------------------------------- #
# main run
# --------------------------------------------------------------------------- #
def run_phase2(cfg: dict, raw_dir: str | Path = "data/raw") -> Dict:
    """Train every model for every seed and evaluate on validation engines."""
    seeds = [int(s) for s in cfg["project"]["seed_registry"]]
    fs_train, fs_val, meta = prepare(cfg, raw_dir)

    feature_cols = fs_train.feature_cols
    X_train = fs_train.frame[feature_cols].to_numpy(dtype=float)
    y_train = fs_train.frame["RUL"].to_numpy(dtype=float)

    val = fs_val.frame
    X_val = val[feature_cols].to_numpy(dtype=float)

    per_seed: Dict[str, List[Dict[str, float]]] = {m: [] for m in PREDICTED_MODELS}
    coverage_rows: List[Dict[str, float]] = []
    primary_predictions: pd.DataFrame | None = None

    for seed in seeds:
        preds: Dict[str, np.ndarray] = {}
        for name in PREDICTED_MODELS:
            estimator = build_model(name, seed=seed)
            estimator.fit(X_train, y_train)
            preds[name] = np.asarray(estimator.predict(X_val), dtype=float)

        for name in PREDICTED_MODELS:
            frame = val[["unit", "cycle", "RUL"]].copy()
            frame["pred"] = preds[name]
            per_seed[name].append(_prediction_metrics(frame))

        coverage = evaluate_mod.interval_coverage(
            y_true=val["RUL"].to_numpy(),
            low=preds[RANGE_LOW],
            high=preds[RANGE_HIGH],
        )
        coverage_rows.append({"seed": seed, "coverage_80": float(coverage)})

        if seed == seeds[0]:
            primary = val[["unit", "cycle", "RUL"]].copy()
            for name in PREDICTED_MODELS:
                primary[name] = preds[name]
            primary_predictions = primary

    results: Dict = {
        "meta": metrics_meta(meta, cfg),
        "seeds": seeds,
        "primary_seed": seeds[0],
        "per_model": {
            name: {
                "per_seed": per_seed[name],
                "aggregate": _aggregate(per_seed[name]),
            }
            for name in PREDICTED_MODELS
        },
        "coverage": {
            "per_seed": coverage_rows,
            "aggregate": _aggregate(
                [{"coverage_80": row["coverage_80"]} for row in coverage_rows]
            ),
        },
        "predictions": primary_predictions,
    }
    return results


def metrics_meta(meta: dict, cfg: dict) -> dict:
    """Small metadata block recorded with the metrics."""
    return {
        "dataset": meta.get("dataset"),
        "cap": meta.get("cap"),
        "window": meta.get("window"),
        "n_features": meta.get("n_features"),
        "n_train_engines": meta.get("n_train_engines"),
        "n_val_engines": meta.get("n_val_engines"),
        "split_seed": meta.get("seed"),
        "dropped_sensors": meta.get("dropped_sensors"),
    }


# --------------------------------------------------------------------------- #
# artifacts
# --------------------------------------------------------------------------- #
def save_artifacts(results: dict, out_dir: str | Path = "results", cfg: dict | None = None) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    predictions = results["predictions"]
    if predictions is not None:
        predictions.to_csv(out / "phase2_val_predictions.csv", index=False)

    metrics = {k: v for k, v in results.items() if k != "predictions"}
    with (out / "phase2_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(metrics), handle, indent=2)

    if cfg is not None:
        with (out / "phase2_resolved_config.json").open("w", encoding="utf-8") as handle:
            json.dump(_jsonable(cfg), handle, indent=2)


def print_summary(results: dict) -> None:
    print(f"val engines: {results['meta']['n_val_engines']}, "
          f"features: {results['meta']['n_features']}, seeds: {results['seeds']}")
    header = f"{'model':18} {'MAE':>12} {'RMSE':>12} {'bias':>12} {'P(opt>10)':>11} {'cov80':>8}"
    print(header)
    print("-" * len(header))
    coverage = results["coverage"]["aggregate"].get("coverage_80", {})
    for name, block in results["per_model"].items():
        agg = block["aggregate"]
        mae = agg["engine_mean_mae"]
        rmse = agg["engine_mean_rmse"]
        bias = agg["engine_mean_bias"]
        opt = agg["too_opt_engine_margin10"]
        cov = f"{coverage.get('mean', float('nan')):.3f}" if name in (RANGE_LOW, RANGE_HIGH) else "-"
        print(f"{name:18} {mae['mean']:6.2f}±{mae['std']:.2f} {rmse['mean']:6.2f} "
              f"{bias['mean']:7.2f} {opt['mean']:10.3f} {cov:>8}")
    print(f"80% interval coverage: {coverage.get('mean', float('nan')):.3f} "
          f"(nominal 0.80)")


def main() -> None:
    cfg = load_config()
    raw_dir = cfg.get("data", {}).get("raw_dir", "data/raw")
    out_dir = cfg.get("outputs", {}).get("dir", "results")
    results = run_phase2(cfg, raw_dir=raw_dir)
    save_artifacts(results, out_dir=out_dir, cfg=cfg)
    print_summary(results)


if __name__ == "__main__":
    main()
