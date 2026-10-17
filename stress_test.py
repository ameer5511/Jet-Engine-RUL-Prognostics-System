"""Phase 4 stress test: measure metric degradation under sensor faults.

Fits models once per seed on clean train, then applies each fault scenario to
the held-out val engines and measures how predictions degrade.

Design rules (MASTER_SPEC.md):
- Faults applied only to held-out (val) data, never training (A5).
- Models trained on uncapped RUL; metrics vs uncapped true RUL (A1).
- Primary metrics per engine; per-flight secondary (A6).
- Deterministic per seed; seed spread = mean ± std across 5 seeds.
- Nothing touches the official test set until the final run.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import data as data_mod
import evaluate as evaluate_mod
from config import load_config
from features import add_features
from faults import training_sensor_std
from models import build_model
from scenarios import get_scenarios, apply_scenario

# Models stress-tested (quantile ranges excluded - they're diagnostic, not primary)
STRESS_MODELS = ["constant", "random_forest", "gradient_boosting", "safe_side"]


def _jsonable(obj):
    """Recursively convert numpy/pandas scalars for json.dump."""
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


def _prepare(cfg, raw_dir, seed):
    """Build train/val feature sets for one split seed."""
    split_cfg = cfg.get("split", {})
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
    return fs_train, fs_val, meta


def _scenario_metrics(frame: pd.DataFrame, true_col: str, model_col: str) -> dict:
    """Compute per-engine metrics for one model on one scenario frame."""
    engine_metrics = evaluate_mod.per_engine_metrics(frame, pred_col=model_col)
    summary = evaluate_mod.summarise_engine_metrics(engine_metrics)
    return {
        "engine_mean_mae": summary["engine_mean_mae"],
        "engine_mean_rmse": summary["engine_mean_rmse"],
        "engine_mean_bias": summary["engine_mean_bias"],
        "engine_worst_mae": summary["engine_worst_mae"],
        "too_opt_engine_margin10": evaluate_mod.too_optimistic_rate(engine_metrics, 10.0),
        "pooled_mae": evaluate_mod.mae(frame[true_col], frame[model_col]),
        "pooled_rmse": evaluate_mod.rmse(frame[true_col], frame[model_col]),
    }


def run_stress_test(cfg: dict, raw_dir: str = "data/raw", out_dir: str = "results") -> dict:
    """Run the full stress test across seeds, scenarios, and models."""
    seeds = [int(s) for s in cfg["project"]["seed_registry"]]
    window = int(cfg.get("features", {}).get("window", 10))

    used_sensors: list[str] | None = None
    sensor_std: dict[str, float] | None = None

    results: dict[str, Any] = {
        "meta": {
            "seeds": seeds,
            "models": STRESS_MODELS,
            "n_scenarios": 0,
        },
        "per_seed": [],
    }

    for seed in seeds:
        fs_train, fs_val, meta = _prepare(cfg, raw_dir, seed)
        if used_sensors is None:
            used_sensors = fs_train.used_sensors
            sensor_std = training_sensor_std(fs_train.frame, used_sensors)
            results["meta"]["n_features"] = len(fs_train.feature_cols)
            results["meta"]["n_val_engines"] = meta["n_val_engines"]

        feature_cols = fs_train.feature_cols
        X_train = fs_train.frame[feature_cols].to_numpy(dtype=float)
        y_train = fs_train.frame["RUL"].to_numpy(dtype=float)

        # Full val raw (with settings + sensors) for fault application
        val_raw = fs_val.frame.copy()
        X_val_clean = fs_val.frame[feature_cols].to_numpy(dtype=float)
        y_val_true = fs_val.frame["RUL"].to_numpy(dtype=float)

        # ---- fit models once on clean train ----
        fitted: dict[str, Any] = {}
        for name in STRESS_MODELS:
            est = build_model(name, seed=seed)
            est.fit(X_train, y_train)
            fitted[name] = est

        # ---- get fault scenarios ----
        scens = get_scenarios(cfg, used_sensors, sensor_std, offline=True)
        results["meta"]["n_scenarios"] = len(scens) + 1  # +1 for clean

        seed_result: dict[str, Any] = {"seed": seed, "scenarios": []}

        # ---- clean baseline ----
        clean_frame = fs_val.frame[["unit", "cycle", "RUL"]].copy()
        clean_metrics: dict[str, dict] = {}
        for name in STRESS_MODELS:
            clean_frame[name] = fitted[name].predict(X_val_clean)
            clean_metrics[name] = _scenario_metrics(clean_frame, "RUL", name)
        seed_result["scenarios"].append({
            "name": "clean",
            "kind": "none",
            "magnitude": 0.0,
            "source": "grid",
            "sensor": None,
            "metrics": clean_metrics,
        })

        # ---- fault scenarios ----
        for scen in scens:
            val_faulted = apply_scenario(val_raw, sensor_std, scen)
            fs_val_faulted = add_features(
                val_faulted,
                data_mod.SENSOR_COLS,
                window=window,
                constant=fs_train.dropped_sensors,
            )
            X_val_faulted = fs_val_faulted.frame[feature_cols].to_numpy(dtype=float)

            fault_frame = fs_val_faulted.frame[["unit", "cycle", "RUL"]].copy()
            scen_metrics: dict[str, dict] = {}
            for name in STRESS_MODELS:
                fault_frame[name] = fitted[name].predict(X_val_faulted)
                scen_metrics[name] = _scenario_metrics(fault_frame, "RUL", name)

            seed_result["scenarios"].append({
                "name": scen["name"],
                "kind": scen["kind"],
                "magnitude": scen["magnitude"],
                "source": scen["source"],
                "sensor": scen.get("sensor"),
                "metrics": scen_metrics,
            })

        results["per_seed"].append(seed_result)

    # ---- aggregate across seeds ----
    results["aggregate"] = _aggregate_across_seeds(results["per_seed"], STRESS_MODELS)

    # ---- save artifacts ----
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "phase4_stress.json").open("w", encoding="utf-8") as f:
        json.dump(_jsonable(results), f, indent=2)
    _save_csv(results, out / "phase4_stress.csv")

    return results


def _aggregate_across_seeds(per_seed: list[dict], models: list[str]) -> dict:
    """Mean ± std across seeds for each scenario × model metric."""
    # Collect scenario names in order (from first seed)
    scenario_names = [s["name"] for s in per_seed[0]["scenarios"]]
    aggregate: dict[str, Any] = {}

    for sname in scenario_names:
        s_agg: dict[str, Any] = {}
        for model in models:
            # Gather values across seeds
            metric_keys = [
                "engine_mean_mae", "engine_mean_rmse", "engine_mean_bias",
                "engine_worst_mae", "too_opt_engine_margin10",
                "pooled_mae", "pooled_rmse",
            ]
            model_agg: dict[str, Any] = {}
            for mk in metric_keys:
                vals = []
                for ps in per_seed:
                    for sc in ps["scenarios"]:
                        if sc["name"] == sname:
                            vals.append(sc["metrics"][model][mk])
                            break
                arr = np.asarray(vals, dtype=float)
                model_agg[mk] = {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}
            s_agg[model] = model_agg
        aggregate[sname] = s_agg

    return aggregate


def _save_csv(results: dict, path: Path) -> None:
    """Flatten results to CSV: one row per seed × scenario × model."""
    rows = []
    for ps in results["per_seed"]:
        seed = ps["seed"]
        for sc in ps["scenarios"]:
            for model, metrics in sc["metrics"].items():
                row = {
                    "seed": seed,
                    "scenario": sc["name"],
                    "kind": sc["kind"],
                    "magnitude": sc["magnitude"],
                    "source": sc["source"],
                    "sensor": sc.get("sensor"),
                    "model": model,
                }
                row.update({k: v for k, v in metrics.items()})
                rows.append(row)

    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(results: dict) -> None:
    """Print a compact summary of degradation vs clean baseline."""
    agg = results["aggregate"]
    models = results["meta"]["models"]
    seeds = results["meta"]["seeds"]

    print(f"\nPhase 4 Stress Test Summary")
    print(f"Seeds: {seeds}, Models: {models}, Scenarios: {results['meta']['n_scenarios']}")
    print(f"{'scenario':22} {'kind':10} {'mag':>6} " + " ".join(f"{m:>12}" for m in models))
    print("-" * (52 + 13 * len(models)))

    clean = agg["clean"]
    for sname, s_agg in agg.items():
        if sname == "clean":
            continue
        sc = s_agg  # just for typing
        kind = sc[models[0]]  # dummy to get structure
        # Find kind/magnitude from per_seed
        km = None
        for ps in results["per_seed"]:
            for sc2 in ps["scenarios"]:
                if sc2["name"] == sname:
                    km = (sc2["kind"], sc2["magnitude"])
                    break
            if km:
                break
        km = km or ("?", 0)

        line = f"{sname:22} {km[0]:10} {km[1]:>6.2f}"
        for model in models:
            mae_mean = sc[model]["engine_mean_mae"]["mean"]
            clean_mae = clean[model]["engine_mean_mae"]["mean"]
            delta = mae_mean - clean_mae
            line += f" {delta:+9.2f}"
        print(line)

    print("\nCost of safety (safe_side vs gradient_boosting, engine_mean_mae):")
    for sname in agg:
        if sname == "clean":
            continue
        safe = agg[sname]["safe_side"]["engine_mean_mae"]["mean"]
        gb = agg[sname]["gradient_boosting"]["engine_mean_mae"]["mean"]
        print(f"  {sname:22} safe_side={safe:7.2f}  gb={gb:7.2f}  cost={safe-gb:+7.2f}")


def main() -> None:
    cfg = load_config()
    raw_dir = cfg.get("data", {}).get("raw_dir", "data/raw")
    out_dir = cfg.get("outputs", {}).get("dir", "results")
    results = run_stress_test(cfg, raw_dir=raw_dir, out_dir=out_dir)
    print_summary(results)


if __name__ == "__main__":
    main()
