"""Integration test for train.py on the real FD001 validation split."""
from __future__ import annotations

import pytest


def test_phase2_pipeline_beats_constant(have_real_data, raw_dir):
    if not have_real_data:
        pytest.skip("real FD001 data not present")

    import train
    from config import load_config

    cfg = load_config()
    cfg["project"]["seed_registry"] = [0]  # one seed keeps the test fast

    results = train.run_phase2(cfg, raw_dir=raw_dir)
    per_model = results["per_model"]

    constant_mae = per_model["constant"]["aggregate"]["engine_mean_mae"]["mean"]
    for name in ("random_forest", "gradient_boosting"):
        assert per_model[name]["aggregate"]["engine_mean_mae"]["mean"] < constant_mae

    assert "coverage_80" in results["coverage"]["aggregate"]
