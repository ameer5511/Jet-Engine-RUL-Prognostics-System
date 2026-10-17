"""Tests for scenarios.py: Nemotron validation, union, cache, offline."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import scenarios
from faults import FaultConfig, FaultSetting, fixed_grid


@pytest.fixture
def sensors():
    return ["s1", "s2", "s3"]


@pytest.fixture
def cfg():
    return {
        "project": {"seed_registry": [0, 1, 2, 3, 4]},
        "nemotron": {"enabled": True, "max_calls": 5},
    }


@pytest.fixture
def std():
    return {"s1": 2.0, "s2": 3.0, "s3": 4.0}


def _clear_cache():
    p = Path("results/nemotron_scenarios.json")
    if p.exists():
        p.unlink()


def _monkeypatch_fetch(monkeypatch, proposals):
    def fake(sensors, seed_registry, max_calls, call_budget):
        return proposals

    monkeypatch.setattr(scenarios, "_fetch_candidate_settings", fake)
    monkeypatch.setattr(scenarios, "_save_cache", lambda payload: None)


def _scenario_names(scenarios_list):
    return {s["name"] for s in scenarios_list}


def test_offline_returns_grid_only(monkeypatch, sensors, cfg, std):
    _clear_cache()
    called = {"n": 0}

    def fake(sensors, seed_registry, max_calls, call_budget):
        called["n"] += 1
        return []
    monkeypatch.setattr(scenarios, "_fetch_candidate_settings", fake)

    scens = scenarios.get_scenarios(cfg, sensors, std, offline=True)
    assert called["n"] == 0
    assert all(s["source"] == "grid" for s in scens)
    # 12 configs (3 kinds x 4 magnitudes), each applied to all sensors
    assert len(scens) == 12
    names = _scenario_names(scens)
    assert "noise_k0.25" in names
    assert "noise_k2" in names
    assert "drift_k0.5" in names
    assert "drift_k4" in names
    assert "missing_k0.05" in names
    assert "missing_k0.4" in names


def test_grid_scenarios_apply_to_all_sensors(sensors, cfg, std):
    scens = scenarios.get_scenarios(cfg, sensors, std, offline=True)
    noise = [s for s in scens if s["name"] == "noise_k0.5"][0]
    assert noise["kind"] == "noise"
    assert noise["magnitude"] == 0.5
    assert noise["sensor"] is None  # all sensors
    assert len(noise["settings"]) == len(sensors)
    assert {st.sensor for st in noise["settings"]} == set(sensors)


def test_nemotron_valid_proposals_added(monkeypatch, sensors, cfg, std):
    _clear_cache()
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key-12345")
    _monkeypatch_fetch(
        monkeypatch,
        [
            {"sensor": "s1", "kind": "noise", "magnitude": 0.75, "seed": 1},
            {"sensor": "s2", "kind": "drift", "magnitude": 1.5, "seed": 2},
        ],
    )

    scens = scenarios.get_scenarios(cfg, sensors, std, offline=False)
    nemotron = [s for s in scens if s["source"] == "nemotron"]
    assert len(nemotron) == 2
    names = {s["name"] for s in nemotron}
    assert "nemotron_noise_k0.75_s1" in names
    assert "nemotron_drift_k1.5_s2" in names
    for s in nemotron:
        assert s["sensor"] is not None  # single sensor
        assert len(s["settings"]) == 1
    assert any(s["source"] == "grid" for s in scens)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)


def test_nemotron_invalid_proposals_rejected(monkeypatch, sensors, cfg, std):
    _clear_cache()
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key-12345")
    _monkeypatch_fetch(
        monkeypatch,
        [
            {"sensor": "s1", "kind": "noise", "magnitude": 99.0, "seed": 1},
            {"sensor": "s9", "kind": "noise", "magnitude": 0.5, "seed": 1},
            {"sensor": "s1", "kind": "bogus", "magnitude": 0.5, "seed": 1},
            {"sensor": "s1", "kind": "noise", "magnitude": 0.5, "seed": 1},
        ],
    )

    scens = scenarios.get_scenarios(cfg, sensors, std, offline=False)
    nemotron = [s for s in scens if s["source"] == "nemotron"]
    assert len(nemotron) == 1
    assert nemotron[0]["name"] == "nemotron_noise_k0.5_s1"
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)


def test_nemotron_duplicate_proposals_deduped(monkeypatch, sensors, cfg, std):
    _clear_cache()
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key-12345")
    _monkeypatch_fetch(
        monkeypatch,
        [
            {"sensor": "s1", "kind": "noise", "magnitude": 0.5, "seed": 5},
            {"sensor": "s1", "kind": "noise", "magnitude": 0.5, "seed": 5},
        ],
    )

    scens = scenarios.get_scenarios(cfg, sensors, std, offline=False)
    nemotron = [s for s in scens if s["source"] == "nemotron"]
    assert len(nemotron) == 1
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)


def test_json_from_text_tolerates_fences():
    text = '```json\n[{"sensor":"s1","kind":"noise","magnitude":0.5,"seed":1}]\n```'
    assert scenarios._json_from_text(text) == [
        {"sensor": "s1", "kind": "noise", "magnitude": 0.5, "seed": 1}
    ]
    assert scenarios._json_from_text("not json at all") == []
    assert scenarios._json_from_text("{\"a\": 1}") == []


def test_cache_skips_http(monkeypatch, sensors, cfg, std):
    _clear_cache()
    cache = {
        "schema_version": 1,
        "model": "nvidia/nemotron-3.5-lightning-30b-a3b",
        "settings": [
            {"sensor": "s1", "kind": "noise", "magnitude": 0.9, "seed": 7},
        ],
    }
    Path("results").mkdir(parents=True, exist_ok=True)
    with Path("results/nemotron_scenarios.json").open("w", encoding="utf-8") as f:
        json.dump(cache, f)
    try:
        monkeypatch.setenv("NVIDIA_API_KEY", "test-key-12345")
        called = {"n": 0}

        def fake(sensors, seed_registry, max_calls, call_budget):
            called["n"] += 1
            raise AssertionError("fetch should not be called when cache hits")

        monkeypatch.setattr(scenarios, "_fetch_candidate_settings", fake)

        scens = scenarios.get_scenarios(cfg, sensors, std, offline=False)
        assert called["n"] == 0
        nemotron = [s for s in scens if s["source"] == "nemotron"]
        assert len(nemotron) == 1
        assert nemotron[0]["magnitude"] == 0.9
        assert nemotron[0]["settings"][0].seed == 7
    finally:
        _clear_cache()
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)


def test_no_api_key_means_grid_only(monkeypatch, sensors, cfg, std):
    _clear_cache()
    monkeypatch.setenv("NVIDIA_API_KEY", "")
    called = {"n": 0}

    def fake(sensors, seed_registry, max_calls, call_budget):
        called["n"] += 1
        return []
    monkeypatch.setattr(scenarios, "_fetch_candidate_settings", fake)

    scens = scenarios.get_scenarios(cfg, sensors, std, offline=False)
    assert called["n"] == 0
    assert all(s["source"] == "grid" for s in scens)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)


def test_apply_scenario_applies_settings(std):
    rows = []
    for unit, base in ((1, 10.0), (2, 50.0)):
        for i in range(1, 5):
            rows.append({"unit": unit, "cycle": i, "s1": base + i, "s2": 3.0})
    raw = pd.DataFrame(rows)
    scenario = {
        "name": "test_drift",
        "kind": "drift",
        "magnitude": 1.0,
        "sensor": None,
        "source": "grid",
        "settings": [FaultSetting("s1", "drift", 1.0, 0)],
    }
    faulted = scenarios.apply_scenario(raw, std, scenario)
    assert not faulted["s1"].equals(raw["s1"])
    assert faulted["s2"].equals(raw["s2"])


def test_fixed_grid_and_config_roundtrip(sensors):
    configs = fixed_grid(sensors)
    assert len(configs) == 12
    assert FaultConfig("noise", 0.25).expand(sensors) == FaultConfig("noise", 0.25).expand(sensors)
