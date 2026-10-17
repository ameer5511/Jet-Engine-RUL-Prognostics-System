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


def test_offline_returns_grid_only(monkeypatch, sensors, cfg, std):
    _clear_cache()
    called = {"n": 0}

    def fake(sensors, seed_registry, max_calls, call_budget):
        called["n"] += 1
        return []

    monkeypatch.setattr(scenarios, "_fetch_candidate_settings", fake)

    settings, prov = scenarios.get_scenarios(cfg, sensors, std, offline=True)
    assert called["n"] == 0
    assert all(p["source"] == "grid" for p in prov)
    assert len(settings) == len(fixed_grid(sensors)) * len(sensors)
    assert len(settings) == len(prov)


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

    settings, prov = scenarios.get_scenarios(cfg, sensors, std, offline=False)
    nemotron = [p for p in prov if p["source"] == "nemotron"]
    assert len(nemotron) == 2
    assert {
        (p["sensor"], p["kind"], p["magnitude"], p["seed"]) for p in nemotron
    } == {("s1", "noise", 0.75, 1), ("s2", "drift", 1.5, 2)}
    assert all(p["scope"] == "single_sensor" for p in nemotron)
    assert any(p["source"] == "grid" for p in prov)
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

    settings, prov = scenarios.get_scenarios(cfg, sensors, std, offline=False)
    nemotron = [p for p in prov if p["source"] == "nemotron"]
    assert len(nemotron) == 1
    assert nemotron[0] == {
        "sensor": "s1",
        "kind": "noise",
        "magnitude": 0.5,
        "seed": 1,
        "source": "nemotron",
        "scope": "single_sensor",
    }
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

    settings, prov = scenarios.get_scenarios(cfg, sensors, std, offline=False)
    nemotron = [p for p in prov if p["source"] == "nemotron"]
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

        settings, prov = scenarios.get_scenarios(cfg, sensors, std, offline=False)
        assert called["n"] == 0
        nemotron = [p for p in prov if p["source"] == "nemotron"]
        assert len(nemotron) == 1
        assert nemotron[0]["magnitude"] == 0.9
        assert nemotron[0]["seed"] == 7
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

    settings, prov = scenarios.get_scenarios(cfg, sensors, std, offline=False)
    assert called["n"] == 0
    assert all(p["source"] == "grid" for p in prov)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)


def test_apply_plan_applies_settings(std):
    rows = []
    for unit, base in ((1, 10.0), (2, 50.0)):
        for i in range(1, 5):
            rows.append({"unit": unit, "cycle": i, "s1": base + i, "s2": 3.0})
    raw = pd.DataFrame(rows)
    settings = [FaultSetting("s1", "drift", 1.0, 0)]
    faulted = scenarios.apply_plan(raw, std, settings)
    assert not faulted["s1"].equals(raw["s1"])
    assert faulted["s2"].equals(raw["s2"])


def test_fixed_grid_and_config_roundtrip(sensors):
    configs = fixed_grid(sensors)
    assert len(configs) == 12
    total = sum(len(c.expand(sensors)) for c in configs)
    assert total == 12 * len(sensors)
    assert FaultConfig("noise", 0.25).expand(sensors) == FaultConfig("noise", 0.25).expand(sensors)
