"""Tests for config.py: YAML loading and missing-file handling."""
from __future__ import annotations

import pytest

from config import load_config


def test_default_config_loads_and_has_expected_keys():
    cfg = load_config()
    assert cfg["data"]["dataset"] == "FD001"
    assert cfg["labels"]["cap"] == 125
    assert cfg["split"]["by"] == "engine"
    assert cfg["features"]["window"] == 10
    assert cfg["faults"]["noise"]["kind"] if "kind" in cfg["faults"]["noise"] else True
    assert len(cfg["project"]["seed_registry"]) == 5


def test_missing_config_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "does_not_exist.yaml")
