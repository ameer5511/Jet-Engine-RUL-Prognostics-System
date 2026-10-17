"""Nemotron scenario generation: propose fault settings, validate, cache.

Only ``scenarios.py`` talks to Nemotron (spec §6). Nemotron never sees the
data, labels, or scores. Every suggestion is strictly validated; invalid ones
are rejected. Validated suggestions form a UNION with the fixed grid, tagged
with provenance (grid vs nemotron). ``--offline`` runs grid-only.

The client is provider-agnostic: any OpenAI-compatible endpoint works
(NVIDIA build.nvidia.com today, Nebius Token Factory later) via env vars:

    NVIDIA_API_KEY      required
    NVIDIA_BASE_URL     defaults to https://integrate.api.nvidia.com/v1
    NEMOTRON_MODEL      defaults to nvidia/nemotron-3.5-lightning-30b-a3b
    NEMOTRON_MAX_CALLS  cap live calls per run to protect credits (default 5)

Scenario structure returned by :func:`get_scenarios`:

    [
        {
            "name": "noise_k0.5",      # stable scenario id
            "kind": "noise",
            "magnitude": 0.5,
            "sensor": None,             # None => applied to all sensors
            "source": "grid",           # "grid" or "nemotron"
            "settings": [FaultSetting, ...],  # one per sensor when sensor is None
        },
        ...
    ]
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from faults import (
    ACCEPTABLE,
    FaultConfig,
    FaultSetting,
    KINDS,
    NOISE_MAGNITUDES,
    DRIFT_MAGNITUDES,
    MISSING_MAGNITUDES,
    fixed_grid,
    training_sensor_std,
    validate_setting,
)

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
MAX_PROPOSED = 6  # ask Nemotron for at most this many extra settings


class NemotronUnavailable(RuntimeError):
    """Raised when Nemotron cannot be used and no fallback is acceptable."""


def _env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)


def nemotron_enabled(cfg: dict) -> bool:
    """Whether Nemotron scenario generation is switched on."""
    return bool(cfg.get("nemotron", {}).get("enabled", True))


def _api_credentials() -> tuple[str | None, str, str]:
    api_key = _env("NVIDIA_API_KEY")
    base_url = _env("NVIDIA_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    model = _env("NEMOTRON_MODEL", DEFAULT_MODEL)
    return api_key, base_url, model


def _json_from_text(text: str) -> list[dict[str, Any]]:
    """Extract a JSON array from model text; tolerate markdown fences."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\\s*```$", "", cleaned)
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start == -1 or end == -1 or end < start:
        return []
    snippet = cleaned[start : end + 1]
    try:
        data = json.loads(snippet)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def _prompt(used_sensors: list[str], seed_registry: list[int], n: int) -> str:
    kinds = ", ".join(f'"{k}"' for k in KINDS)
    sensors = ", ".join(f'"{s}"' for s in used_sensors)
    return (
        "You are helping design sensor-fault scenarios for a jet-engine "
        "Remaining Useful Life robustness study on NASA C-MAPSS FD001. "
        "Propose up to "
        + str(n)
        + " additional single-sensor fault settings as a JSON array. "
        "Each item must have exactly these fields: "
        '{"sensor": str, "kind": str, "magnitude": number, "seed": int}. '
        f"sensor: one of [{sensors}]. "
        f"kind: one of [{kinds}]. "
        "magnitude ranges: noise k in (0, 4] (sigma = k * std_train(sensor)); "
        "drift k in (0, 8] (total shift across engine = k * std_train(sensor)); "
        "missing f in (0, 1] (fraction of flights blanked then forward-filled). "
        f"seed: an integer in {seed_registry}. "
        "Rules: output ONLY a JSON array, no markdown fences, no explanation. "
        "Each sensor/kind/magnitude/seed combination at most once. "
        "Do not reference any engine data, labels, or scores. "
        "If you cannot produce valid JSON, output an empty array []."
    )


def _fetch_candidate_settings(
    used_sensors: list[str],
    seed_registry: list[int],
    max_calls: int,
    call_budget: dict[str, int],
) -> list[dict[str, Any]]:
    """Call Nemotron once and return raw proposal dicts (unvalidated)."""
    if call_budget["used"] >= max_calls:
        return []
    api_key, base_url, model = _api_credentials()
    if not api_key:
        return []
    call_budget["used"] += 1
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": _prompt(used_sensors, seed_registry, MAX_PROPOSED)}],
        "temperature": 0.2,
        "top_p": 0.95,
        "max_tokens": 256,
        "seed": int(seed_registry[0]) if seed_registry else 0,
        "stream": False,
    }
    try:
        resp = requests.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json=payload,
            timeout=60,
        )
    except requests.RequestException:
        return []
    if resp.status_code >= 400:
        return []
    try:
        body = resp.json()
    except ValueError:
        return []
    choices = body.get("choices") or []
    if not choices:
        return []
    message = choices[0].get("message") or {}
    content = message.get("content") or ""
    return _json_from_text(content)


def _validate_proposals(
    proposals: list[dict[str, Any]], used_sensors: list[str]
) -> list[FaultSetting]:
    """Strictly validate Nemotron proposals; reject anything invalid."""
    valid: list[FaultSetting] = []
    seen: set[tuple[str, str, float, int]] = set()
    for item in proposals:
        try:
            sensor = item["sensor"]
            kind = item["kind"]
            magnitude = float(item["magnitude"])
            seed = int(item["seed"])
        except (KeyError, TypeError, ValueError):
            continue
        try:
            validate_setting(sensor, kind, magnitude, used_sensors)
        except ValueError:
            continue
        key = (sensor, kind, magnitude, seed)
        if key in seen:
            continue
        seen.add(key)
        valid.append(FaultSetting(sensor=sensor, kind=kind, magnitude=magnitude, seed=seed))
    return valid


def _cache_path() -> Path:
    return Path("results") / "nemotron_scenarios.json"


def _load_cache() -> dict[str, Any] | None:
    path = _cache_path()
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def _save_cache(payload: dict[str, Any]) -> None:
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _scenario_key(kind: str, magnitude: float, sensor: str | None) -> tuple:
    return (kind, magnitude, sensor)


def _make_scenario(
    name: str,
    kind: str,
    magnitude: float,
    settings: list[FaultSetting],
    source: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "kind": kind,
        "magnitude": magnitude,
        "sensor": None,  # grid scenarios apply to all sensors
        "source": source,
        "settings": settings,
    }


def _nemotron_scenario(
    st: FaultSetting,
) -> dict[str, Any]:
    return {
        "name": f"nemotron_{st.kind}_k{st.magnitude}_{st.sensor}",
        "kind": st.kind,
        "magnitude": st.magnitude,
        "sensor": st.sensor,
        "source": "nemotron",
        "settings": [st],
    }


def get_scenarios(
    cfg: dict,
    used_sensors: list[str],
    sensor_std: dict[str, float],
    offline: bool = False,
) -> list[dict[str, Any]]:
    """Return a list of scenario dicts.

    Each scenario bundles the FaultSettings that must be applied together.
    Grid scenarios apply one (kind, magnitude) to *all* sensors. Nemotron
    scenarios are single-sensor proposals.

    Results are cached so reruns cost no credits. ``--offline`` skips Nemotron
    entirely (grid-only).
    """
    seed_registry = [int(s) for s in cfg.get("project", {}).get("seed_registry", [0])]
    sensors = list(used_sensors)

    # ---- fixed grid (always) ----
    scenarios: list[dict[str, Any]] = []
    seen: set[tuple[str, float, str | None]] = set()

    for config in fixed_grid(sensors):
        settings = config.expand(sensors, seed_base=0)
        name = f"{config.kind}_k{config.magnitude:g}"
        key = _scenario_key(config.kind, config.magnitude, None)
        if key in seen:
            continue
        seen.add(key)
        scenarios.append(_make_scenario(name, config.kind, config.magnitude, settings, "grid"))

    # ---- nemotron (optional) ----
    nemotron_cfg = cfg.get("nemotron", {})
    max_calls = int(nemotron_cfg.get("max_calls", _env("NEMOTRON_MAX_CALLS", "5")))
    use_cache = bool(nemotron_cfg.get("cache_file", True)) or _cache_path().exists()

    if nemotron_enabled(cfg) and not offline:
        api_key, _, _ = _api_credentials()
        if api_key:
            cached = _load_cache() if use_cache else None
            nemotron_settings: list[FaultSetting] = []
            if cached and cached.get("settings"):
                for item in cached["settings"]:
                    try:
                        validate_setting(
                            item["sensor"], item["kind"], float(item["magnitude"]), sensors
                        )
                        nemotron_settings.append(
                            FaultSetting(
                                sensor=item["sensor"],
                                kind=item["kind"],
                                magnitude=float(item["magnitude"]),
                                seed=int(item["seed"]),
                            )
                        )
                    except (KeyError, TypeError, ValueError):
                        continue
            else:
                call_budget = {"used": 0}
                proposals = _fetch_candidate_settings(
                    sensors, seed_registry, max_calls, call_budget
                )
                nemotron_settings = _validate_proposals(proposals, sensors)
                _save_cache(
                    {
                        "schema_version": 1,
                        "model": _env("NEMOTRON_MODEL", DEFAULT_MODEL),
                        "base_url": _env("NVIDIA_BASE_URL", DEFAULT_BASE_URL),
                        "settings": [
                            {
                                "sensor": st.sensor,
                                "kind": st.kind,
                                "magnitude": st.magnitude,
                                "seed": st.seed,
                            }
                            for st in nemotron_settings
                        ],
                    }
                )

            for st in nemotron_settings:
                key = _scenario_key(st.kind, st.magnitude, st.sensor)
                if key in seen:
                    continue
                seen.add(key)
                scenarios.append(_nemotron_scenario(st))

    return scenarios


def apply_scenario(
    raw: pd.DataFrame,
    sensor_std: dict[str, float],
    scenario: dict[str, Any],
) -> pd.DataFrame:
    """Apply one scenario's fault settings to raw held-out data. Returns a copy."""
    from faults import apply_settings

    return apply_settings(raw, sensor_std, scenario["settings"])
