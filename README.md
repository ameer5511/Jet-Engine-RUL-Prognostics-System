# Jet Engine RUL Prognostics System

**License: MIT** (see [LICENSE](LICENSE))

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Disclaimer.** AeroGuard uses NASA's C-MAPSS **simulation** data, not real airline
> or engine data. This is a research and portfolio project and must **not** be used
> for real maintenance or airworthiness decisions.

AeroGuard predicts the Remaining Useful Life (RUL) of a simulated jet engine — how
many more flights it can fly before maintenance — and then stress-tests that
prediction: **what happens when the sensors go bad?**

## The question

Maintenance too early wastes money; maintenance too late risks failure. The
dangerous error is predicting **too many** flights remaining. AeroGuard measures
both the *robustness* of RUL models to sensor faults and the *cost of safety*
(the extra early maintenance a cautious model asks for).

## What we build

1. **A predictor** — a constant baseline, Random Forest, gradient boosting, and a
   conservative "safe-side" quantile gradient-boosting model (with P10/P90 ranges).
2. **A Robustness Lab** — sensor data is damaged three ways (noise, drift, missing
   readings) at four magnitudes each, and prediction degradation is measured.
3. **A dashboard** — Streamlit + Plotly, hosted on Nebius, reading pre-computed
   results only (nothing trains at request time).

NVIDIA **Nemotron 3.5 Lightning 30B-A3B** (free hosted endpoint,
`https://integrate.api.nvidia.com/v1`, keyed by `NVIDIA_API_KEY`) has exactly one
job: it *proposes* fault settings to test. Our code validates every suggestion
against a strict schema and computes every number. Nemotron is optional — the
pipeline runs fully offline with a fixed fault grid if Nemotron is unavailable,
and it can be pointed at Nebius Token Factory instead via `NVIDIA_BASE_URL`.

> **Disclaimer.** All data is NASA's C-MAPSS simulation, not real engine data.
> Research/portfolio use only — not for real maintenance or airworthiness decisions.

## Status

| Phase | Description | State |
| --- | --- | --- |
| 0 | Setup & spec | Done |
| 1 | Data & features | Done |
| 2 | Models & baselines | Done |
| 3 | Robustness Lab | Done |
| 4 | Stress test & evaluation | Done |
| 5 | Dashboard & demo | Not started |
| 6 | Polish, tests, submission | Not started |

See `MASTER_SPEC.md` for the full design, decisions, and acceptance criteria.

## Repository layout

```
.
├── MASTER_SPEC.md        # source of truth (design + decisions + phases)
├── config.yaml           # all tunable knobs (paths, seeds, faults, metrics)
├── requirements.txt      # Python dependencies
├── .env.example          # template for secrets (never commit .env)
├── LICENSE               # MIT
├── data/                 # NASA C-MAPSS FD001 (git-ignored, see below)
├── results/              # generated artifacts (git-ignored)
├── tests/                # one small test per module
└── (data.py, features.py, models.py, faults.py, scenarios.py,
     train.py, evaluate.py, stress_test.py, dashboard.py)
```

## Environment setup

The project targets **Python 3.11 or newer**. Your Mac's system Python is 3.9 and
should not be used for this project. Create a dedicated environment:

```bash
# Option A — conda (available on this machine)
conda create -n aeroguard python=3.11 -y
conda activate aeroguard

# Option B — Homebrew Python + venv
brew install python@3.12
python3.12 -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt
```

Exact dependency versions will be locked with `pip freeze > requirements.lock.txt`
once the environment exists.

### Running the tests

```bash
conda run -n ml python -m pytest tests/ -q
```

Tests that need the real FD001 files skip automatically when the data is absent,
so the suite always runs.

## Data

Download NASA C-MAPSS **FD001** (train/test time series and RUL files) and place the
raw files under `data/raw/`. The raw data is not committed to this repository.

- Dataset: NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation).
- Attribution and exact filenames/columns: to be confirmed against the dataset
  readme before Phase 1 (tracked as Open Question B1 in `MASTER_SPEC.md`).

## Using the NVIDIA model (Nemotron via Nebius Token Factory)

Copy the environment template and fill in your credentials:

```bash
cp .env.example .env      # then edit .env
```

Nemotron is only called by `scenarios.py`. Validated suggestions are cached to
`results/`, so reruns are deterministic and cost nothing. Use `--offline` to run
grid-only.

### Labels & metrics

Training targets use **uncapped RUL** so predictions span the full true range,
matching metrics which score against uncapped truth. The historical C-MAPSS cap
of 125 is retained for reference only (it previously caused systematic
under-prediction and biased coverage numbers).
 Exact model name, endpoint, and request format must be confirmed in the
Token Factory console (Open Question B5 in `MASTER_SPEC.md`).

## Disclaimer

This project uses **simulated** NASA C-MAPSS data. It is a research and portfolio
project. It must **not** be used for real maintenance scheduling, airworthiness, or
operational decisions of any kind.
