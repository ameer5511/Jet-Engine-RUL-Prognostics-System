# Jet Engine RUL Prognostics System

**License: MIT** (see [LICENSE](LICENSE))

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-94%20passing-brightgreen.svg)](#running-the-tests)

> **Disclaimer.** This project uses NASA's C-MAPSS **simulation** data, not real airline
> or engine data. It is a research and portfolio project and must **not** be used for
> real maintenance or airworthiness decisions.

The Jet Engine RUL Prognostics System predicts the Remaining Useful Life (RUL) of a
simulated jet engine — how many more flights it can fly before maintenance — and then
stress-tests that prediction: **what happens when the sensors go bad?**

- **Live dashboard:** [GitHub repository](https://github.com/ameer5511/Jet-Engine-RUL-Prognostics-System) · deployed with Streamlit Community Cloud. *(Add your `https://<app>.streamlit.app` link here.)*

---

## Tech stack

| Layer | Technology | Role |
| --- | --- | --- |
| Language | Python 3.11+ | Core pipeline and dashboard |
| Data | pandas, NumPy | Loading C-MAPSS, time-series features, evaluation |
| ML | scikit-learn | Random Forest, quantile Gradient Boosting (α = 0.30 / 0.10 / 0.90) |
| AI | **NVIDIA Nemotron-3.5-Lightning** (`nvidia/nemotron-3.5-lightning-30b-a3b`) | Proposes sensor-fault scenarios for the Robustness Lab via NVIDIA's free hosted endpoint (`https://integrate.api.nvidia.com/v1`, build.nvidia.com) |
| Dashboard | Streamlit + Plotly | Four read-only pages rendering pre-computed results |
| Config | PyYAML + python-dotenv | All knobs in `config.yaml`; secrets only in `.env` |
| Quality | pytest | 94 tests: data splits, metrics, faults, scenarios, dashboard loaders |
| Hosting | Streamlit Community Cloud | Auto-redeploys on every push to GitHub |

**Order of operations:** configs from `config.yaml` → artifact generation offline
(`train.py`, `stress_test.py`, `test_run.py`) → the dashboard only ever *reads*
committed artifacts. No model trains at request time, no AI call happens on page load.

---

## The question

Maintenance too early wastes money; maintenance too late risks failure. The dangerous
error is predicting **too many** flights remaining. This project measures both the
*robustness* of RUL models to sensor faults and the *cost of safety* — the extra early
maintenance a cautious model asks for.

---

## What we build

1. **A predictor** — a constant baseline, Random Forest, gradient boosting, and a
   conservative "safe-side" quantile gradient-boosting model (α = 0.30), plus P10/P90
   interval models (α = 0.10 / 0.90).
2. **A Robustness Lab** — 100 test train engines are damaged three ways (noise, drift,
   missing readings) at four dataset-relative magnitudes each — 12 fixed grid scenarios
   — and prediction degradation is measured over 5 seeds.
3. **A dashboard** — Streamlit + Plotly on Streamlit Community Cloud, reading
   pre-computed artifacts only.

### Where NVIDIA fits (a required dependency, not a gimmick)

NVIDIA **Nemotron-3.5-Lightning** (`nvidia/nemotron-3.5-lightning-30b-a3b`, called
through the [build.nvidia.com](https://build.nvidia.com) hosted endpoint, keyed by
`NVIDIA_API_KEY`) has exactly **one job**: it *proposes* additional single-sensor fault
settings to stress. The pipeline:

1. Calls the live endpoint once per run (`scenarios.py`, ≤ 5 calls, capped by `NEMOTRON_MAX_CALLS`);
2. **Validates** every proposal against a strict schema — sensor whitelist, fault-kind
   whitelist, per-kind magnitude range, integer seed in the registry, dedupe;
3. Unions the validated suggestions with the fixed 12-scenario grid, tagged with
   provenance (*grid* vs *nemotron*);
4. **Caches** validated replies to `results/nemotron_scenarios.json`, so reruns are
   deterministic and cost nothing.

The live call was executed and verified; 6 validated settings are cached and committed.
Nemotron never sees engine data, labels, or scores — it only suggests test scenarios.

---

## Results (committed artifacts)

All numbers below are computed by `evaluate.py` from artifacts committed under
`results/` — nothing is hand-written. Validation uses a 20-engine seeded holdout from
the 100 FD001 training engines; metrics are computed against uncapped true RUL.

### Validation (20 engines, seed 0)

| Model | Engine-mean MAE | Mean bias | Too-optimistic engines (bias > +10) |
| --- | --- | --- | --- |
| Constant (baseline) | 55.2 | +2.9 | 50% |
| Random Forest | 27.6 | −5.2 | 30% |
| Gradient Boosting | 27.6 | −3.6 | 35% |
| Safe-side (α = 0.30) | 30.7 | −23.3 | **10%** |

**Cost of safety** (same definition everywhere: pooled safe-side MAE − gradient-boosting
MAE, on the same validation engines): **+4.9 flights** — the safe-side model trims
too-optimistic engines from 35% to 10% at a price of ≈ 5 flights of earlier maintenance.

### Locked official test run (never used before this run)

Gradient Boosting trained on the **full** 100-engine training set, scored on the official
FD001 test engines (100 units) **at each engine's last cycle only**:

| Model | NASA score (last cycle) | MAE |
| --- | --- | --- |
| Gradient Boosting | 63,259.2 | 23.7 |

### Stress test (clean + 12 grid scenarios × 5 seeds)

- Gradient Boosting degrades badly under **noise**: at noise k = 2σ its engine-mean MAE
  rises ≈ +25 flights versus the clean baseline.
- Under **drift** GB actually *improves* (tree models absorb smooth systematic shifts),
  while the safe-side model pays more there.
- **Missing readings** have minimal impact (forward-fill handles it).
- The safe-side model stays visibly conservative (largest negative bias) under almost
  all faults — safety persists, at a measured cost.

---

## Metrics, exactly

- **Error convention:** `d = prediction − truth`. Positive `d` = predicted too many
  flights left (optimistic / dangerous).
- **MAE / RMSE** — computed per engine, then averaged over engines (per-engine primary,
  per-flight secondary).
- **NASA score** (Saxena & Goebel, PHM 2008), at each engine's **last cycle**:
  `d < 0 → exp(−d/13) − 1` (under-prediction, less steep);
  `d ≥ 0 → exp(d/10) − 1` (over-prediction, **steeper** — the dangerous branch).
- **Too-optimistic rate** — fraction of rows/engines with `d > +10` flights (margin fixed
  before runs; 5/10/20 sweep reported in artifacts).
- **Cost of safety** — safe-side MAE − Gradient-Boosting MAE on the same engines.
- **Range calibration** — P10–P90 empirical coverage on validation; quantiles are sorted
  so Q10 ≤ prediction ≤ Q90 at inference (crossing eliminated at render time).
- **Spread** — mean ± standard deviation across the 5 seeds.

---

## The dashboard (4 read-only pages)

| Page | What it shows |
| --- | --- |
| **Overview** | Landing page, headline result, locked official test result, links, disclaimer |
| **Engine View** | Pick a validation engine: predicted vs true RUL by flight, GB headline, safe-side comparison, P10–P90 band, per-engine metrics |
| **Model Comparison** | All models side by side: MAE, bias, NASA scores, split tables, cost of safety |
| **Stress Lab** | Degradation per fault kind and magnitude, diverging colour heatmap (red = worse), cost of safety per fault, "grid + Nemotron" provenance and seed notes |

Every page displays the disclaimer: *"NASA C-MAPSS simulation data only. Not for real
maintenance or airworthiness decisions."* The sidebar shows artifact status, the
Nemotron scenario count, and a GitHub link.

---

## Quickstart

### 1. Environment

```bash
conda create -n rul python=3.11 -y
conda activate rul
pip install -r requirements.txt
```

### 2. Data

Download NASA C-MAPSS **FD001** and place the raw files under `data/raw/`
(git-ignored):

- `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`, `readme.txt`
- Source: NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation), FD001.
- Verified format: whitespace-separated, no header; 26 columns — unit, cycle,
  3 operational settings, 21 sensor readings; 100 train engines (run to failure),
  100 truncated test engines.

### 3. NVIDIA key (for the Robustness Lab)

```bash
cp .env.example .env       # then edit .env — add your build.nvidia.com key
```

### 4. Generate artifacts

```bash
python train.py        # trains all models on the train split, evaluates on validation
python stress_test.py  # all models × 12 grid scenarios × 5 seeds
python test_run.py     # single locked run on the official test set (last cycle)
```

### 5. Run the dashboard

```bash
streamlit run app.py   # → http://localhost:8501
```

---

## Repository layout

```
.
├── app.py                    # Streamlit shell: navigation, sidebar, global styles
├── views/                    # dashboard pages
│   ├── 0_Overview.py
│   ├── 1_Engine_View.py
│   ├── 2_Model_Comparison.py
│   └── 3_Stress_Lab.py
├── styles/                   # design tokens + Plotly template
├── dashboard_data.py         # read-only artifact loaders (Streamlit-cached)
├── config.yaml               # all tunable knobs (paths, seeds, faults, metrics)
├── data.py                   # FD001 loading, RUL labels, engine-level split
├── features.py               # past-only mean/trend features (window resets per engine)
├── models.py                 # constant / RF / GB / quantile models
├── faults.py                 # noise, drift, missing-readings application
├── scenarios.py              # NVIDIA Nemotron client: prompt → validate → cache
├── train.py                  # phase 2: fit models, evaluate on validation
├── stress_test.py            # phase 4: models × scenarios × seeds
├── test_run.py               # locked official test-set run (last-cycle scoring)
├── evaluate.py               # metrics incl. NASA score at last cycle
├── results/                  # committed artifacts (see below)
├── tests/                    # 94 tests — one suite per module
├── docs/                     # design + UI documentation
├── MASTER_SPEC.md            # source of truth: design, decisions, acceptance
├── requirements.txt          # Python dependencies
├── .env.example              # NVIDIA endpoint template (never commit .env)
└── LICENSE                   # MIT
```

### Committed artifacts (`results/`)

| File | Contents |
| --- | --- |
| `phase2_val_predictions.csv` | Per-cycle validation predictions, all models |
| `phase2_metrics.json` | Validation metrics aggregate (+ resolved config snapshot) |
| `phase4_stress.json` / `.csv` | Stress test: clean + 12 scenarios × models × 5 seeds |
| `phase3_test_results.json` | Locked official test result (last-cycle NASA score) |
| `nemotron_scenarios.json` | Live-validated Nemotron fault proposals (6 settings) |

Raw training data (`data/raw/`) and your `.env` are git-ignored.

---

## Reproducibility & safety

- All seeds live in `config.yaml` and every run logs the resolved config
  (`results/phase2_resolved_config.json`).
- Train/validation split is by **whole engines** (never by row); a test asserts the
  engine-id sets are disjoint.
- Features are **past-only** with rolling windows reset at each engine boundary.
- Faults apply **only to held-out data**, deterministically per seed; official test
  engines are touched exactly once (`test_run.py`).
- Training targets are **uncapped RUL** (the historical cap of 125 is retained for
  reference only), so metrics score against the full true range.
- No maintenance or airworthiness advice anywhere — README, dashboard header and pages
  all carry the disclaimer.

---

## Running the tests

```bash
python -m pytest tests/ -q
```

94 tests cover data splitting, features, models, faults, Nemotron validation/caching,
metrics, evaluation, and dashboard loaders. Tests that need the real FD001 files skip
automatically when data is absent, so the suite always runs.

---

## Status

| Phase | Description | State |
| --- | --- | --- |
| 0 | Setup & spec | ✅ Done |
| 1 | Data & features | ✅ Done |
| 2 | Models & baselines | ✅ Done |
| 3 | Robustness Lab | ✅ Done |
| 4 | Stress test & evaluation | ✅ Done |
| 5 | Dashboard & demo | ✅ Done — deployed on Streamlit Community Cloud |
| 6 | Polish, tests, submission | ✅ Done |

---

## Disclaimer

This project uses **simulated** NASA C-MAPSS data. It is a research and portfolio
project. It must **not** be used for real maintenance scheduling, airworthiness, or
operational decisions of any kind.

## License

MIT — see [LICENSE](LICENSE).
