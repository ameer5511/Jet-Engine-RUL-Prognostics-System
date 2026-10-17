# AeroGuard v2 — MASTER SPEC

**Status:** Draft for review
**Date:** 17 Oct 2026
**Owner:** Ameer Muhammed
**Source of truth:** This spec consolidates *AeroGuard v2: Project Plan and Design* plus the structured review. Where the design doc and this spec disagree, resolve the conflict, then update the design doc.

> **Disclaimer.** All data is NASA's C-MAPSS simulation, not real airline/engine data. This is a research and portfolio project and must **not** be used for real maintenance or airworthiness decisions.

---

## 1. What we build

A RUL (Remaining Useful Life) pipeline on NASA C-MAPSS **FD001** with three parts:

1. **Predictor.** A constant baseline, Random Forest, gradient boosting (main), and a "safe-side" quantile gradient-boosting model that leans cautious, plus P10/P90 range models.
2. **Robustness Lab.** Damage held-out sensor data three ways (noise, drift, missing readings) at four magnitudes each, and measure how predictions degrade.
3. **Dashboard.** A Streamlit + Plotly app hosted on Nebius that reads saved artifacts only (no training at request time).

**Headline question:** When sensors go bad, does the safe-side model stay safe, and how many flights of early maintenance does that safety cost?

**NVIDIA dependency:** Nemotron Nano (via Nebius Token Factory) proposes candidate fault settings only. Our code validates every suggestion and produces every number.

**Out of scope for the hackathon:** FD002–FD004, N-CMAPSS, neural nets (GRU/LSTM), custom NASA-score loss, anomaly alarms, drift monitoring, chatbot, own API, database, accounts, 3D plots, real hardware.

---

## 2. Scope and cut order

**In scope**
- FD001 only.
- Labels + engine-level split.
- Four models: constant, RF, GB, safe-side quantile GB (+ P10/P90).
- Metrics: MAE (main), RMSE, NASA score, too-optimistic rate.
- Robustness Lab: 3 fault types × 4 magnitudes, fixed seeds.
- Nemotron Nano proposing extra fault settings, validated and saved.
- Streamlit dashboard on Nebius: engine view, model comparison, stress-test charts.
- Public GitHub repo, MIT licence, README, <3 min video, written feedback.

**Cut order if time runs short**
1. Nemotron suggestions → fixed grid only, keep a small *checked* notes-writer so an NVIDIA model is still used.
2. Dashboard polish.
3. One fault type.

**Never cut:** the safe-side vs normal comparison.

---

## 3. Data

- NASA C-MAPSS **FD001**: 100 train engines, 100 test engines (each run to failure in train; test units truncated). Each row = one engine on one flight.
- Columns: unit/engine id, time (cycle), 3 operational settings, 21 sensor readings. **Exact column names must be confirmed against the dataset readme (Open Q-B1).**
- Raw data lives in `data/` and is **git-ignored**.
- Readme/attribution for C-MAPSS to be included in `README.md`.

### Labels
- RUL = flights remaining until failure for that engine.
- **Cap at 125** for training labels (healthy engines look alike in the early region).
- **DECIDED (A1):** training labels are capped at 125; all reported metrics are computed against the **uncapped** true RUL, so the cap never flatters the score. Evaluation uses the official true-RUL values (uncapped).
- Cap value is a hyperparameter chosen on validation data only.

### Split
- Split by **whole engines**, never by row.
- ~80 train engines / ~20 validation engines from the 100 training engines; deterministic by seed.
- Official 100 test engines **untouched until the final run** (guarded — Open Q-A5).
- Enforced by a test asserting the engine-id sets are disjoint.

### Features
- Per sensor: current value, mean of last 10 flights, trend (change over last 10 flights).
- **Past-only**; rolling window **resets at each engine boundary**.
- **DECIDED (A4):** partial windows at the start of an engine use only the flights available so far (no padding, no peeking); trend is undefined → set to 0 when fewer than 2 flights exist.
- Drop sensors that never change.
- Expected ~40 inputs; exact count confirmed in code.
- No scaling is required for tree models; if scaling is introduced it must be fit on train only.

---

## 4. Models

| Model | Purpose |
| --- | --- |
| Constant | Floor: always predicts the mean train RUL. Every real model must beat it. |
| Random Forest | Strong, low tuning. |
| Gradient Boosting | Main predictor. |
| Safe-side quantile GB (α≈0.3) | Leans cautious; the comparison against GB is never cut. |
| Quantile GB α=0.10 and α=0.90 | Low/high range for each prediction. |

- α=0.3 is a **training target**, not a guarantee on new data.
- **Cost of safety** is the price of the cautious model, measured explicitly.
- Baseline gate: all models must beat the constant baseline on validation MAE before the final test run.

---

## 5. Robustness Lab

- Faults applied **only to held-out data**, never training.
- A fault setting = 4 fields: `{sensor, kind, magnitude, seed}`.
- Kinds: **noise** (additive), **drift** (slow shift over the engine's flights), **missing** (blank + forward-fill).
- Four magnitudes per kind, fixed seeds.

**DECIDED (A3) — magnitudes are defined in dataset-relative units so they are portable and reproducible:**

| Kind | Formula | Magnitudes (defaults) |
| --- | --- | --- |
| noise | add `N(0, σ)` where `σ = k × std_train(sensor)` | k ∈ {0.25, 0.50, 1.00, 2.00} |
| drift | add `k × std_train(sensor) × (t / T_engine)` to flight `t` (sensor shifts by k stds across the engine's sequence) | k ∈ {0.50, 1.00, 2.00, 4.00} |
| missing | randomly blank a fraction `f` of that engine's flights for the sensor, then forward-fill (first-flight gaps keep the original value) | f ∈ {0.05, 0.10, 0.20, 0.40} |

- `std_train(sensor)` is the standard deviation of that sensor over the **training engines only**.
- Drift is applied per engine across its own sequence; magnitudes are configurable in `config.yaml`.
- Faults must not alter true RUL labels.
- Validate suggestion fields against a sensor whitelist, kind whitelist, and per-kind magnitude range.

---

## 6. Nemotron integration (NVIDIA model)

- Only `scenarios.py` calls Nemotron.
- Sends a short structured request; expects a list in the fixed 4-field format.
- Strict validation: reject unknown sensor, unknown kind, magnitude out of range, missing/typed fields.
- **Cache replies to file** → reruns are deterministic and cost nothing.
- Nemotron never sees the data, labels, or scores.
- Compare its validated suggestions with a fixed grid; report honestly whether it found failures the grid missed (it may not).
- **DECIDED:** validated suggestions form a **union** with the fixed grid (not just a side comparison), and the dashboard tags fault provenance (grid vs Nemotron). This keeps the NVIDIA dependency meaningful. `--offline` mode runs grid-only.
- Prompt limited to schema + a few examples; no dataset statistics.

---

## 7. Evaluation metrics

- **MAE** (main), **RMSE**.
- **NASA score** — must use a documented formula (Open Q-B4).
- **Too-optimistic rate**: fraction where `predicted − true > margin`; margin fixed *before* runs. **DECIDED:** default margin = 10 flights, and a margin-sensitivity sweep (5/10/20) is reported so the number is not an artifact of one threshold.
- **DECIDED (A2) — Cost of safety:**
  - Let `d = RUL_main − RUL_safe` per evaluation unit.
  - `cost_of_safety = mean(d)` (positive ⇒ the safe model asks for maintenance earlier).
  - Report alongside: `fraction_conservative = mean(d > 0)` and `max_cost = max(d)`.
  - Report separately for clean data and each fault setting.
  - The safety *benefit* is the reduction in too-optimistic rate; report both so the trade is explicit.
- **Stress-test view**: metric change vs fault magnitude.
- **5 seeds**, report spread as **mean ± std** across seeds (also show median for skew if needed) — **DECIDED**.
- **DECIDED (A6):** primary metrics are computed per engine (aggregate each engine to one value, e.g. its final-flight or mean prediction error), because maintenance decisions are per-engine; per-flight numbers reported as secondary.
- **Range calibration**: report P10–P90 empirical coverage on validation.

---

## 8. Architecture & repository

Offline-first: heavy work writes CSV/JSON; dashboard only reads.

| File | Job |
| --- | --- |
| `data.py` | Load FD001, build labels, split by engine |
| `features.py` | Past-only mean/trend features |
| `models.py` | Train the models |
| `faults.py` | Noise, drift, missing readings on one sensor |
| `scenarios.py` | Ask Nemotron, validate, save |
| `stress_test.py` | Run every model on every damaged version, save scores |
| `evaluate.py` | Compute the scores |
| `dashboard.py` | Streamlit pages |
| `tests/` | Small test per file |

**Additions**
- `config.yaml` — all knobs (paths, cap, margin, τ, seeds, fault grid, Nemotron model/endpoint).
- `.env.example` — API key name only; real key in `.env` (git-ignored).
- `requirements.txt` — pinned versions.
- `README.md`, `LICENSE` (MIT), `.gitignore`.
- `results/` and `data/` git-ignored.

**Dependencies (planned):** pandas, numpy, scikit-learn, streamlit, plotly, requests (Nemotron), pyyaml, python-dotenv. Confirm sklearn quantile loss support before relying on it (Open Q-B3).

**Deployment:** Streamlit on Nebius. Decide **DevPod vs Serverless Endpoint** (Open Q-B2); deploy a hello-world in Week 1.

---

## 9. Assumptions to verify (do NOT treat as fact)

### A. Design decisions — RESOLVED
- **A1 (resolved):** cap training labels at 125; metrics vs uncapped true RUL.
- **A2 (resolved):** cost of safety = `mean(RUL_main − RUL_safe)` + fraction conservative + max cost.
- **A3 (resolved):** fault magnitudes in train-std / fraction units (see table in §5).
- **A4 (resolved):** partial windows use available history only; trend = 0 when <2 flights.
- **A5 (open):** guardrail that the official test set is untouched until the final run (separate script/flag).
- **A6 (resolved):** primary metrics per engine; per-flight secondary.

### B. External facts to confirm
- B1: FD001 column names, engine counts, and RUL convention vs NASA readme.
- B2: Nebius hosting choice (DevPod vs Serverless Endpoint) and limits.
- B3: sklearn quantile regression API/loss details.
- B4: Exact NASA score formula (and whether published NASA-aware losses exist).
- B5: Token Factory: exact Nemotron model name, endpoint, auth, request/response format.
- B6: Token pricing and available credits (design lists $0.06/$0.24 per M tokens for Nano — *confirm in console*).
- B7: Country eligibility (India not explicitly listed — read full rules).

---

## 10. Reproducibility & safety rules

- All seeds in `config.yaml`; log the resolved config with every run.
- Engine-disjointness test enforced in CI/local tests.
- Faults deterministic per seed; no global RNG leakage.
- Past-only features; engine-boundary reset.
- No maintenance/airworthiness advice anywhere. Disclaimer in README, app header, video.

---

## 11. Phases & acceptance criteria

### Phase 0 — Setup & Spec
- Git repo, MIT, `.gitignore` (data/, .env, results/, caches).
- Finalize this spec; second-opinion review.
- Confirm Nebius/Token Factory details and country eligibility.
- **Accept:** repo public + MIT visible; spec complete; Nebius model/endpoint confirmed; eligibility checked.

### Phase 1 — Data & Features
- Load FD001; build RUL + cap rule; seeded engine split; past-only features; constant-sensor drop.
- **Accept:** loads cleanly; engine-disjoint test passes; RUL cap justified by distribution; feature shapes documented; no NaNs silently filled.

### Phase 2 — Models & Baselines
- Constant, RF, GB, safe-side quantile GB, P10/P90.
- **Accept:** all train; all beat constant on validation MAE; safe-side is ≤ main on too-optimistic rate (directional) at fixed margin; ranges produced; coverage reported.

### Phase 3 — Robustness Lab
- `faults.py` (3 kinds × 4 magnitudes, fixed seeds) + `scenarios.py` (Nemotron optional, validated, cached, union with grid).
- **Accept:** faults only on held-out data; invalid suggestions rejected; `--offline` works; artifacts saved with a schema/version.

### Phase 4 — Stress Test & Evaluate
- Run all models × all faults × 5 seeds; compute all metrics.
- **Accept:** spread reported; cost of safety computed; too-optimistic rate at fixed margin; NASA score matches documented formula.

### Phase 5 — Dashboard & Demo
- 3 pages, read-only, fault provenance, disclaimer.
- **Accept:** reads artifacts only; shows engine view/comparison/stress tests; disclaimer present; local run OK; Nebius deploy tested.

### Phase 6 — Polish, Tests, Submission
- Per-file tests; README; <3 min video; feedback; final run; tag.
- **Accept:** tests pass; README covers setup, Nemotron usage, Token Factory, Nebius, disclaimer; video <3 min; clean tagged commit.

---

## 12. Open questions

- **A5** official test-set guardrail.
- **B1** FD001 fields/counts; **B4** NASA score formula; **B5** Token Factory model/endpoint.
- B2 Nebius hosting; B3 sklearn quantile API; B6 credits/pricing; B7 country eligibility.
- Video outline and demo flow not yet defined.
- Whether Nemotron's union-with-grid dependency is strong enough for the judges' "depends on" wording.

---

## 13. Change log
- 17 Oct 2026 — Initial spec from design doc + review.
- 17 Oct 2026 — Copied to repo root; resolved A1–A4, A6; added margin sensitivity; pinned spread/aggregation choices.
