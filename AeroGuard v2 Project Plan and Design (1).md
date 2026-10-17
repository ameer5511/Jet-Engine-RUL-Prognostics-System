# AeroGuard v2: Project Plan and Design

Oct 16, 2026 · @Ameer Muhammed

## Overview

AeroGuard predicts how many more flights a simulated jet engine can do before it needs maintenance, and then stress-tests that prediction: what happens to it when the sensors go bad?

**The problem.** Maintenance too early wastes money. Maintenance too late risks an engine failure. The dangerous mistake is guessing too many flights left. The number we predict is called Remaining Useful Life, or RUL.

**What we build, in three parts**

1. **A predictor.** A normal model, plus a "safe-side" model trained to lean cautious.
2. **A stress test (the Robustness Lab).** We damage the sensor data in three ways (random noise, a slow drift on one sensor, missing readings) and measure how much the predictions get worse.
3. **A dashboard hosted on Nebius** that shows the predictions and the stress-test results.

Nemotron, NVIDIA's open model, runs through Nebius Token Factory and has one small job: it suggests fault settings to test. Our own code checks every suggestion and produces every number.

**The headline question.** When sensors go bad, does the safe-side model stay safe, and how many flights of early maintenance does that safety cost?

**Boundary.** The data is NASA's simulation, not real airline data. This is a research and portfolio project and must not be used for real maintenance or airworthiness decisions.

## Scope

**In scope for the hackathon**

- NASA C-MAPSS **FD001**, the simplest of its four parts.
- Labels (true remaining flights) and a split by whole engines.
- Four models: a constant baseline, Random Forest, gradient boosting, and a safe-side gradient-boosting model.
- Scores: MAE as the main one, RMSE, the NASA score, and how often the model is too optimistic.
- The Robustness Lab: three fault types at four sizes each.
- Nemotron Nano suggesting extra fault settings, which our code validates and saves.
- A dashboard hosted on Nebius with an engine view, a model comparison and the stress-test charts.
- A public GitHub repository with an MIT licence, a README, a short video and written feedback.

**Left out until after the deadline**

- The other three parts of C-MAPSS and the large N-CMAPSS dataset.
- Neural networks (GRU or LSTM), a custom NASA-score loss, anomaly alarms and drift monitoring.
- A chatbot or any AI-written text on the dashboard, our own API, a database, user accounts, 3D plots and real hardware.

**If time runs short, cut in this order:** Nemotron's suggestions (use fixed tests only, plus a small checked notes-writer so the entry still uses an NVIDIA model), then dashboard polish, then one fault type. The comparison of the safe-side model against the normal one is never cut.

## How it works

&#91;embedded content: architecture · offline work, saved results, hosted dashboard\]

The heavy work runs offline and writes files. The dashboard only reads them, so nothing trains while a page loads. Nemotron's suggestions are checked by our code before any test runs.

### The data

FD001 is one of the four parts of NASA's C-MAPSS simulation. It has 100 engines run until failure for training and 100 more for testing. Each row is one engine on one flight, with 3 operating settings and 21 sensor readings.

### Labels

The label is the answer the model must learn: how many flights are left. If a training engine flew 200 flights, flight 150 has the label 50 and flight 190 has the label 10. We cap the label at 125, because a healthy engine looks the same with 200 flights left as with 160, and telling those apart only confuses the model. The cap is tested on validation data only.

### Features

Features are what the model looks at. For each sensor we use three numbers: the current value, the average of the last 10 flights, and the trend (how much it changed over those flights). If a sensor read 47.2, 47.3 and 47.5, the average is about 47.3 and the trend is up 0.3. We use past flights only, never the future, and we drop sensors that never change. That leaves roughly 40 inputs; the exact count is checked in the data.

### The split

We split by whole engines, never by single rows, because rows from one engine look alike and would give a fake score. About 80 engines train the models and about 20 validate them. The official test engines stay untouched until the very end.

## Detailed design

### Files

Each file does one job, so each can be tested on its own.

| File | Job |
| --- | --- |
| data.py | Load FD001, build the labels, split by engine |
| features.py | Build the averages and trends from past flights |
| models.py | Train the models below |
| faults.py | Add noise, drift or missing readings to one sensor |
| scenarios.py | Ask Nemotron for fault settings, check them, save them |
| stress\_test.py | Run every model on every damaged version and save the scores |
| evaluate.py | Work out the scores |
| dashboard.py | The Streamlit pages |
| tests/ | One small test per file |

### Models

| Model | What it does | Why it is here |
| --- | --- | --- |
| Constant | Always predicts the average remaining life | The floor every real model must beat |
| Random Forest | Many small decision trees, answers averaged | Strong with little tuning |
| Gradient boosting | Trees built one after another, each fixing the last one's mistakes | Often the best on table data; the main model |
| Safe-side boosting | The same boosting, trained with scikit-learn's quantile setting at about 0.3 | Aims low on purpose, so it leans cautious |

The quantile setting at 0.3 means the model is trained to be too optimistic only about 30% of the time instead of about 50%. That is a training target, not a promise on new data. The price of safety is that it will sometimes ask for maintenance earlier than needed, and we measure that price.

Two more boosting runs, for the 10th and 90th percentiles, give a low and a high estimate, so each prediction comes with a range.

### Faults in the Robustness Lab

Faults are applied only to held-out data, never to training data. There are three kinds, each at four sizes with a fixed random seed:

- **Noise:** random wobble added to a sensor.
- **Drift:** one sensor slowly shifts away over the engine's flights.
- **Missing readings:** some readings are blanked and filled with the last known value.

A fault setting has four fields: which sensor, which kind, how big, and a seed.

### What Nemotron does

The only place Nemotron appears is scenarios.py. It sends Nemotron a short request for fault settings and expects a list in the fixed four-field format. Our code throws out any setting with an unknown sensor, an unknown kind, a size outside the allowed range or a missing field. Replies are saved to a file, so a rerun gives the same tests and costs nothing. Nemotron never sees the data or the scores.

We also compare its suggestions with a plain grid of fixed settings and report honestly whether it found any failures the grid missed. The answer may be no.

### Scores

- **MAE** (average error in flights) is the main score, with RMSE next to it.
- **NASA score:** a standard penalty that grows faster when the model is too optimistic.
- **Too-optimistic rate:** the share of test engines predicted too high by more than a set margin. We fix the margin before running (start at 10 flights).
- **Cost of safety:** how many flights early the cautious model is on average.
- **Stress-test view:** how each score changes as a fault gets bigger.
- Everything is run with 5 random seeds, and we report the spread.

### Dashboard

Three pages: an **engine view** (pick an engine, move a slider to a flight, see the predicted and the true flights left), a **model comparison**, and a **stress-test page** (a line showing error against fault size, and a grid showing which faults hurt most). It is hosted on Nebius; check whether a DevPod or a Serverless Endpoint is the simplest for a Streamlit app.

## Technology choices and why

The rule behind every choice: use the simplest thing that answers the question, and write down when we would switch.

### The prediction work

| Choice | Why we chose it | Alternatives | Why not those now |
| --- | --- | --- | --- |
| Python | Has the most machine-learning tools, and you are learning it | R, Julia | Fewer tools and examples for this work |
| scikit-learn | One consistent way to train all four models; the quantile setting is built in; well documented | XGBoost, LightGBM, PyTorch | Needed only for a custom NASA-score loss or a neural network, which come in the paper phase |
| Random Forest and gradient boosting | Strong on table data and quick to run on your Mac | GRU or LSTM neural networks | They need more tuning and a GPU, and are harder to debug while learning; simple models also make a fairer baseline |
| NASA FD001 | The simplest part: 100 engines, one condition, one fault type, so it proves the pipeline | FD002 to FD004, N-CMAPSS | Harder or much bigger; they come after the pipeline works |
| Split by whole engines | Rows from one engine look alike, so a row-level split gives a fake score | Random row split | Gives inflated results |
| Quantile setting for the safe-side model | Simple, built in, and also gives a low and high range | A custom loss tied to the NASA score | More code; planned for the paper phase |
| Robustness Lab | Asks a real reliability question using models we already have, and gives Nemotron a real job | Drift Sentinel, anomaly detection | They need more statistics, and an alarm tested on faults we planted ourselves proves little |
| Plain CSV and JSON files | Enough for a static benchmark | A database | No users or live data to store |
| Streamlit and Plotly | Python only and quick to build | Gradio, FastAPI with React | Only needed if several clients must use the model |

### NVIDIA, Nebius and tools

| Choice | Why we chose it | Alternatives | Why not those now |
| --- | --- | --- | --- |
| NVIDIA Nemotron Nano | The hackathon requires at least one NVIDIA open model. Nano is the small, cheap one, and we only need short structured output. A third-party listing prices it at $0.06 and $0.24 per million input and output tokens, against $0.30 and $0.90 for Super and $1.00 and $3.00 for Ultra (confirm in your console) | Nemotron Super or Ultra | Cost more and add nothing for this job |
| Not using Nemotron to predict | It is a language model, not built for table data, and its mistakes would sound confident | Fine-tuning it on engine data | Heavy GPU work for a worse, less testable result |
| Not using NIM, TensorRT-LLM, NeMo Curator, NeMo Evaluator or DGX Spark | They need extra infrastructure or hardware (the evaluator is set up on Kubernetes), and our own checks do the job | The same tools | Kept as future steps for private, self-hosted use |
| Nebius | The rules need the project to call Token Factory or run on Nebius compute. Hosting the dashboard there meets this with one account | Streamlit Community Cloud plus Token Factory calls | Works only if the app makes a live call; hosting on Nebius is the safer fit |
| No GPU for now | The data is small and runs on your Mac | Colab, Kaggle, Nebius GPU, AMD Developer Cloud | Needed only for the GRU in the paper phase |
| GitHub with an MIT licence | The rules need a public repository with an open-source licence, and commits let you roll back | GitLab | Same idea, but GitHub is the one named first in the rules |
| A coding helper such as OpenCode | Speeds up writing code | None needed | It does not count as using Nemotron, and we still review and test every core function ourselves |

NVIDIA is in this project because the hackathon requires an NVIDIA model. The predictions themselves do not depend on it.

## Plan by week

&#91;embedded content: weekly plan · two weeks to submission, paper phase after\]

The two weeks add up to about 56 hours. Each gate must pass before the next stage starts. If a week slips, cut in the order given in Scope, and tag a clean commit at the end of each week so you can roll back to it.

## Working rules and learning plan

This is engineering with better tools, not vibe coding. The prep work is what makes the tools useful.

### Working rules

1. **GitHub from day one.** Create the repository today with an MIT licence and a .gitignore that keeps the raw data out. Commit after every working step, and tag the end of each week so you can roll back to a clean state.
2. **Write SPEC.md before any code.** One page covering what we build, the rules (split by engine, past data only, no maintenance advice), the file structure, what each file takes in and returns, what "done" means, and what is out of scope.
3. **Review the spec before building.** Ask a model to find gaps and contradictions, then give the same spec to a different model for a second opinion. Fix the spec first.
4. **No code you cannot explain.** Every core function gets a small test, and every library call is checked against its official documentation.
5. **Keep an experiment log.** For each run write what you changed and what happened. It becomes the method section of the paper.

### Learning plan for the first three days

| Day | Learn | Proof you learned it |
| --- | --- | --- |
| 1 (16 Oct) | What a turbofan engine does (one 15-minute video) and NASA's C-MAPSS description | A one-page note in your own words |
| 2 | Python for data: Kaggle's Pandas course, skimmed | A notebook that loads FD001 and plots four sensors for three engines |
| 3 | Kaggle's Intro to ML, the leakage lesson in Intermediate ML, and StatQuest on gradient boosting | Constant baseline and Random Forest scored on an engine-level split |

After that, learn while building. For reference: Google's Rules of Machine Learning, Chip Huyen's Designing Machine Learning Systems, the free Interpretable Machine Learning book for later work on explanations, and the Python edition of An Introduction to Statistical Learning.

## Hackathon checklist

The Nebius x NVIDIA Global AI Hackathon closes on 30 Oct 2026 at 10:00 am PT, which is about 10:30 pm IST. We plan to submit on 29 Oct. Judging runs 1 to 15 Dec. The source is the Official Rules text; recheck the Devpost page for changes.

### This week

- [ ] Register on Devpost and click Join Hackathon
- [ ] Join the Nebius Builder Program and note the credit you actually receive
- [ ] Open the Token Factory console; note the Nemotron models it lists and the exact model name and web address to call
- [ ] Confirm the track (we expect Best Apps and Agents)
- [ ] Join the AMD AI Developer Program to see whether the GPU credits arrive without entering a hackathon (for the paper phase)

### Rules the project must meet

- [ ] It runs on Nebius compute or makes a live call to Token Factory (hosting the dashboard on Nebius covers this)
- [ ] It uses at least one NVIDIA open model in a way the project depends on
- [ ] It is a genuine attempt at the track, because Stage One is pass or fail
- [ ] AeroGuard existed before 26 Aug, so a written note explains what was significantly updated during the submission period

### Submission items

- [ ] Working demo link, free and open for testing until judging ends on 15 Dec
- [ ] Public repository with an MIT or Apache 2.0 licence visible at the top
- [ ] README with setup steps, how Nemotron is used, where Token Factory helped, and any other Nebius services
- [ ] Public YouTube video under 3 minutes showing the project running, with no copyrighted music
- [ ] Project description, track choice and written feedback on Nebius and NVIDIA tools

### How judges score

Four equal criteria: technological implementation, design, potential impact and quality of the idea. Our answers are a clean tested pipeline with a real Nemotron job, a complete dashboard, a clear safety result, and an honest report of what the stress test found. Treat any prize as a bonus, since the field is large; the credits and the project are the sure gains.

## Risks and fallbacks

The biggest risk is time. Everything below has a fallback that keeps the project eligible and honest.

| Risk | What could happen | What we do |
| --- | --- | --- |
| Time runs out while learning Python | Missed deadline | Follow the cut order in Scope, use the weekly gates, and tag a clean commit each week |
| Nemotron's suggestions add nothing over a plain grid of faults | A weak use of the NVIDIA model | Report it honestly and keep the fixed tests; fall back to a small notes-writer whose every number is checked by code |
| Nemotron returns odd or invalid settings | Broken tests | Strict format checks, discard anything invalid, and save replies |
| Reruns or judge visits use up the credits | The demo stops working | Saved replies, the small Nano model, a limit on live calls, and a weekly look at the credit balance |
| Data leakage between flights of one engine | A fake, inflated score | Split by engine, add a test that no engine appears on both sides, and use past-only features |
| The safe-side model asks for maintenance far too early | The safety looks too expensive | Measure and report the cost of safety, and choose the quantile on validation data only |
| Trouble hosting on Nebius | The demo cannot be opened | Deploy a hello-world page in Week 1, and keep a recorded video as backup |
| Simulated data mistaken for real engines | Unsafe or misleading claims | A clear disclaimer in the README, the app and the video |
| The demo goes down before 15 Dec | It cannot be judged | Keep it running and check it weekly |

## Future scope and open items

### Future scope

| Extension | When to start |
| --- | --- |
| A custom loss tied to the NASA score (LightGBM or XGBoost) | Paper phase, after the quantile version is measured |
| FD002 to FD004, with 6 operating conditions and 2 fault modes | After the FD001 pipeline is stable |
| GRU or LSTM networks in PyTorch on a GPU | After the hackathon; GPU options are Nebius, free Colab or Kaggle, AMD credits, or IndiaAI subsidised GPUs |
| The large N-CMAPSS full-flight dataset | When GPU time is available |
| Drift Sentinel, an alarm for data that has changed | After the Robustness Lab shows which faults matter |
| Self-hosted Nemotron (NIM, TensorRT-LLM) for private data | If data must not leave your own network |
| A latency test on a Jetson | If you want an edge-deployment claim |
| A PHM Society data challenge | When the next one opens |
| A literature check for novelty | Before any claim that the work is new |

### Open items to verify

- [ ] Your Nebius credit amount, and whether free GPU hours exist (the rules do not say)
- [ ] Which Nemotron models your Token Factory account lists, with the exact model name and web address
- [ ] Which Nebius hosting option (DevPod or Serverless Endpoint) is simplest for Streamlit
- [ ] The final track
- [ ] The FD001 engine counts (100 training and 100 test) against NASA's readme
- [ ] Whether loss functions aware of the NASA score are already published
- [ ] That your country is eligible: India is not among the places the rules name, but read the full list
- [ ] The unsourced figures in the earlier supervisor report (for example "23% of LSTM outputs non-monotonic") before anyone quotes them2
