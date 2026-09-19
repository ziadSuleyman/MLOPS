# Olist Late Delivery Predictor

> **MLOps Task 3** — From Notebooks to Production  
> Qafza MLOps Training 2026/2027

A production inference service that predicts whether an Olist e-commerce order
will be delivered **late** or **on time**. For each order it returns a calibrated
probability, a risk percentile, an alert decision under a fixed alert budget, and
the version of the artifacts that produced it. Built from the 6-notebook pipeline
in Task 2 — and it reproduces the notebooks exactly (ROC-AUC 0.7107 on the test split).

---

## Table of Contents

- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Running Locally](#running-locally-without-docker)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Pipeline Flow](#pipeline-flow)
- [Model Details](#model-details)
- [From Score to Decision](#from-score-to-decision)
- [Data Validation](#data-validation)
- [Monitoring & Observability](#monitoring--observability)
- [Scoring Predictions Against Real Deliveries](#scoring-predictions-against-real-deliveries)
- [DVC — Artifact Versioning](#dvc--artifact-versioning)
- [MLflow — Model Registry](#mlflow--model-registry)
- [Testing](#testing)
- [CI/CD](#cicd)
- [Configuration](#configuration)
- [Scripts](#scripts)
- [Key Design Decisions](#key-design-decisions)

---

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────────┐
│   Client     │────▶│  FastAPI      │────▶│  Inference Pipeline   │
│  (curl/app)  │     │  /predict     │     │  ┌──────────────────┐ │
└──────────────┘     │  /drift       │     │  │ Validate          │ │
                     └──────┬───────┘     │  │ rules + GE(batch) │ │
                            │             │  ├──────────────────┤ │
                     ┌──────▼───────┐     │  │ Features          │ │
                     │  Prometheus   │     │  │ derive, TE, OHE,  │ │
                     │  /metrics     │     │  │ impute (fitted)   │ │
                     └──────────────┘     │  ├──────────────────┤ │
                                          │  │ Predict           │ │
                     ┌──────────────┐     │  │ score → calibrate │ │
                     │  MLflow       │     │  │ → alert budget    │ │
                     │  :5000        │     │  ├──────────────────┤ │
                     └──────────────┘     │  │ Record            │ │
                                          │  │ JSONL (order_id), │ │
                     ┌──────────────┐     │  │ metrics, drift    │ │
                     │  PostgreSQL   │◀────┼──┤ (outcomes later)  │ │
                     │  :5433        │     │  └──────────────────┘ │
                     └──────────────┘     └──────────────────────┘
```

**Three containers and one start-up step** via Docker Compose:

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| `db` | postgres:16 | 5433 | Olist database (same schema as Task 1) |
| `mlflow` | ghcr.io/mlflow/mlflow:v3.16.1 | 5000 | Experiment tracking, model registry, artifact store |
| `register` | Built from `Dockerfile` | — | One-off: registers the model if the registry does not hold it yet, then exits |
| `api` | Built from `Dockerfile` (python:3.13-slim) | 8000 | Inference API — loads the model **from the registry** |

Start-up order: `db` + `mlflow` healthy → `register` finished → `api`.

---

## Quick Start

```bash
# 1. Clone and enter
git clone https://github.com/ziadSuleyman/MLOPS.git
cd MLOPS/mlops-task3

# 2. Copy the env file and edit if needed
cp .env.example .env

# 3. Start everything (DB + MLflow + model registration + API)
docker compose up --build

# 4. Test the API
curl http://localhost:8000/health
curl http://localhost:8000/docs
```

The API is live at `http://localhost:8000`. Interactive docs at `/docs`.
`/health` should report `"model_source": "olist-late-predictor@production (version N, …)"`.

Port 8000 taken by something else? Pick another host port: `API_PORT=8010 docker compose up --build`.
Always keep `--build`: without it Compose reuses an old image.

---

## Running Locally (without Docker)

Use **Python 3.13** — the model was trained under it, with scikit-learn 1.9.0.

```bash
# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac

# Install dependencies
pip install -r requirements/dev.txt

# Make sure the Task 1 database is running (only needed for evaluate_outcomes.py)
cd ../mlops-task1 && docker compose up -d && cd ../mlops-task3

# Copy .env
cp .env.example .env

# Run the API
uvicorn app.main:app --reload --port 8000

# Run the tests
pytest
```

---

## API Reference

### `GET /health`

Health check — is the service up and the model loaded?

**Response** `200`:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "f9fcc48ef9ef",
  "service_version": "1.2.0",
  "model_source": "olist-late-predictor@production (version 2, run e90dde2c919c4cf18383b421f76b368c)"
}
```

`model_version` is not typed by hand: it is a hash of the four serving artifacts
(model, transformers, feature list, serving reference). Replace any of them and
it changes by itself. `model_source` says where they were loaded from — the MLflow
registry, or `models/` when the registry was not reachable.

### `GET /model/info`

Model metadata — type, versions, features, hyperparameters, metrics, the md5 of
every artifact (the same md5 DVC records), the calibration and the alert policy.

**Response** `200` (abridged):
```json
{
  "model_type": "LogisticRegression",
  "model_version": "f9fcc48ef9ef",
  "service_version": "1.2.0",
  "model_source": {"source": "registry", "detail": "olist-late-predictor@production (version 2, …)", "registry_version": "2"},
  "n_features": 31,
  "hyperparameters": {"C": 0.3, "class_weight": "balanced", "max_iter": 3000},
  "label_definition": "calendar-day rule",
  "metrics": {
    "validation": {"ROC-AUC": 0.7739, "PR-AUC": 0.1709},
    "test": {"ROC-AUC": 0.7107, "PR-AUC": 0.0849}
  },
  "artifacts_md5": {"model": "…", "transformers": "…", "feature_list": "…", "reference": "…"},
  "calibration": {"method": "platt", "fitted_on": "validation", "fitted_on_late_rate": 0.0553},
  "alerting": {"budget": 0.05, "current_threshold": 0.7538, "threshold_source": "validation split", "window_size": 0}
}
```

### `POST /predict`

Predict for a single order. All fields available at **purchase time only** — no
post-delivery data allowed. `order_id` is optional but recommended: without it the
prediction cannot be scored against the real delivery later.

**Request body** (a real order from the test split):
```json
{
  "order_id": "8a9be36ffd78382f9ac518945e909636",
  "order_purchase_timestamp": "2018-06-01 04:19:11",
  "order_estimated_delivery_date": "2018-07-05",
  "shipping_limit_first": "2018-06-11 04:30:37",
  "customer_state": "MG",
  "main_seller_state": "SP",
  "customer_lat": -19.5175,
  "customer_lng": -42.6117,
  "seller_lat": -23.6524,
  "seller_lng": -46.7555,
  "freight_total": 18.43,
  "freight_max": 18.43,
  "items_price_total": 78.0,
  "items_price_max": 78.0,
  "payment_total": 96.43,
  "installments_max": 4,
  "n_payments": 1,
  "main_payment_type": "credit_card",
  "n_items": 1,
  "n_products": 1,
  "n_sellers": 1,
  "n_categories": 1,
  "weight_g_total": 250,
  "photos_avg": 4,
  "main_category": "relogios_presentes"
}
```

**Response** `200`:
```json
{
  "order_id": "8a9be36ffd78382f9ac518945e909636",
  "prediction": "on_time",
  "probability": 0.0221,
  "score": 0.3197,
  "risk_percentile": 20.8,
  "alert_threshold": 0.7538,
  "model_version": "f9fcc48ef9ef",
  "latency_ms": 44.4,
  "validation_warnings": null
}
```

| Field | Meaning |
|-------|---------|
| `prediction` | `"late"` = the order is inside the alert budget (the riskiest 5% of recent orders) |
| `probability` | Calibrated probability of a late delivery |
| `score` | Raw model score — a ranking, not a probability (the model is class-balanced) |
| `risk_percentile` | Share of validation orders that scored lower |
| `alert_threshold` | The score needed for `"late"` at the time of the request |
| `validation_warnings` | Rules the order broke, only when `validation.on_failure: flag` |

**Response** `422` (a broken rule):
```json
{
  "detail": "Input validation failed",
  "errors": ["freight_total must be >= 0, got -5.0"]
}
```

### `POST /predict/batch`

Predict for up to `inference.batch_max_size` (500) orders at once. Every order is
validated, predicted and logged individually; the Great Expectations suite also
checks the batch as a whole.

**Request body**:
```json
{
  "orders": [
    { "order_id": "…", "order_purchase_timestamp": "…", "…": "…" },
    { "order_id": "…", "order_purchase_timestamp": "…", "…": "…" }
  ]
}
```

**Response** `200`: `{"predictions": [ … one object as in /predict … ], "count": 2}`

### `GET /drift`

Drift over the most recent predictions (window of 1,000). Measured after serving
the whole test split in purchase order — the window then holds 20–29 August 2018:

```json
{
  "window_size": 1000,
  "min_window": 200,
  "psi": 0.6646,
  "psi_status": "alert",
  "alert_ratio": 0.035,
  "mean_probability": 0.0758
}
```

### `GET /metrics`

Prometheus metrics in text format. Scrape this endpoint with Prometheus.

### `GET /docs`

Interactive Swagger UI documentation (auto-generated by FastAPI).

---

## Project Structure

```
mlops-task3/
├── app/                        # FastAPI service
│   ├── main.py                 #   routes, lifespan checks, error handling
│   └── schemas.py              #   Pydantic request/response models
│
├── src/                        # Inference pipeline modules
│   ├── config.py               #   central config loader (YAML + .env), fails fast on bad settings
│   ├── logger.py               #   rotating file + console logging
│   ├── artifacts.py            #   model version = artifact hash, DVC check, sklearn version guard
│   ├── preprocessing.py        #   derived features (haversine, ratios)
│   ├── features.py             #   apply fitted transformers (TE, OHE, impute)
│   ├── validation.py           #   per-order rules (fatal vs. warnings)
│   ├── ge_validation.py        #   Great Expectations suite for batches (70 expectations)
│   ├── reference.py            #   calibration + validation score distribution
│   ├── alerting.py             #   alert budget → rolling threshold
│   ├── predict.py              #   model loading, score → probability → decision
│   ├── pipeline.py             #   end-to-end: validate → features → predict → record
│   ├── monitoring.py           #   Prometheus, PSI drift, JSONL prediction log
│   ├── registry.py             #   register the bundle in MLflow; load it back (alias production)
│   └── evaluation.py           #   score logged predictions against real deliveries
│
├── config/
│   └── settings.yaml           # All configuration — no hardcoded values
│
├── models/                     # Serving artifacts (DVC-tracked; registered in MLflow)
│   ├── 05_transformers.joblib  #   target encoders, OHE, imputer, scaler  (Task 2)
│   ├── 05_feature_list.json    #   31-feature contract + metadata          (Task 2)
│   ├── 06_model.joblib         #   LogisticRegression(C=0.3, balanced)     (Task 2)
│   ├── 07_serving_reference.json # calibration + validation score percentiles
│   └── *.dvc                   #   DVC pointer files
│
├── scripts/                    # Operations & maintenance scripts
│   ├── build_serving_reference.py  # fit calibration on validation → 07_serving_reference.json
│   ├── make_parity_fixture.py  #   real test orders + notebook features → tests/fixtures
│   ├── register_model.py       #   register the serving bundle in MLflow (alias + stage + md5 tags)
│   ├── compare_models.py       #   compare local artifacts vs MLflow registry
│   ├── analyze_predictions.py  #   prediction log: alert ratio, latency, PSI drift
│   └── evaluate_outcomes.py    #   prediction log + orders table → real-world accuracy
│
├── tests/                      # pytest suite (173 tests)
│   ├── conftest.py             #   shared fixtures (a real order, temp logs)
│   ├── helpers.py              #   small shared helpers
│   ├── fixtures/parity_orders.parquet  # 314 real test orders + notebook features
│   └── test_*.py               #   14 modules, see Testing
│
├── notebooks/                  # Reference notebooks from Task 2 (read-only)
├── logs/                       # Runtime logs (gitignored)
├── requirements/
│   ├── base.txt                # Runtime dependencies (pinned to the training env)
│   └── dev.txt                 # Dev: testing, linting, DVC, Jupyter
│
├── Dockerfile                  # Production image (python:3.13-slim)
├── docker-compose.yml          # Full stack: DB + MLflow + API
├── .env.example                # Template for secrets
├── .dvc/config                 # DVC remote (see DVC section)
├── .dvcignore                  # DVC ignore rules
├── .gitignore                  # Git ignore
├── .pre-commit-config.yaml     # Pre-commit hooks (ruff, no-commit-to-main)
├── ../.github/workflows/task3-ci.yml  # CI (GitHub reads workflows from the repo root)
└── pyproject.toml              # ruff + pytest config
```

---

## Pipeline Flow

```
Raw Order (dict, optional order_id)
    │
    ▼
┌─────────────────────────────┐
│  1. Validate                │  src/validation.py  (every order)
│     fatal: missing/unread-  │  - always rejected (cannot be scored)
│       able dates, non-      │
│       numbers, Infinity     │
│     problems: unknown state │  - "reject" → 422
│       or payment, negative  │  - "flag"   → scored, warnings attached
│       money, promise before │
│       purchase, bad coords  │  src/ge_validation.py  (batches only)
│     GE suite on batches     │  - same on_failure policy
└────────────┬────────────────┘
             ▼
┌─────────────────────────────┐
│  2. Build Derived Features  │  src/preprocessing.py
│     haversine distance,     │  23 derived columns
│     promised_days, ratios,  │  (pure row-level transforms)
│     same_state, geo flags   │
└────────────┬────────────────┘
             ▼
┌─────────────────────────────┐
│  3. Build Features          │  src/features.py
│     target encoding, OHE,   │  fitted objects from 05_transformers.joblib
│     ±Infinity → missing,    │  → exactly 31 features, never refitted
│     impute (train median)   │
└────────────┬────────────────┘
             ▼
┌─────────────────────────────┐
│  4. Predict                 │  src/predict.py
│     scale → score           │  06_model.joblib
│     score → probability     │  Platt calibration (07_serving_reference.json)
│     score → late/on_time    │  alert budget 5%, rolling threshold
└────────────┬────────────────┘
             ▼
┌─────────────────────────────┐
│  5. Record & Return         │  src/monitoring.py
│     JSONL, one line/order,  │  with order_id + input → scorable later
│     Prometheus, PSI window  │
└─────────────────────────────┘
```

---

## Model Details

| Property | Value |
|----------|-------|
| **Algorithm** | Logistic Regression |
| **Hyperparameters** | C=0.3, class_weight="balanced", max_iter=3000 |
| **Features** | 31 (see `models/05_feature_list.json`) |
| **Most important feature** | `promised_days` (permutation importance 0.126, Notebook 6) |
| **Note** | `promise_per_100km` was the EDA's strongest *single* signal (AUC 0.612) but is nearly irrelevant inside the model (0.005): once the model sees promise and distance separately, it asks "is this promise enough for this distance?" itself |
| **Label** | `is_late` — calendar-day rule (delivered on a later date than promised) |
| **Late rate** | 7.95% train · 5.53% validation · 3.61% test (6.77% over all 96,470 labeled orders) |
| **Train split** | < 2018-04 (64,320 orders) |
| **Validation split** | Apr–May 2018 (13,547 orders) |
| **Test split** | Jun+ 2018 (18,603 orders) |
| **Validation ROC-AUC** | 0.7739 |
| **Test ROC-AUC** | 0.7107 |
| **Requires scaling** | Yes (StandardScaler in transformers bundle) |
| **Refitting** | Never — fitted objects loaded as-is from Task 2 |
| **Trained with** | Python 3.13, scikit-learn 1.9.0 (the service refuses any other scikit-learn) |

---

## From Score to Decision

The model was trained with `class_weight="balanced"`, so its raw score averages
**0.49** while the real late rate is 3–8%. The score ranks orders well
(ROC-AUC 0.71) but it is not a probability, and cutting it at 0.5 flags about half
of all orders. Two separate fixes turn it into something usable, both fitted
offline by `scripts/build_serving_reference.py` and stored in
`models/07_serving_reference.json`:

1. **Probability — Platt calibration fitted on the validation split.**
2. **Decision — an alert budget, not a threshold.** Notebook 6 found that a numeric
   threshold does not survive a change of period, because the late rate keeps
   moving (7.95% → 5.53% → 3.61%). What carries over is the budget: *flag the
   riskiest 5%*. The threshold is the 95th percentile of the last 1,000 scores; until
   200 scores have been seen, the validation split's 95th percentile (0.7538) stands in.

Measured on the 18,603 test orders (evaluation only, nothing fitted on them):

| | Share flagged "late" | Precision | Recall |
|---|---|---|---|
| Threshold 0.5 (before) | 49.3% | 6.0% | 82.0% |
| Fixed validation threshold | 8.7% | 8.6% | 20.8% |
| **Rolling 5% budget (now)** | **5.3%** | **10.5%** | **15.2%** |

| | Mean "probability" | Brier score |
|---|---|---|
| Raw score (before) | 0.493 | 0.2685 |
| **Calibrated (now)** | **0.070** | **0.0379** |
| Real late rate | 0.036 | — |

The calibrated probabilities are still about twice the real test rate: the late
rate fell from 5.5% (validation, where calibration was fitted) to 3.6% (test). That
is drift, and it is exactly what `scripts/evaluate_outcomes.py` exists to measure —
recalibrate on the latest delivered orders when the gap opens.

---

## Data Validation

Two layers, with a policy for what happens when they fail
(`validation.on_failure` in `config/settings.yaml`):

- `reject` (default) — HTTP 422 with the list of problems.
- `flag` — the order is scored anyway; the problems travel with it in the
  response (`validation_warnings`), the prediction log and a Prometheus counter.

Problems that make a prediction impossible always reject, whatever the policy.

### 1. Per-order rules (`src/validation.py`)

Every order, every request. Checked against all 96,470 real orders: none breaks a rule.

| Kind | Rule |
|------|------|
| fatal | the 23 input fields present; purchase and promised dates present, parseable, without a timezone |
| fatal | numeric fields numeric and finite |
| problem | `customer_state` / `main_seller_state` ∈ 27 Brazilian states |
| problem | `main_payment_type` ∈ {credit_card, boleto, debit_card, voucher} |
| problem | money and weight ≥ 0; items, products, sellers, payments ≥ 1; installments ≥ 0 (0 occurs in the data) |
| problem | coordinates on Earth (|lat| ≤ 90, |lng| ≤ 180) |
| problem | promised delivery after the purchase |

8 leakage columns (delivery dates, status, labels) can never become features:
`check_no_leakage` runs on the feature contract at startup.

### 2. Great Expectations Suite (`src/ge_validation.py`)

70 statistical expectations, run on every **batch** request (`validation.ge_on_batch`).
They describe a population — "95% of freight values in the usual range", "at most
10% missing coordinates" — so they run on batches, not on single orders (one run
costs ~0.9 s against ~40 ms for a prediction). Ranges live in `settings.yaml`.

- Column existence (24)
- Plausible numeric ranges (13, `mostly=0.95`)
- Coordinates inside Brazil's bounding box (4, `mostly=0.90`) — lng up to −28, since
  Recife and João Pessoa lie east of −35
- Categorical set membership (3, `mostly=0.99`)
- Null rate limits (22)
- Positive counts (4)
- Leakage column absence (1 custom check)

```python
from src.ge_validation import validate_orders_ge

result = validate_orders_ge(df)
# result["success"], result["statistics"], result["failed_expectations"]
```

---

## Monitoring & Observability

### Prometheus Metrics (`GET /metrics`)

| Metric | Type | Labels | Purpose |
|--------|------|--------|---------|
| `prediction_requests_total` | Counter | `prediction` (late/on_time) | Orders scored (single and batch) |
| `prediction_latency_seconds` | Histogram | — | Request latency p50/p95/p99 |
| `prediction_errors_total` | Counter | `error_type` (validation/internal) | Error tracking |
| `prediction_batch_size` | Histogram | — | Batch request sizes |
| `prediction_validation_warnings_total` | Counter | — | Orders served with warnings (`flag`) |
| `prediction_score_psi` | Gauge | — | **Drift**: PSI of recent scores vs. the validation split |
| `prediction_alert_ratio` | Gauge | — | Share of recent orders flagged (should sit near the budget) |
| `prediction_mean_probability` | Gauge | — | Mean calibrated probability of recent orders |

**Why PSI, not "late ratio vs. 6.77%".** The share flagged is set by the alert
budget, so it cannot drift; and the real late rate moved 7.95% → 3.61% between
periods, so comparing predictions with one fixed rate would alarm forever. What
drifts is the input, and the score distribution summarises it. PSI < 0.10 is ok,
0.10–0.25 warn, ≥ 0.25 alert (`monitoring.psi_warn` / `psi_alert`).

Serving the test split month by month shows what it catches:

| Test month | PSI vs. validation | Status | Median promised days |
|------------|-------------------|--------|----------------------|
| June 2018 | 0.208 | warn | 27 |
| July 2018 | 0.195 | warn | 19 |
| August 2018 | 0.749 | **alert** | 13 |
| Whole test split | 0.124 | warn | — |

The delivery promise — the model's most important feature — shrank from 27 to 13
days, and the scores moved with it. (Part of August's shift may be the dataset's
cut-off: only orders delivered before it are labeled, which favours short promises.)
This is the same period in which ROC-AUC fell from 0.774 to 0.711.

### JSONL Prediction Log

Every order — single or batch — is one line in `logs/predictions.jsonl`:

```json
{"timestamp": "2026-09-19T09:37:32+00:00", "order_id": "8a9be36ffd78382f9ac518945e909636",
 "input_hash": "bf1070a171f3", "prediction": "on_time", "probability": 0.0221, "score": 0.3197,
 "risk_percentile": 20.8, "alert_threshold": 0.7538, "model_version": "f9fcc48ef9ef",
 "latency_ms": 44.4, "batch_size": 1, "validation_warnings": null, "input": {"…": "…"}}
```

Analyze with the built-in script:

```bash
python scripts/analyze_predictions.py              # last 24 hours
python scripts/analyze_predictions.py --hours 168   # last 7 days
python scripts/analyze_predictions.py --all          # everything
```

Output: volume, share with an `order_id`, alert ratio, mean probability, latency
percentiles, model versions, and the score PSI against the validation split.

---

## Scoring Predictions Against Real Deliveries

Because every log line carries its `order_id`, a prediction can be scored once the
order is delivered:

```bash
python scripts/evaluate_outcomes.py --all
```

It looks the orders up in the `orders` table, applies Task 2's calendar-day label,
and reports ROC-AUC, PR-AUC, alert precision/recall, and the calibration gap (mean
probability − real late rate). This is the signal for recalibrating or retraining.

The whole loop, run end to end: the 18,603 test orders served through the service in
batches, logged, then scored against the Task 1 database:

```json
{
  "n_logged": 18603, "n_with_order_id": 18603, "n_delivered": 18603,
  "late_rate": 0.0361, "roc_auc": 0.7107, "pr_auc": 0.0849,
  "flagged_share": 0.0539, "precision": 0.1027, "recall": 0.1533,
  "mean_probability": 0.0697, "brier": 0.0379
}
  Calibration gap (mean probability - real late rate): +3.36%
```

ROC-AUC and PR-AUC equal Notebook 6's test numbers exactly — measured this time
from the prediction log and the database, not from a notebook.

---

## DVC — Artifact Versioning

The four serving artifacts are tracked by [DVC](https://dvc.org/); their `.dvc`
pointer files record each file's md5. At startup the service compares every
artifact with its pointer and logs any mismatch.

```bash
python -m dvc status          # working copy vs. pointers
python -m dvc push            # upload artifacts to the remote
python -m dvc pull            # restore them from the remote
python -m dvc status -c       # local cache vs. remote
```

The default remote `localstore` is a folder **outside** the repository
(`../../dvc-storage/mlops-task3`, i.e. next to the `MLOPS` folder). `dvc pull` from
it has been verified with the local cache and the artifacts moved away. It is a
separate copy, not yet a *shared* one: for a team or CI, point DVC at cloud
storage, e.g.

```bash
pip install "dvc[gdrive]"     # or dvc[s3], dvc[azure]
python -m dvc remote add -d -f storage gdrive://<folder-id>
python -m dvc push
```

The artifacts are small (under 10 KB each), so they are **also** committed to git
— CI runs the tests without a DVC remote.

---

## MLflow — Model Registry

The service **loads its model from the registry**. `docker compose up` does the whole
round trip: the one-off `register` step puts the model in MLflow (only if the registry
does not already hold these exact files), then the API downloads it back.

**Registration** (`scripts/register_model.py` → `src/registry.register_bundle`):
- one run in the experiment `olist-late-predictor`
- **Params**: C, class_weight, max_iter, n_features, requires_scaling, label definition
- **Metrics**: ROC-AUC, PR-AUC, recall/precision at 5%, lift — validation and test
- **Artifacts**: `serving/` = the four files the service loads, plus a manifest of their md5s
- **Registered model** `olist-late-predictor`: the sklearn model with its signature
- the new version gets the alias **`production`** and the stage **`Production`**
  (MLflow deprecated stages in 2.9 in favour of aliases; both are set, the service reads the alias)
- tags on the version: `model_version` (the hash `/health` reports) and each file's md5

**Loading** (`model.source: registry`, at API start-up):
1. look up `olist-late-predictor@production`
2. download that version's `serving/` files
3. check each md5 against the version's tags **and** against the DVC pointers in `models/`
4. only then load them

| Situation | What the API does |
|-----------|-------------------|
| Registry reachable, files check out | serves the registry version (`model_source` says which) |
| `MLFLOW_TRACKING_URI` not set, or MLflow down | serves `models/` (the same DVC-tracked bytes) and logs why |
| Registry serves different bytes than recorded | **refuses to start** |

```bash
# By hand (outside compose)
docker compose up mlflow -d
python scripts/register_model.py --if-missing   # reuse a version with these exact files
python scripts/register_model.py                # always a new version
python scripts/compare_models.py                # local files vs the version behind "production"
```

To promote another version, point the alias at it (MLflow UI → model → version → aliases)
and restart the API.

MLflow UI: `http://localhost:5000` — Models → `olist-late-predictor`.

---

## Testing

173 tests across 14 modules. Tests write to a temporary log directory — never to
`logs/predictions.jsonl` — and read the artifacts from `models/` (`MODEL_SOURCE=local`);
`test_registry.py` exercises the registry against a throw-away MLflow store. It needs
MLflow (in `requirements/base.txt`); where MLflow is missing those 5 tests are skipped.

```bash
# Run all tests
pytest

# Run a specific test file
pytest tests/test_parity.py

# Run with coverage
pytest --cov=src --cov=app --cov-report=term-missing
```

| Module | Tests | Covers |
|--------|-------|--------|
| `test_parity.py` | 6 | **service = notebooks**: features and scores of 314 real test orders; full test split → ROC-AUC 0.7107 (when the Task 2 artifacts are present) |
| `test_preprocessing.py` | 11 | haversine, build_derived, edge cases |
| `test_features.py` | 15 | transformers, 31 features, no NaN, no leakage, Infinity |
| `test_validation.py` | 25 | rules, fatal vs. problems, every real order passes |
| `test_ge_validation.py` | 15 | GE suite: ranges, sets, leakage, real batch, east-coast longitudes |
| `test_reference.py` | 7 | calibration, risk percentile, PSI bins, model guard |
| `test_alerting.py` | 8 | alert budget, rolling threshold, fallback |
| `test_predict.py` | 15 | model loading, outputs, threshold, calibration order |
| `test_pipeline.py` | 17 | end-to-end, logging every order, on_failure, GE on batches |
| `test_monitoring.py` | 11 | JSONL log, PSI, drift snapshot, test-log isolation |
| `test_evaluation.py` | 9 | calendar-day label, join by order_id, metrics |
| `test_artifacts.py` | 8 | model version hash, DVC pointers, sklearn version guard |
| `test_api.py` | 17 | all endpoints, 422s (incl. Infinity), batch limit from config |
| `test_registry.py` | 9 | register (alias, stage, tags), re-register, download + md5 check, tampering refused, same prediction from the registry, fallback |

`tests/fixtures/parity_orders.parquet` holds 314 real test-split orders (a random
sample plus the awkward cases: missing coordinates, missing category, east-coast
longitudes, coordinates outside Brazil) with the features Notebook 5 computed for
them. Rebuild it with `python scripts/make_parity_fixture.py`.

---

## CI/CD

GitHub Actions (`.github/workflows/task3-ci.yml` at the **repository root** — GitHub ignores workflows inside sub-folders; every step runs in `mlops-task3/`), Python 3.13:

```
push to main/develop
    ├── lint     → ruff check + ruff format --check
    ├── test     → pip install + pytest (depends on lint)
    └── build    → docker build (depends on test)
                   + push to ghcr.io/ziadsuleyman/olist-late-predictor:<sha> and :latest
                     (main only — so only an image whose lint and tests passed is published)
```

The push uses the workflow's own `GITHUB_TOKEN` (`packages: write`), no extra secret.
The package appears under the GitHub profile → Packages; it is private until its
visibility is changed there. Pull it with
`docker pull ghcr.io/ziadsuleyman/olist-late-predictor:latest`.

Pre-commit hooks (`.pre-commit-config.yaml`):
- Trailing whitespace, end-of-file fixer, YAML/JSON check
- No direct commits to `main`
- ruff lint + format

The Task 2 notebooks under `notebooks/` are byte-for-byte reference copies and are
excluded from ruff.

---

## Configuration

### `config/settings.yaml`

All runtime configuration in one file. Invalid values stop the service at startup.

| Section | Key settings |
|---------|-------------|
| `project` | name, version, description |
| `model` | source (registry/local), registry name, alias `production`, stage, fallback; artifact paths; label definition |
| `inference` | refit: false (true is refused), batch_max_size: 500 |
| `alerting` | budget: 0.05, window: 1000, min_window: 200 |
| `api` | title, docs_url, health_path |
| `logging` | level, format, file, rotation (10 MB × 5 files) |
| `monitoring` | prometheus, metrics_path, prediction_log_file, drift window, PSI thresholds |
| `validation` | on_failure (reject/flag), ge_on_batch, allowed states/payments, GE ranges |

### `.env` (secrets and overrides)

| Variable | Default | Purpose |
|----------|---------|---------|
| `DB_HOST` | localhost | PostgreSQL host |
| `DB_PORT` | 5433 | PostgreSQL port |
| `DB_NAME` | olist | Database name |
| `DB_USER` | olist | Database user |
| `DB_PASSWORD` | olist123 | Database password |
| `MLFLOW_TRACKING_URI` | http://localhost:5000 | MLflow server |
| `API_HOST` | 0.0.0.0 | API bind address |
| `API_PORT` | 8000 | API port |
| `LOG_LEVEL` | INFO | Logging level |
| `MODEL_SOURCE` | registry | `registry` or `local` (overrides `model.source`) |
| `PREDICTION_LOG_FILE` | logs/predictions.jsonl | Prediction log location |
| `SERVICE_LOG_FILE` | logs/service.log | Service log location |

---

## Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `scripts/build_serving_reference.py` | Fit calibration on validation, store score percentiles | `python scripts/build_serving_reference.py` |
| `scripts/make_parity_fixture.py` | Real test orders + notebook features for the parity tests | `python scripts/make_parity_fixture.py` |
| `scripts/register_model.py` | Register the serving bundle in MLflow, alias `production` | `python scripts/register_model.py --if-missing` |
| `scripts/compare_models.py` | Compare local artifacts vs MLflow registry | `python scripts/compare_models.py` |
| `scripts/analyze_predictions.py` | Prediction log: alert ratio, latency, PSI drift | `python scripts/analyze_predictions.py --all` |
| `scripts/evaluate_outcomes.py` | Score logged predictions against real deliveries | `python scripts/evaluate_outcomes.py --all` |

The first two read Task 2's artifacts (`../mlops-task2/artifacts` by default).

---

## Key Design Decisions

1. **No retraining at inference.** The saved fitted objects from notebooks 5 and 6
   are loaded once at startup and never refitted. `inference.refit: true` is refused.

2. **Parity is tested, not assumed.** 314 real test orders go through the service in
   CI and must produce Notebook 5's features and Notebook 6's scores exactly.

3. **Same environment as training.** Python 3.13 and scikit-learn 1.9.0, the versions
   the artifacts were pickled with; loading under another scikit-learn is an error.

4. **Budget, not threshold.** The model's score is class-balanced; it is calibrated
   into a probability, and "late" means "inside the 5% alert budget" — the rule
   Notebook 6 showed survives a change of period.

5. **Config-driven.** Every path, parameter, and threshold lives in
   `config/settings.yaml` or `.env`, and every setting there is used.

6. **Two-layer validation with a policy.** Per-order rules on every request; the
   Great Expectations suite on batches; `on_failure` decides reject or flag.

7. **Every prediction is scorable.** One log line per order, with `order_id` and the
   full input, so accuracy can be measured when the delivery date arrives.

8. **Drift = input drift.** PSI of the score distribution against the validation
   split — the one quantity that does not move just because the budget or the
   base rate does.

9. **The version is the content.** `model_version` is a hash of the artifacts; the
   same hash is tagged in MLflow, and each file's md5 matches its DVC pointer.

10. **Loaded from the registry, checked against DVC.** The API serves the version
    behind the `production` alias, and only after its bytes match both the registry's
    records and the DVC pointers. An unreachable registry degrades to the same bytes
    from `models/`; a registry that serves other bytes stops the service.

11. **Chronological splits.** Train < 2018-04, val Apr–May 2018, test Jun+ 2018.
    No random splitting — the model is evaluated on data from the future
    relative to training, as it would be in production.
