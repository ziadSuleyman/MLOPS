# Architecture — Olist Late Delivery Predictor

## System Overview

The system is an **inference-only** service. It does not train or retrain models.
The fitted objects (transformers + model) were produced by the Task 2 notebook
pipeline and are loaded at startup as immutable artifacts. One more artifact is
built offline in Task 3 — the serving reference (calibration + the validation
score distribution) — and is equally read-only at serving time.

At startup the service takes all four from the **MLflow registry** (the version behind
the alias `production`), checks their md5s against the version's tags and the DVC
pointers, and only then loads them. If MLflow is not configured or not reachable it
loads the same DVC-tracked files from `models/`; if MLflow serves different bytes it
refuses to start (`src/registry.py`).

## Component Map

```
┌──────────────────────────────────────────────────────────────────┐
│                          FastAPI (app/)                           │
│                                                                  │
│  ┌────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐ ┌────────────┐  │
│  │/health │ │ /predict │ │ /predict │ │ /drift │ │ /metrics   │  │
│  │/model/ │ │          │ │ /batch   │ │        │ │(Prometheus)│  │
│  │ info   │ └────┬─────┘ └────┬─────┘ └────────┘ └────────────┘  │
│  └────────┘      │            │                                  │
└──────────────────┼────────────┼──────────────────────────────────┘
                   ▼            ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Pipeline (src/pipeline.py)                     │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐  │
│  │ validation   │  │ preprocessing│  │ features               │  │
│  │ (every order)│→ │              │→ │ (05_transformers)      │  │
│  │ ge_validation│  └──────────────┘  └───────────┬────────────┘  │
│  │ (batches)    │                                ▼               │
│  └──────────────┘  ┌──────────────┐  ┌────────────────────────┐  │
│                    │ alerting     │→ │ predict                │  │
│  ┌──────────────┐  │ (budget →    │  │ (06_model → score,     │  │
│  │ monitoring   │◀─│  threshold)  │  │  reference → probability│  │
│  │              │  └──────────────┘  └────────────────────────┘  │
│  └──────┬───────┘                                                │
└─────────┼────────────────────────────────────────────────────────┘
          ▼
   ┌───────────────┐ ┌───────────┐ ┌──────────┐      ┌──────────────┐
   │ JSONL log     │ │Prometheus │ │ Log file │ ───▶ │ evaluation   │
   │ (order_id,    │ │ + PSI     │ │ (debug)  │      │ (+ orders    │
   │  input)       │ │ gauges    │ │          │      │   table)     │
   └───────────────┘ └───────────┘ └──────────┘      └──────────────┘
```

## Module Responsibilities

| Module | Responsibility | Artifacts |
|--------|---------------|-----------|
| `src/config.py` | Load `settings.yaml` + `.env`, expose typed singletons, refuse invalid settings | None |
| `src/logger.py` | Rotating file handler + console handler | None |
| `src/artifacts.py` | Artifact hashes → `model_version`; DVC pointer check; scikit-learn version guard | All four |
| `src/preprocessing.py` | Pure row-level transforms: haversine, derived columns | None |
| `src/features.py` | Apply trained transformers: TE, OHE, impute → 31 features | `05_transformers.joblib` |
| `src/validation.py` | Per-order rules; fatal problems vs. warnings | None |
| `src/ge_validation.py` | Great Expectations suite (70 checks) for batches | None |
| `src/reference.py` | Calibration, risk percentile, PSI bins | `07_serving_reference.json` |
| `src/alerting.py` | Alert budget → rolling score threshold | `07_serving_reference.json` (fallback) |
| `src/predict.py` | Load model, scale, score → probability → late/on_time | `06_model.joblib` |
| `src/pipeline.py` | Orchestrate: validate → features → predict → record | All above |
| `src/monitoring.py` | Prometheus, PSI drift window, JSONL prediction log | None |
| `src/evaluation.py` | Score logged predictions against real deliveries | None |
| `src/registry.py` | Register the bundle in MLflow; pick and verify the source at startup | All four |
| `app/main.py` | FastAPI routes, startup checks, error handling | None |
| `app/schemas.py` | Pydantic request/response models | None |

## Data Flow

1. **Client** sends a JSON order to `POST /predict` (optionally with `order_id`)
2. **Pydantic** validates the schema: types only, and no NaN/Infinity
3. **validation.py** checks the per-order rules. Fatal problems reject; other problems
   reject or travel as warnings, depending on `validation.on_failure`
4. For batches, **ge_validation.py** runs the Great Expectations suite, same policy
5. **preprocessing.py** computes 23 derived features (haversine, ratios, flags)
6. **features.py** applies the fitted pipeline:
   - Group rare categories → target encoding
   - One-hot encode payment type
   - Infinite values → missing
   - Impute missing values with training medians
   - Result: exactly 31 features in the contract order
7. **predict.py** scales (StandardScaler), runs `predict_proba` → score; calibrates the
   score into a probability; compares the score with the alert threshold
8. **alerting.py** fixed that threshold before the request, and adds the new scores to
   its window afterwards
9. **monitoring.py** writes one JSONL line per order and updates Prometheus and the
   drift window
10. **FastAPI** returns `{order_id, prediction, probability, score, risk_percentile,
    alert_threshold, model_version, latency_ms, validation_warnings}`

## Artifacts

All four are DVC-tracked and hashed into `model_version`.

### `05_transformers.joblib` (Task 2)

Contains a dict with keys:
- `kept_categories` — categories not grouped into "rare"
- `target_encoders` — smoothed target encoding maps for 3 categorical features
- `onehot_payment` — one-hot encoder for payment type
- `imputer` — training median values for imputation
- `feature_columns` — the 31-feature contract
- `scaler` — StandardScaler fit on training data
- `metadata` — training data statistics

### `06_model.joblib` (Task 2)

Contains a dict with keys:
- `model` — sklearn LogisticRegression object
- `features` — list of 31 feature names
- `model_type` — "LogisticRegression"
- `hyperparameters` — {C: 0.3, class_weight: "balanced", max_iter: 3000}
- `metrics` — validation and test set performance
- `label_definition` — "calendar-day rule"
- `requires_scaling` — True

Both were pickled with scikit-learn 1.9.0; `src/artifacts.load_pickle` refuses to
load them under any other version.

### `07_serving_reference.json` (Task 3, `scripts/build_serving_reference.py`)

Plain JSON, built from the **validation** split:
- `model_md5` — the model it belongs to; the service refuses a mismatch
- `calibration` — Platt scaling coefficients
- `score_percentiles` — percentiles 0..100 of the validation scores
- `evaluation` — how calibration and the alert policy behave on the test split

## Error Handling

| Error Type | HTTP Code | Prometheus Label | Example |
|-----------|-----------|-----------------|---------|
| Schema validation | 422 | `validation` | Missing required field, Infinity |
| Fatal rule | 422 | `validation` | Unparseable purchase date |
| Rule problem (`reject`) | 422 | `validation` | Unknown state "XX" |
| Rule problem (`flag`) | 200 + `validation_warnings` | — | Unknown state "XX" |
| Model error | 500 | `internal` | Corrupt model file |
| Unexpected error | 500 | `internal` | Out of memory |

Startup fails (the service does not come up) on: `inference.refit: true`, an unknown
`on_failure` or `model.source` value, registry files whose md5 differs from what was
recorded for that version, a scikit-learn version mismatch, a serving reference built for
another model, or a leakage column in the feature contract.

## Configuration Hierarchy

```
.env (secrets — passwords, URIs; log-file overrides)
  ↓  overrides
config/settings.yaml (structure — paths, budgets, thresholds, lists)
  ↓  read by
src/config.py (typed singletons, validated)
  ↓  imported by
every other module
```

Environment variables in `.env` override their `settings.yaml` counterparts
where applicable (`LOG_LEVEL`, `API_PORT`, `PREDICTION_LOG_FILE`, `SERVICE_LOG_FILE`).
