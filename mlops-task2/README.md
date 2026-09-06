# MLOps Task 2 — From Tables to Notebooks

Qafza MLOps Training 2026/2027 · Task 2.
Six notebooks, each doing one job, each reading the artifact of the step before it
and writing its own for the step after it.

**The problem:** predict whether an order will be delivered late or on time.

## Prerequisites

The Task 1 database must be running:

```bash
cd ../mlops-task1 && docker compose up -d
```

Connection: `postgresql+psycopg://olist:olist123@localhost:5433/olist`

## Start JupyterLab

```bash
python -m jupyterlab --notebook-dir="C:/Users/Rings-1/MLOPS/mlops-task2"
```

## The notebooks

| # | Notebook | Reads | Writes |
|---|---|---|---|
| 1 | `01_read_and_join.ipynb` | the database | `01_ml_table.parquet` |
| 2 | `02_labels.ipynb` | `01_ml_table.parquet` | `02_labeled.parquet` |
| 3 | `03_split.ipynb` | `02_labeled.parquet` | `03_train/val/test.parquet` |
| 4 | `04_eda.ipynb` | `03_train.parquet` | `figures/`, `04_findings.md` |
| 5 | `05_features.ipynb` | the three splits | feature tables, fitted transformers, feature list |
| 6 | `06_train_evaluate.ipynb` | feature tables | trained model, results summary |

Run them in order from a clean kernel. Never open the test split before notebook 6.

## Layout

```
notebooks/   the six notebooks, numbered
artifacts/   every file a notebook hands to the next one
figures/     charts saved by notebook 4
```

## Leakage rules

Columns that do NOT exist at prediction time and must never become features:

- `order_approved_at`
- `order_delivered_carrier_date`
- `order_delivered_customer_date` (this is the answer)
- every column of `order_reviews` (91% of reviews are written after delivery)
- the final `order_status`

Notebook 1 keeps the delivery timestamps because notebook 2 needs them to build the
label. They are dropped in notebook 5, before any feature is built.
