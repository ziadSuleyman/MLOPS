# MLOps — Olist Late Delivery Prediction

Qafza MLOps Training 2026/2027. All tasks work on one problem: predict whether an
order from the [Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
will be delivered late or on time.

| Folder | Task | Contents |
|---|---|---|
| [`mlops-task1`](mlops-task1) | Data → PostgreSQL | Postgres 16 in Docker, schema, ingestion script |
| [`mlops-task2`](mlops-task2) | Tables → notebooks | Six notebooks: join, labels, split, EDA, features, training |
| [`mlops-task3`](mlops-task3) | Notebooks → production | FastAPI service, MLflow registry, DVC, tests, CI |

Each folder has its own README with the full details.

## Requirements

- Docker and Docker Compose
- Python 3.13
- The Olist CSVs from Kaggle, unzipped into `mlops-task1/data/` (not committed)

## Task 1 — database

```bash
cd mlops-task1
docker compose up -d
pip install pandas sqlalchemy "psycopg[binary]" jupyterlab
python ingest.py
```

Postgres runs on `localhost:5433` (user `olist`, password `olist123`, database `olist`).

## Task 2 — notebooks

Needs the Task 1 database running.

```bash
pip install pandas pyarrow sqlalchemy "psycopg[binary]" scikit-learn matplotlib seaborn joblib jupyterlab
python -m jupyterlab --notebook-dir=mlops-task2
```

Run `notebooks/01` to `06` in order. Each one writes its output to `artifacts/`.

## Task 3 — API

```bash
cd mlops-task3
cp .env.example .env
docker compose up --build
```

- API: http://localhost:8000 (interactive docs at `/docs`)
- MLflow: http://localhost:5000

Task 3 starts its own Postgres on port 5433, so stop the Task 1 database first, or
move it: `DB_PORT=5434 docker compose up --build`. If port 8000 is taken:
`API_PORT=8010 docker compose up --build`.

Tests:

```bash
cd mlops-task3
pip install -r requirements/dev.txt
pytest
```

## CI

`.github/workflows/task3-ci.yml` runs on every push or pull request that touches
`mlops-task3`: lint, tests, then a Docker build. Pushes to `main` also publish the
image to `ghcr.io/ziadsuleyman/olist-late-predictor`.
