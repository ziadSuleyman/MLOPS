# Deployment Guide

## Prerequisites

- Docker and Docker Compose installed
- The Task 1 database has been initialized (or use the bundled `db` service)
- `.env` file configured (copy from `.env.example`)

## Option 1: Docker Compose (recommended)

```bash
# Start all services
docker compose up --build -d

# Check logs
docker compose logs -f api

# Stop
docker compose down
```

Services:

| Service | URL | Notes |
|---------|-----|-------|
| API | http://localhost:8000 | Swagger at /docs |
| MLflow | http://localhost:5000 | Experiment tracking (v3.16.1, same as the client) |
| PostgreSQL | localhost:5433 | Olist database |

## Option 2: Local Development

Python **3.13** (the version the model was trained under).

```bash
# 1. Virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements/dev.txt

# 2. Environment
cp .env.example .env
# Edit .env if the DB is at a different host

# 3. Run
uvicorn app.main:app --reload --port 8000
```

## Option 3: Production Deployment

### Build the Docker image

```bash
docker build -t olist-predictor:1.1.0 .
```

### Run standalone

```bash
docker run -d \
  --name olist-api \
  -p 8000:8000 \
  -e DB_HOST=your-db-host \
  -e DB_PORT=5432 \
  -e DB_NAME=olist \
  -e DB_USER=olist \
  -e DB_PASSWORD=secure-password \
  -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
  -v $(pwd)/logs:/app/logs \
  olist-predictor:1.1.0
```

Run **one** worker per container: the alert window and the drift window live in
process memory, so several workers would each keep their own.

### Health Check

```bash
curl http://localhost:8000/health
# {"status":"healthy","model_loaded":true,"model_version":"f9fcc48ef9ef","service_version":"1.1.0"}
```

## Post-Deployment Checklist

- [ ] Verify `/health` returns `healthy`, and its `model_version` matches the MLflow tag
- [ ] Check the startup log has no `DVC:` warnings (artifacts = DVC-recorded versions)
- [ ] Send a test prediction to `/predict` (the README example is a real order)
- [ ] Check `/metrics` endpoint is accessible
- [ ] Verify `logs/predictions.jsonl` is being written, with `order_id`
- [ ] Register the model in MLflow: `python scripts/register_model.py`
- [ ] Set up Prometheus to scrape `/metrics`
- [ ] Alert on `prediction_score_psi` ≥ 0.25 (drift) and on `prediction_alert_ratio`
      far from the 5% budget
- [ ] Schedule `python scripts/evaluate_outcomes.py` to score predictions once orders
      are delivered

## Updating the Model

1. Place the new `.joblib` files in `models/`
2. Rebuild the serving reference: `python scripts/build_serving_reference.py`
   (the service refuses a reference built for another model)
3. Rebuild the parity fixture: `python scripts/make_parity_fixture.py`
4. Track with DVC and push: `python -m dvc add models/*.joblib models/*.json` then `python -m dvc push`
5. Bump `project.version` in `config/settings.yaml` if the API changed
   (`model_version` changes by itself — it is the artifacts' hash)
6. Register in MLflow: `python scripts/register_model.py --run-name "v1.2.0"`
7. Rebuild the Docker image: `docker compose up --build -d`
8. Verify with `/health` and `/model/info`
