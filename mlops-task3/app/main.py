"""
FastAPI application — Olist Late Delivery Predictor.

Routes:
    GET  /health         → service health and model status
    GET  /model/info     → model metadata, artifact hashes, calibration, alert policy
    POST /predict        → predict for a single order
    POST /predict/batch  → predict for a batch of orders
    GET  /drift          → score drift (PSI) over recent predictions
    GET  /metrics        → Prometheus metrics

The prediction routes are plain `def`: the work is synchronous pandas/scikit-learn,
so FastAPI runs them in its thread pool instead of blocking the event loop.
"""

from __future__ import annotations

import math
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse

from app.schemas import (
    BatchInput,
    BatchOutput,
    DriftOutput,
    ErrorOutput,
    HealthOutput,
    ModelInfoOutput,
    OrderInput,
    PredictionOutput,
)
from src.artifacts import dvc_mismatches, model_version
from src.config import api as api_cfg
from src.config import monitoring as mon_cfg
from src.config import project_version
from src.config import validation as val_cfg
from src.features import get_feature_columns, load_transformers
from src.ge_validation import warm_up
from src.logger import log
from src.monitoring import drift_snapshot, record_error
from src.pipeline import OrderValidationError, run_batch_inference, run_inference
from src.predict import get_model_info, load_model
from src.reference import load_reference
from src.validation import check_no_leakage


# ── Startup / shutdown ──────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and check every artifact once at startup."""
    log.info("Starting %s v%s", api_cfg.title, project_version)
    try:
        load_transformers()
        load_model()
        load_reference()
        check_no_leakage(get_feature_columns())
        for problem in dvc_mismatches():
            log.warning("DVC: %s", problem)
        if val_cfg.ge_on_batch:
            warm_up()
        log.info("Serving model %s", model_version())
    except Exception:
        log.exception("Failed to load model at startup")
        raise
    yield
    log.info("Shutting down")


app = FastAPI(
    title=api_cfg.title,
    version=project_version,
    docs_url=api_cfg.docs_url,
    lifespan=lifespan,
)


def _json_safe(value):
    """Replace what JSON cannot carry (NaN, Infinity, exception objects) with text."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


@app.exception_handler(RequestValidationError)
async def request_validation_handler(request: Request, exc: RequestValidationError):
    """Schema errors echo the bad input back — which must not crash on Infinity."""
    record_error(error_type="validation")
    return JSONResponse(status_code=422, content={"detail": _json_safe(exc.errors())})


def _validation_error(e: OrderValidationError) -> JSONResponse:
    record_error(error_type="validation")
    return JSONResponse(status_code=422, content={"detail": e.message, "errors": e.errors})


def _internal_error(e: Exception, what: str) -> JSONResponse:
    record_error(error_type="internal")
    log.exception("%s failed", what)
    return JSONResponse(status_code=500, content={"detail": f"Internal error: {e!s}"})


# ── Routes ──────────────────────────────────────────────────────────────────


@app.get(api_cfg.health_path, response_model=HealthOutput)
def health():
    """Health check — is the service up and the model loaded?"""
    try:
        load_model()
        return HealthOutput(
            status="healthy",
            model_loaded=True,
            model_version=model_version(),
            service_version=project_version,
        )
    except Exception:
        return HealthOutput(
            status="unhealthy",
            model_loaded=False,
            model_version="unknown",
            service_version=project_version,
        )


@app.get("/model/info", response_model=ModelInfoOutput)
def model_info():
    """Model metadata — type, version, features, hyperparameters, metrics."""
    return get_model_info()


@app.post(
    "/predict",
    response_model=PredictionOutput,
    responses={422: {"model": ErrorOutput}},
)
def predict_single(order: OrderInput):
    """Predict for a single order."""
    try:
        return PredictionOutput(**run_inference(order.model_dump()))
    except OrderValidationError as e:
        return _validation_error(e)
    except Exception as e:
        return _internal_error(e, "Prediction")


@app.post(
    "/predict/batch",
    response_model=BatchOutput,
    responses={422: {"model": ErrorOutput}},
)
def predict_batch(batch: BatchInput):
    """Predict for a batch of orders (max `inference.batch_max_size`)."""
    try:
        results = run_batch_inference([o.model_dump() for o in batch.orders])
        predictions = [PredictionOutput(**r) for r in results]
        return BatchOutput(predictions=predictions, count=len(predictions))
    except OrderValidationError as e:
        return _validation_error(e)
    except Exception as e:
        return _internal_error(e, "Batch prediction")


@app.get("/drift", response_model=DriftOutput)
def drift():
    """PSI of recent scores against the validation split, plus alert ratio."""
    return drift_snapshot()


# ── Prometheus metrics endpoint ─────────────────────────────────────────────
if mon_cfg.enable_prometheus:

    @app.get(mon_cfg.metrics_path, response_class=PlainTextResponse)
    def metrics():
        """Prometheus metrics endpoint."""
        from prometheus_client import generate_latest

        return PlainTextResponse(
            content=generate_latest().decode("utf-8"),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )
