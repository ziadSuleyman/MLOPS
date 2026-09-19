"""
Shared fixtures for all tests.

Provides: a real sample order, a bad order, loaded transformers and model,
the FastAPI test client, and the parity fixture (real test-split orders with
the features Notebook 5 computed for them).
"""

import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

# Ensure project root is on sys.path so `src` and `app` are importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Tests must never write into the production logs (logs/predictions.jsonl).
# Set before any `src` import — src.config reads these once.
TEST_LOG_DIR = Path(tempfile.mkdtemp(prefix="olist-tests-"))
os.environ["PREDICTION_LOG_FILE"] = str(TEST_LOG_DIR / "predictions.jsonl")
os.environ["SERVICE_LOG_FILE"] = str(TEST_LOG_DIR / "service.log")
# Tests read the artifacts from models/, whether or not an MLflow server happens to run;
# tests/test_registry.py exercises the registry path against a throw-away store.
os.environ["MODEL_SOURCE"] = "local"

FIXTURES = Path(__file__).resolve().parent / "fixtures"
SAMPLE_ORDER_ID = "8a9be36ffd78382f9ac518945e909636"


@pytest.fixture(scope="session")
def sample_order() -> dict:
    """
    A real order from the test split (order_id 8a9be36ffd78382f9ac518945e909636):
    the earliest complete order there. Minas Gerais customer, São Paulo seller,
    delivered on time. It is also in the parity fixture, so its features are
    checked against Notebook 5's.
    """
    return {
        "order_id": SAMPLE_ORDER_ID,
        "order_purchase_timestamp": "2018-06-01 04:19:11",
        "order_estimated_delivery_date": "2018-07-05 00:00:00",
        "shipping_limit_first": "2018-06-11 04:30:37",
        "customer_state": "MG",
        "main_seller_state": "SP",
        "customer_lat": -19.51753103934325,
        "customer_lng": -42.6116577517932,
        "seller_lat": -23.652442526788203,
        "seller_lng": -46.75549168747648,
        "freight_total": 18.43,
        "freight_max": 18.43,
        "items_price_total": 78.0,
        "items_price_max": 78.0,
        "payment_total": 96.43,
        "installments_max": 4.0,
        "n_payments": 1.0,
        "main_payment_type": "credit_card",
        "n_items": 1.0,
        "n_products": 1.0,
        "n_sellers": 1.0,
        "n_categories": 1.0,
        "weight_g_total": 250.0,
        "photos_avg": 4.0,
        "main_category": "relogios_presentes",
    }


@pytest.fixture(scope="session")
def sample_order_bad() -> dict:
    """An intentionally bad payload — missing required fields, wrong types."""
    return {
        "order_purchase_timestamp": "not-a-date",
        "customer_state": "XX",
        "freight_total": -10,
    }


@pytest.fixture(scope="session")
def parity_orders() -> pd.DataFrame:
    """Real test-split orders + Notebook 5 features (columns prefixed nb__)."""
    return pd.read_parquet(FIXTURES / "parity_orders.parquet")


@pytest.fixture(scope="session")
def transformers():
    """Load the fitted transformers once for the test session."""
    from src.features import load_transformers

    return load_transformers()


@pytest.fixture(scope="session")
def model_bundle():
    """Load the model bundle once for the test session."""
    from src.predict import load_model

    return load_model()


@pytest.fixture(scope="session")
def api_client():
    """FastAPI TestClient — runs the full app without a server."""
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        yield client


@pytest.fixture(autouse=True)
def _fresh_windows():
    """Every test starts with an empty alert window and drift window."""
    from src.alerting import get_alert_policy
    from src.monitoring import reset_windows

    get_alert_policy().reset()
    reset_windows()
    yield


@pytest.fixture
def prediction_log() -> Path:
    """The (test) prediction log, emptied for this test."""
    path = TEST_LOG_DIR / "predictions.jsonl"
    path.unlink(missing_ok=True)
    return path
