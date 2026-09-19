"""
Integration tests for the FastAPI application.

Tests: health, model info, single predict, batch predict, bad payloads,
batch size limit from config, drift, metrics, docs.
Uses FastAPI's TestClient — no actual server needed.
"""

import json
import re

from src.config import inference as inference_cfg


class TestHealthEndpoint:
    def test_health_returns_200(self, api_client):
        """GET /health returns 200."""
        resp = api_client.get("/health")
        assert resp.status_code == 200

    def test_health_body(self, api_client):
        """Health response carries the artifact-hash model version and the service version."""
        data = api_client.get("/health").json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert re.fullmatch(r"[0-9a-f]{12}", data["model_version"])
        assert data["service_version"]


class TestModelInfoEndpoint:
    def test_model_info_returns_200(self, api_client):
        """GET /model/info returns 200."""
        resp = api_client.get("/model/info")
        assert resp.status_code == 200

    def test_model_info_body(self, api_client):
        """Model info has expected fields."""
        data = api_client.get("/model/info").json()
        assert data["model_type"] == "LogisticRegression"
        assert data["n_features"] == 31
        assert len(data["features"]) == 31
        assert set(data["artifacts_md5"]) == {"model", "transformers", "feature_list", "reference"}
        assert data["calibration"]["method"] == "platt"
        assert data["alerting"]["budget"] == 0.05


class TestPredictEndpoint:
    def test_predict_valid_order(self, api_client, sample_order):
        """POST /predict with a valid order returns 200 + prediction."""
        resp = api_client.post("/predict", json=sample_order)
        assert resp.status_code == 200
        data = resp.json()
        assert data["prediction"] in ("late", "on_time")
        assert 0 <= data["probability"] <= 1
        assert 0 <= data["risk_percentile"] <= 100
        assert data["order_id"] == sample_order["order_id"]
        assert "model_version" in data

    def test_probability_is_calibrated(self, api_client, sample_order):
        """The probability is far below the raw class-balanced score."""
        data = api_client.post("/predict", json=sample_order).json()
        assert data["probability"] < data["score"]
        assert data["probability"] < 0.2

    def test_predict_bad_payload_422(self, api_client):
        """POST /predict with bad data returns 422."""
        resp = api_client.post("/predict", json={"garbage": "data"})
        assert resp.status_code == 422

    def test_predict_empty_body_422(self, api_client):
        """POST /predict with an empty body returns 422."""
        resp = api_client.post("/predict", json={})
        assert resp.status_code == 422

    def test_rule_violation_lists_errors(self, api_client, sample_order):
        """A broken business rule returns 422 with the list of problems."""
        resp = api_client.post("/predict", json={**sample_order, "freight_total": -5})
        assert resp.status_code == 422
        assert any("freight_total" in e for e in resp.json()["errors"])

    def test_infinity_rejected_cleanly(self, api_client, sample_order):
        """Infinity is rejected with 422 — and echoing it back must not crash."""
        body = json.dumps({**sample_order, "freight_total": float("inf")})
        resp = api_client.post(
            "/predict", content=body, headers={"content-type": "application/json"}
        )
        assert resp.status_code == 422

    def test_order_id_optional(self, api_client, sample_order):
        """An order without order_id is still predicted (it just cannot be scored later)."""
        order = {k: v for k, v in sample_order.items() if k != "order_id"}
        resp = api_client.post("/predict", json=order)
        assert resp.status_code == 200
        assert resp.json()["order_id"] is None


class TestBatchPredictEndpoint:
    def test_batch_predict(self, api_client, sample_order):
        """POST /predict/batch with valid orders returns 200."""
        payload = {"orders": [sample_order, sample_order]}
        resp = api_client.post("/predict/batch", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2
        assert len(data["predictions"]) == 2

    def test_batch_empty_list_422(self, api_client):
        """POST /predict/batch with empty orders list returns 422."""
        resp = api_client.post("/predict/batch", json={"orders": []})
        assert resp.status_code == 422

    def test_batch_size_limit_from_config(self, api_client, sample_order):
        """One order more than inference.batch_max_size is rejected."""
        payload = {"orders": [sample_order] * (inference_cfg.batch_max_size + 1)}
        resp = api_client.post("/predict/batch", json=payload)
        assert resp.status_code == 422


class TestDriftEndpoint:
    def test_drift_structure(self, api_client, sample_order):
        """GET /drift reports the window; PSI waits for enough predictions."""
        api_client.post("/predict", json=sample_order)
        data = api_client.get("/drift").json()
        assert data["window_size"] == 1
        assert data["psi"] is None
        assert data["psi_status"] == "not enough data"


class TestMetricsEndpoint:
    def test_metrics_exposed(self, api_client, sample_order):
        """GET /metrics exposes the prediction counters and drift gauges."""
        api_client.post("/predict", json=sample_order)
        text = api_client.get("/metrics").text
        assert "prediction_requests_total" in text
        assert "prediction_alert_ratio" in text
        assert "prediction_score_psi" in text


class TestDocsEndpoint:
    def test_docs_available(self, api_client):
        """GET /docs returns 200 (interactive API docs)."""
        resp = api_client.get("/docs")
        assert resp.status_code == 200
