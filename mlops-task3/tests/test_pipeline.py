"""
Integration tests for src/pipeline.py

Tests: end-to-end single and batch prediction, rejection, logging of every order,
the on_failure policy, the Great Expectations batch check, the alert window.
"""

import pytest

from src.alerting import get_alert_policy
from src.config import alerting as alert_cfg
from src.config import validation as val_cfg
from src.pipeline import OrderValidationError, run_batch_inference, run_inference
from tests.helpers import read_log


class TestRunInference:
    def test_single_order_e2e(self, sample_order):
        """Full pipeline: raw order → prediction dict."""
        result = run_inference(sample_order)
        assert result["prediction"] in ("late", "on_time")
        assert 0 <= result["probability"] <= 1
        assert result["latency_ms"] > 0
        assert result["order_id"] == sample_order["order_id"]
        assert result["validation_warnings"] is None

    def test_bad_input_raises(self, sample_order_bad):
        """Bad input raises ValueError, not a crash."""
        with pytest.raises(ValueError, match="validation failed"):
            run_inference(sample_order_bad)

    def test_deterministic(self, sample_order):
        """Same input → same output (no randomness in the pipeline)."""
        r1 = run_inference(sample_order)
        r2 = run_inference(sample_order)
        assert r1["prediction"] == r2["prediction"]
        assert r1["probability"] == r2["probability"]

    def test_logged_with_order_id_and_input(self, sample_order, prediction_log):
        """The log line carries what is needed to score the prediction later."""
        run_inference(sample_order)
        (record,) = read_log(prediction_log)
        assert record["order_id"] == sample_order["order_id"]
        assert record["input"]["customer_state"] == sample_order["customer_state"]
        assert "order_id" not in record["input"]
        assert {"score", "probability", "alert_threshold", "model_version"} <= set(record)


class TestRunBatchInference:
    def test_batch_e2e(self, sample_order):
        """Batch prediction returns one result per order."""
        results = run_batch_inference([sample_order] * 3)
        assert len(results) == 3
        for r in results:
            assert r["prediction"] in ("late", "on_time")

    def test_batch_bad_input_raises(self, sample_order, sample_order_bad):
        """A bad order in the batch rejects the whole batch, naming its index."""
        with pytest.raises(OrderValidationError) as exc:
            run_batch_inference([sample_order, sample_order_bad])
        assert exc.value.errors[0]["index"] == 1

    def test_every_batch_order_logged(self, sample_order, prediction_log):
        """A batch of 3 writes 3 log lines, one per order_id."""
        orders = [{**sample_order, "order_id": f"id-{i}"} for i in range(3)]
        run_batch_inference(orders)
        records = read_log(prediction_log)
        assert [r["order_id"] for r in records] == ["id-0", "id-1", "id-2"]
        assert all(r["batch_size"] == 3 for r in records)

    def test_batch_feeds_drift_window(self, sample_order):
        """Batch predictions reach the drift window too."""
        from src.monitoring import drift_snapshot

        run_batch_inference([sample_order] * 4)
        assert drift_snapshot()["window_size"] == 4


class TestOnFailurePolicy:
    def test_reject_mode(self, sample_order, monkeypatch):
        """on_failure = reject: an unknown state rejects the order."""
        monkeypatch.setattr(val_cfg, "on_failure", "reject")
        with pytest.raises(OrderValidationError):
            run_inference({**sample_order, "customer_state": "XX"})

    def test_flag_mode_predicts_with_warnings(self, sample_order, monkeypatch, prediction_log):
        """on_failure = flag: the order is scored and the problem travels with it."""
        monkeypatch.setattr(val_cfg, "on_failure", "flag")
        result = run_inference({**sample_order, "customer_state": "XX"})
        assert result["prediction"] in ("late", "on_time")
        assert any("customer_state" in w for w in result["validation_warnings"])
        (record,) = read_log(prediction_log)
        assert record["validation_warnings"] == result["validation_warnings"]

    def test_flag_mode_still_rejects_fatal(self, sample_order, monkeypatch):
        """An unreadable date cannot be scored, whatever the policy."""
        monkeypatch.setattr(val_cfg, "on_failure", "flag")
        with pytest.raises(OrderValidationError):
            run_inference({**sample_order, "order_purchase_timestamp": "not-a-date"})


class TestGreatExpectationsOnBatch:
    FAILED = {
        "success": False,
        "statistics": {},
        "failed_expectations": [
            {"expectation_type": "expect_column_values_to_be_between", "kwargs": {"column": "x"}}
        ],
    }

    def test_ge_failure_rejects_batch(self, sample_order, monkeypatch):
        """A failed batch expectation rejects the batch under 'reject'."""
        monkeypatch.setattr("src.pipeline.validate_orders_ge", lambda df: self.FAILED)
        with pytest.raises(OrderValidationError) as exc:
            run_batch_inference([sample_order] * 2)
        assert "batch" in exc.value.errors[-1]

    def test_ge_failure_flags_batch(self, sample_order, monkeypatch):
        """Under 'flag' the batch is served and every order carries the batch warning."""
        monkeypatch.setattr("src.pipeline.validate_orders_ge", lambda df: self.FAILED)
        monkeypatch.setattr(val_cfg, "on_failure", "flag")
        results = run_batch_inference([sample_order] * 2)
        assert all(r["validation_warnings"] for r in results)

    def test_ge_not_run_on_single_order(self, sample_order, monkeypatch):
        """A single order skips the batch suite (it costs ~0.9 s)."""

        def boom(df):
            raise AssertionError("GE must not run on single orders")

        monkeypatch.setattr("src.pipeline.validate_orders_ge", boom)
        run_inference(sample_order)

    def test_ge_can_be_switched_off(self, sample_order, monkeypatch):
        """validation.ge_on_batch = false skips the suite on batches."""

        def boom(df):
            raise AssertionError("GE is switched off")

        monkeypatch.setattr("src.pipeline.validate_orders_ge", boom)
        monkeypatch.setattr(val_cfg, "ge_on_batch", False)
        run_batch_inference([sample_order] * 2)


class TestAlertWindow:
    def test_threshold_moves_to_rolling_window(self, sample_order):
        """After min_window orders, the threshold comes from recent scores."""
        policy = get_alert_policy()
        assert policy.source() == "validation split"
        run_batch_inference([sample_order] * alert_cfg.min_window)
        assert policy.source() == "rolling window"

    def test_request_does_not_set_its_own_threshold(self, sample_order):
        """The threshold is fixed before the request's scores join the window."""
        first = run_inference(sample_order)
        assert first["alert_threshold"] == round(get_alert_policy().fallback_threshold, 4)
