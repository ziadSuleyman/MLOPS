"""
Tests for src/monitoring.py

Covers: JSONL logging, PSI drift, drift snapshot, Prometheus recording,
and that tests write to a temporary log, never the production one.
"""

import numpy as np

from src.config import monitoring as mon_cfg
from src.monitoring import (
    compute_psi,
    drift_snapshot,
    get_prediction_log_path,
    log_predictions,
    psi_status,
    record_batch_size,
    record_error,
    record_predictions,
    record_request,
)
from src.reference import load_reference, psi_bins
from tests.helpers import read_log


def _result(score, prediction="on_time", probability=0.03):
    return {"score": score, "prediction": prediction, "probability": probability}


def _validation_like_scores(n, seed=0):
    """Scores drawn from the validation distribution (via its percentiles)."""
    pct = load_reference()["percentiles"]
    u = np.random.default_rng(seed).uniform(0, 100, n)
    return np.interp(u, np.arange(101), pct)


class TestLogPredictions:
    def test_one_line_per_record(self, tmp_path, monkeypatch):
        """log_predictions writes one valid JSONL line per order."""
        log_file = tmp_path / "predictions.jsonl"
        monkeypatch.setattr(mon_cfg, "prediction_log", log_file)

        log_predictions([{"order_id": "a", "score": 0.4}, {"order_id": "b", "score": 0.9}])

        records = read_log(log_file)
        assert [r["order_id"] for r in records] == ["a", "b"]
        assert all("timestamp" in r for r in records)

    def test_appends(self, tmp_path, monkeypatch):
        """Multiple calls append to the same file."""
        log_file = tmp_path / "predictions.jsonl"
        monkeypatch.setattr(mon_cfg, "prediction_log", log_file)
        for i in range(5):
            log_predictions([{"order_id": f"id_{i}"}])
        assert len(read_log(log_file)) == 5

    def test_creates_directory(self, tmp_path, monkeypatch):
        """log_predictions creates the parent directory if missing."""
        log_file = tmp_path / "subdir" / "deep" / "predictions.jsonl"
        monkeypatch.setattr(mon_cfg, "prediction_log", log_file)
        log_predictions([{"order_id": "x"}])
        assert log_file.exists()


class TestTestIsolation:
    def test_tests_use_a_temporary_log(self):
        """The suite writes to a temp directory, not logs/predictions.jsonl."""
        path = get_prediction_log_path()
        assert "olist-tests-" in path
        assert not path.replace("\\", "/").endswith("mlops-task3/logs/predictions.jsonl")


class TestPSI:
    def test_same_distribution_is_stable(self):
        """Scores from the validation distribution give a PSI near zero."""
        psi = compute_psi(_validation_like_scores(5000), *psi_bins())
        assert psi < 0.02
        assert psi_status(psi) == "ok"

    def test_shifted_distribution_alerts(self):
        """Only high scores (top decile) → PSI far above the alert threshold."""
        pct = load_reference()["percentiles"]
        high = np.linspace(pct[91], pct[99], 1000)
        psi = compute_psi(high, *psi_bins())
        assert psi > mon_cfg.psi_alert
        assert psi_status(psi) == "alert"

    def test_status_thresholds(self):
        """ok / warn / alert follow settings.yaml."""
        assert psi_status(None) == "not enough data"
        assert psi_status(mon_cfg.psi_warn - 0.01) == "ok"
        assert psi_status(mon_cfg.psi_warn) == "warn"
        assert psi_status(mon_cfg.psi_alert) == "alert"


class TestDriftSnapshot:
    def test_empty(self):
        """No predictions yet → nothing to report."""
        snap = drift_snapshot()
        assert snap["window_size"] == 0
        assert snap["psi"] is None

    def test_psi_after_min_window(self):
        """With enough predictions the PSI is computed and reported."""
        scores = _validation_like_scores(mon_cfg.drift_min_window)
        record_predictions([_result(s) for s in scores])
        snap = drift_snapshot()
        assert snap["window_size"] == mon_cfg.drift_min_window
        assert snap["psi"] is not None
        assert snap["psi_status"] in ("ok", "warn")

    def test_alert_ratio(self):
        """alert_ratio is the share flagged late in the window."""
        record_predictions([_result(0.9, "late")] + [_result(0.2)] * 3)
        assert drift_snapshot()["alert_ratio"] == 0.25


class TestRecording:
    def test_record_calls_do_not_raise(self):
        """Request, batch-size and error recording run without error."""
        record_request(0.015)
        record_batch_size(10)
        for kind in ("validation", "internal"):
            record_error(kind)
