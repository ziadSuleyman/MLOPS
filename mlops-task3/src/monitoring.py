"""
Monitoring — Prometheus metrics, drift, and the prediction log.

Prometheus:
    prediction_requests_total{prediction}  — orders scored, by outcome
    prediction_latency_seconds             — request latency (single or batch)
    prediction_errors_total{error_type}    — errors (validation, internal)
    prediction_batch_size                  — histogram of batch sizes
    prediction_validation_warnings_total   — orders served with warnings (on_failure: flag)
    prediction_score_psi                   — PSI of recent scores vs the validation split
    prediction_alert_ratio                 — share of recent orders flagged "late"
    prediction_mean_probability            — mean calibrated probability of recent orders

Why PSI and not "late-prediction ratio vs 6.77%": the share flagged is fixed by the
alert budget, so it cannot drift; and the true late rate moved 7.95% → 3.61% between
periods, so any comparison with one fixed rate alarms forever. What drifts is the
input, and the score distribution summarises it.

JSONL log:
    logs/predictions.jsonl — one line per order, single or batch, with the order_id
    and the full input, so outcomes can be joined later (src/evaluation.py).
"""

from __future__ import annotations

import json
import threading
from collections import deque
from datetime import UTC, datetime

import numpy as np

from src.config import monitoring as mon_cfg
from src.logger import log
from src.reference import psi_bins

# ── Rolling window of recent predictions (drift) ─────────────────────────────
_recent: deque[tuple[float, bool, float]] = deque(maxlen=mon_cfg.drift_window)
_lock = threading.Lock()

# ── Prometheus metrics (lazy init — only if enabled) ────────────────────────
_metrics_initialized = False
_REQUEST_COUNT = None
_LATENCY_HISTOGRAM = None
_ERROR_COUNT = None
_BATCH_SIZE = None
_WARNING_COUNT = None
_PSI = None
_ALERT_RATIO = None
_MEAN_PROBABILITY = None


def _init_metrics():
    """Initialize Prometheus metrics. Called once on first use."""
    global _metrics_initialized, _REQUEST_COUNT, _LATENCY_HISTOGRAM, _ERROR_COUNT
    global _BATCH_SIZE, _WARNING_COUNT, _PSI, _ALERT_RATIO, _MEAN_PROBABILITY

    if _metrics_initialized:
        return

    if not mon_cfg.enable_prometheus:
        _metrics_initialized = True
        return

    from prometheus_client import Counter, Gauge, Histogram

    _REQUEST_COUNT = Counter(
        "prediction_requests_total",
        "Orders scored, by outcome",
        ["prediction"],  # label: "late" or "on_time"
    )
    _LATENCY_HISTOGRAM = Histogram(
        "prediction_latency_seconds",
        "Request latency in seconds (a batch counts once)",
        buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    )
    _ERROR_COUNT = Counter(
        "prediction_errors_total",
        "Total number of prediction errors",
        ["error_type"],  # label: "validation", "internal"
    )
    _BATCH_SIZE = Histogram(
        "prediction_batch_size",
        "Size of batch prediction requests",
        buckets=(1, 5, 10, 25, 50, 100, 250, 500),
    )
    _WARNING_COUNT = Counter(
        "prediction_validation_warnings_total",
        "Orders served despite failing validation rules (on_failure: flag)",
    )
    _PSI = Gauge(
        "prediction_score_psi",
        "PSI of the recent score distribution vs the validation split (drift)",
    )
    _ALERT_RATIO = Gauge(
        "prediction_alert_ratio",
        "Share of recent orders flagged late (should sit near the alert budget)",
    )
    _MEAN_PROBABILITY = Gauge(
        "prediction_mean_probability",
        "Mean calibrated late probability of recent orders",
    )

    _metrics_initialized = True
    log.info("Prometheus metrics initialized")


# ── Drift ────────────────────────────────────────────────────────────────────


def compute_psi(scores: np.ndarray, edges: np.ndarray, expected: np.ndarray) -> float:
    """Population Stability Index of `scores` against the expected bin shares."""
    counts, _ = np.histogram(np.asarray(scores, dtype=float), bins=edges)
    actual = np.clip(counts / max(counts.sum(), 1), 1e-6, None)
    expected = np.clip(expected, 1e-6, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def psi_status(psi: float | None) -> str:
    """ok / warn / alert, from the thresholds in settings.yaml."""
    if psi is None:
        return "not enough data"
    if psi >= mon_cfg.psi_alert:
        return "alert"
    if psi >= mon_cfg.psi_warn:
        return "warn"
    return "ok"


def drift_snapshot() -> dict:
    """Current drift picture over the rolling window."""
    with _lock:
        recent = list(_recent)

    n = len(recent)
    snapshot = {
        "window_size": n,
        "min_window": mon_cfg.drift_min_window,
        "psi": None,
        "psi_status": psi_status(None),
        "alert_ratio": None,
        "mean_probability": None,
    }
    if n == 0:
        return snapshot

    scores = np.array([r[0] for r in recent])
    snapshot["alert_ratio"] = round(float(np.mean([r[1] for r in recent])), 4)
    snapshot["mean_probability"] = round(float(np.mean([r[2] for r in recent])), 4)
    if n >= mon_cfg.drift_min_window:
        psi = compute_psi(scores, *psi_bins())
        snapshot["psi"] = round(psi, 4)
        snapshot["psi_status"] = psi_status(psi)
    return snapshot


def reset_windows() -> None:
    """Clear the rolling window (tests, or after a model change)."""
    with _lock:
        _recent.clear()


# ── Recording ────────────────────────────────────────────────────────────────


def record_request(latency_s: float) -> None:
    """Record the latency of one request (single order or whole batch)."""
    _init_metrics()
    if _LATENCY_HISTOGRAM is not None:
        _LATENCY_HISTOGRAM.observe(latency_s)


def record_predictions(results: list[dict]) -> None:
    """Record served predictions: counters, rolling window, drift gauges."""
    _init_metrics()

    with _lock:
        for r in results:
            _recent.append((r["score"], r["prediction"] == "late", r["probability"]))

    if _REQUEST_COUNT is None:
        return

    for r in results:
        _REQUEST_COUNT.labels(prediction=r["prediction"]).inc()
    n_warned = sum(1 for r in results if r.get("validation_warnings"))
    if n_warned:
        _WARNING_COUNT.inc(n_warned)

    snap = drift_snapshot()
    _ALERT_RATIO.set(snap["alert_ratio"])
    _MEAN_PROBABILITY.set(snap["mean_probability"])
    if snap["psi"] is not None:
        _PSI.set(snap["psi"])


def record_batch_size(size: int) -> None:
    """Record the size of a batch request."""
    _init_metrics()
    if _BATCH_SIZE is not None:
        _BATCH_SIZE.observe(size)


def record_error(error_type: str = "internal") -> None:
    """Record a prediction error in Prometheus."""
    _init_metrics()
    if _ERROR_COUNT is not None:
        _ERROR_COUNT.labels(error_type=error_type).inc()


# ── Prediction log (JSONL file) ─────────────────────────────────────────────


def log_predictions(records: list[dict]) -> None:
    """
    Append one line per order to the JSONL log.

    Each record carries the order_id and the input, so that when the real delivery
    date arrives the prediction can be scored (scripts/evaluate_outcomes.py).
    """
    timestamp = datetime.now(UTC).isoformat()
    lines = [json.dumps({"timestamp": timestamp, **rec}, default=str) for rec in records]

    try:
        mon_cfg.prediction_log.parent.mkdir(parents=True, exist_ok=True)
        with open(mon_cfg.prediction_log, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except OSError:
        log.warning("Could not write prediction log to %s", mon_cfg.prediction_log)


def get_prediction_log_path() -> str:
    """Return the path to the JSONL prediction log."""
    return str(mon_cfg.prediction_log)
