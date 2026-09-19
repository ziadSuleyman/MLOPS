"""
End-to-end inference pipeline.

Input: a raw order (dict) or a list of them.
Output: prediction, calibrated probability, score, risk percentile, model version.

This is the single entry point that the API and the CLI both call.
The pipeline must reproduce exactly what the notebook produced on the same input
(tests/test_parity.py checks it on real test-split orders).

Steps:
    1. validate    — per-order rules; plus the Great Expectations suite on batches
    2. features    — the fitted Task 2 transformers, never refitted
    3. predict     — score, calibrate, flag against the alert budget
    4. record      — one JSONL line per order (with order_id), Prometheus, drift window
"""

from __future__ import annotations

import hashlib
import json
import time

import pandas as pd

from src.alerting import get_alert_policy
from src.config import validation as val_cfg
from src.features import build_features, load_transformers
from src.ge_validation import describe_failures, validate_orders_ge
from src.logger import log
from src.monitoring import log_predictions, record_batch_size, record_predictions, record_request
from src.predict import predict
from src.validation import check_order


class OrderValidationError(ValueError):
    """Raised when orders are rejected; `errors` holds the details."""

    def __init__(self, message: str, errors: list):
        self.message = message
        self.errors = errors
        super().__init__(f"{message}: {errors}")


def _input_hash(order_input: dict) -> str:
    return hashlib.md5(json.dumps(order_input, sort_keys=True, default=str).encode()).hexdigest()[
        :12
    ]


def _infer(orders: list[dict], is_batch: bool) -> list[dict]:
    start = time.perf_counter()
    order_ids = [o.get("order_id") for o in orders]
    inputs = [{k: v for k, v in o.items() if k != "order_id"} for o in orders]

    # 1) Validate
    checks = [check_order(inp) for inp in inputs]
    fatal = [(i, f) for i, (f, _) in enumerate(checks) if f]
    problems = [(i, p) for i, (_, p) in enumerate(checks) if p]

    batch_problems: list[str] = []
    if is_batch and val_cfg.ge_on_batch and not fatal:
        ge_result = validate_orders_ge(pd.DataFrame(inputs))
        if not ge_result["success"]:
            batch_problems = describe_failures(ge_result)

    reject = val_cfg.on_failure == "reject" and (problems or batch_problems)
    if fatal or reject:
        if is_batch:
            errors: list = [
                {"index": i, "errors": checks[i][0] + checks[i][1]}
                for i in sorted({i for i, _ in fatal + problems})
            ]
            if batch_problems:
                errors.append({"batch": batch_problems})
            raise OrderValidationError("Batch validation failed", errors)
        raise OrderValidationError("Input validation failed", checks[0][0] + checks[0][1])

    # 2) Features — fitted transformers only
    transformers = load_transformers()
    features = build_features(pd.DataFrame(inputs))

    # 3) Predict — the threshold is fixed before this request's scores join the window
    policy = get_alert_policy()
    results = predict(features, transformers, threshold=policy.threshold())
    policy.observe([r["score"] for r in results])

    # 4) Record
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    records = []
    for i, r in enumerate(results):
        warnings = checks[i][1] + batch_problems
        r["latency_ms"] = latency_ms
        r["order_id"] = order_ids[i]
        r["validation_warnings"] = warnings or None
        records.append(
            {
                "order_id": order_ids[i],
                "input_hash": _input_hash(inputs[i]),
                "prediction": r["prediction"],
                "probability": r["probability"],
                "score": r["score"],
                "risk_percentile": r["risk_percentile"],
                "alert_threshold": r["alert_threshold"],
                "model_version": r["model_version"],
                "latency_ms": latency_ms,
                "batch_size": len(orders),
                "validation_warnings": r["validation_warnings"],
                "input": inputs[i],
            }
        )

    log_predictions(records)
    record_request(latency_ms / 1000)
    record_predictions(results)
    if is_batch:
        record_batch_size(len(orders))

    log.info(
        "%s: %d order(s)  late=%d  threshold=%.4f  latency=%.1fms  model=%s",
        "Batch prediction" if is_batch else "Prediction",
        len(results),
        sum(1 for r in results if r["prediction"] == "late"),
        results[0]["alert_threshold"],
        latency_ms,
        results[0]["model_version"],
    )
    return results


def run_inference(order: dict) -> dict:
    """
    Predict for a single order.

    Parameters
    ----------
    order : dict with the raw order fields (and optionally "order_id").

    Returns
    -------
    dict with keys: prediction, probability, score, risk_percentile, alert_threshold,
    model_version, latency_ms, order_id, validation_warnings.

    Raises
    ------
    OrderValidationError (a ValueError) if the order is rejected.
    """
    return _infer([order], is_batch=False)[0]


def run_batch_inference(orders: list[dict]) -> list[dict]:
    """
    Predict for a batch of orders. Every order is validated and logged individually;
    the Great Expectations suite checks the batch as a whole.

    Returns
    -------
    List of prediction dicts, in input order.
    """
    return _infer(orders, is_batch=True)
