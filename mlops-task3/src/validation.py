"""
Input validation — checks incoming orders before they reach the model.

Two kinds of problems, because `validation.on_failure` treats them differently:

    fatal    — the prediction cannot be computed (a date is missing or unreadable,
               a field is absent or not a number). Always rejected.
    problems — the order can be scored, but something is off (unknown state,
               negative price, impossible coordinates, delivery promised before
               the purchase). Rejected under "reject", attached as warnings under "flag".

These are per-order rules: every one of them holds for all 96,470 real orders
(checked against the Task 2 splits). The statistical checks — "95% of a batch
inside the usual range" — live in the Great Expectations suite (src/ge_validation.py).
"""

from __future__ import annotations

import math

import pandas as pd

from src.config import validation as val_cfg
from src.logger import log

# ── Required columns for inference ───────────────────────────────────────────
REQUIRED_COLUMNS = [
    "order_purchase_timestamp",
    "order_estimated_delivery_date",
    "customer_state",
    "main_seller_state",
    "customer_lat",
    "customer_lng",
    "seller_lat",
    "seller_lng",
    "freight_total",
    "freight_max",
    "items_price_total",
    "items_price_max",
    "payment_total",
    "installments_max",
    "n_payments",
    "n_items",
    "n_products",
    "n_sellers",
    "n_categories",
    "weight_g_total",
    "photos_avg",
    "shipping_limit_first",
    "main_payment_type",
]

# Columns that must NEVER appear in features (leakage)
BLOCKED_COLUMNS = [
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_status",
    "days_late",
    "is_late",
    "is_late_ts",
    "is_late_cal",
]

# Lowest value each numeric field can take in a real order
MINIMUMS = {
    "freight_total": 0,
    "freight_max": 0,
    "items_price_total": 0,
    "items_price_max": 0,
    "payment_total": 0,
    "weight_g_total": 0,
    "photos_avg": 0,
    "n_categories": 0,
    "installments_max": 0,  # 0 occurs in the real data (2 orders)
    "n_payments": 1,
    "n_items": 1,
    "n_products": 1,
    "n_sellers": 1,
}

COORDINATE_LIMITS = {
    "customer_lat": 90,
    "seller_lat": 90,
    "customer_lng": 180,
    "seller_lng": 180,
}

TIMESTAMP_COLUMNS = [
    "order_purchase_timestamp",
    "order_estimated_delivery_date",
    "shipping_limit_first",
]


def check_order(data: dict) -> tuple[list[str], list[str]]:
    """
    Check a single order.

    Returns
    -------
    (fatal, problems) — two lists of human-readable messages.
    """
    fatal: list[str] = []
    problems: list[str] = []

    # 1) Required columns present
    missing = [col for col in REQUIRED_COLUMNS if col not in data]
    if missing:
        fatal.append(f"Missing required fields: {missing}")

    # 2) Timestamps: the two dates the promise is built from must exist and parse
    parsed: dict[str, pd.Timestamp] = {}
    for col in TIMESTAMP_COLUMNS:
        value = data.get(col)
        if value is None:
            if col != "shipping_limit_first" and col in data:
                fatal.append(f"{col} is required")
            continue
        try:
            ts = pd.Timestamp(value)
        except (ValueError, TypeError):
            fatal.append(f"Cannot parse timestamp: {col}='{value}'")
            continue
        if ts is pd.NaT:
            fatal.append(f"Cannot parse timestamp: {col}='{value}'")
        elif ts.tzinfo is not None:
            fatal.append(f"{col} must be local time without a timezone, got '{value}'")
        else:
            parsed[col] = ts

    purchase = parsed.get("order_purchase_timestamp")
    promised = parsed.get("order_estimated_delivery_date")
    if purchase is not None and promised is not None and promised <= purchase:
        problems.append("order_estimated_delivery_date must be after order_purchase_timestamp")

    # 3) Categories
    for col, allowed in [
        ("customer_state", val_cfg.allowed_states),
        ("main_seller_state", val_cfg.allowed_states),
        ("main_payment_type", val_cfg.allowed_payment_types),
    ]:
        value = data.get(col)
        if value and value not in allowed:
            problems.append(f"Unknown {col}: '{value}'. Allowed: {allowed}")

    # 4) Numbers: numeric, finite, not below the real minimum
    for col, minimum in MINIMUMS.items():
        value = _as_number(data, col, fatal)
        if value is not None and value < minimum:
            problems.append(f"{col} must be >= {minimum}, got {value}")

    # 5) Coordinates must exist on Earth
    for col, limit in COORDINATE_LIMITS.items():
        value = _as_number(data, col, fatal)
        if value is not None and abs(value) > limit:
            problems.append(f"{col} must be between -{limit} and {limit}, got {value}")

    return fatal, problems


def validate_input(data: dict) -> tuple[bool, list[str]]:
    """
    Validate a single order payload.

    Returns
    -------
    (is_valid, errors) — if is_valid is False, errors lists what's wrong.
    """
    fatal, problems = check_order(data)
    errors = fatal + problems
    if errors:
        log.warning("Validation failed with %d error(s): %s", len(errors), errors)
    return not errors, errors


def validate_batch(items: list[dict]) -> tuple[bool, list[dict]]:
    """
    Validate a batch of orders.

    Returns
    -------
    (all_valid, per_item_errors) — per_item_errors[i] has {"index": i, "errors": [...]}
    """
    all_errors = []
    all_valid = True
    for i, item in enumerate(items):
        valid, errs = validate_input(item)
        if not valid:
            all_valid = False
            all_errors.append({"index": i, "errors": errs})
    return all_valid, all_errors


def check_no_leakage(columns: list[str]) -> None:
    """Raise if any blocked column made it into the feature set."""
    leaked = [col for col in columns if col in BLOCKED_COLUMNS]
    if leaked:
        raise ValueError(f"Leakage detected! Blocked columns in features: {leaked}")


# ── Helpers ──────────────────────────────────────────────────────────────────


def _as_number(data: dict, key: str, fatal: list[str]) -> float | None:
    """The field as a float, or None if absent/null. Non-numbers are fatal."""
    value = data.get(key)
    if value is None:
        return None
    try:
        number = float(value)
    except (ValueError, TypeError):
        fatal.append(f"{key} must be numeric, got '{value}'")
        return None
    if math.isnan(number):
        return None
    if math.isinf(number):
        fatal.append(f"{key} must be a finite number, got {value}")
        return None
    return number
