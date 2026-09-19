"""
Great Expectations data validation layer.

Defines the "order_input_suite" — the statistical contract a batch of incoming orders
must satisfy before it reaches the feature pipeline. It runs on every batch request
(src/pipeline.py, `validation.ge_on_batch`), on top of the per-order rules in
src/validation.py.

Why batches only: the expectations are about a population ("95% of freight values
inside the usual range", "at most 10% missing coordinates"). On a single order they
degenerate into hard limits, and one run costs ~0.9 s against ~37 ms for the prediction.

GE adds:
    - Column existence expectations
    - Plausible-range checks (ranges live in config/settings.yaml)
    - Null-rate expectations
    - Set-membership expectations
    - Leakage column absence

Compatible with Great Expectations >= 1.20 (ephemeral context API).

Usage:
    from src.ge_validation import validate_orders_ge

    result = validate_orders_ge(df)
    if not result["success"]:
        print(result["failed_expectations"])
"""

from __future__ import annotations

import threading

import great_expectations as gx
import great_expectations.expectations as gxe
import pandas as pd
from great_expectations.data_context.types.base import ProgressBarsConfig

from src.config import validation as val_cfg
from src.logger import log
from src.validation import BLOCKED_COLUMNS

# ── Column groups (from the training data contract) ─────────────────────────

NUMERIC_COLUMNS = list(val_cfg.numeric_ranges)

COORDINATE_COLUMNS = [
    "customer_lat",
    "customer_lng",
    "seller_lat",
    "seller_lng",
]

CATEGORICAL_COLUMNS = [
    "customer_state",
    "main_seller_state",
    "main_payment_type",
]

TIMESTAMP_COLUMNS = [
    "order_purchase_timestamp",
    "order_estimated_delivery_date",
    "shipping_limit_first",
]

# Built once, reused by every call (building costs ~1.4 s)
_validation_definition = None
_lock = threading.Lock()


def _build_suite(context) -> gx.ExpectationSuite:
    """
    Build and return the order_input expectation suite.

    All expectations are added to the suite object (GE 1.x pattern).
    """
    suite = context.suites.add(gx.ExpectationSuite(name=val_cfg.suite_name))

    # ── 1. Required columns exist ──────────────────────────────────────────
    required = (
        NUMERIC_COLUMNS
        + COORDINATE_COLUMNS
        + CATEGORICAL_COLUMNS
        + TIMESTAMP_COLUMNS
        + ["main_category"]
    )
    for col in required:
        suite.add_expectation(gxe.ExpectColumnToExist(column=col))

    # ── 2. Numeric ranges ──────────────────────────────────────────────────
    for col, (lo, hi) in val_cfg.numeric_ranges.items():
        suite.add_expectation(
            gxe.ExpectColumnValuesToBeBetween(column=col, min_value=lo, max_value=hi, mostly=0.95)
        )

    # ── 3. Coordinate ranges — Brazil's bounding box (allow more outliers) ──
    for col in COORDINATE_COLUMNS:
        lo, hi = val_cfg.lat_range if col.endswith("_lat") else val_cfg.lng_range
        suite.add_expectation(
            gxe.ExpectColumnValuesToBeBetween(column=col, min_value=lo, max_value=hi, mostly=0.90)
        )

    # ── 4. Categorical set membership ─────────────────────────────────────
    for col, allowed in [
        ("customer_state", val_cfg.allowed_states),
        ("main_seller_state", val_cfg.allowed_states),
        ("main_payment_type", val_cfg.allowed_payment_types),
    ]:
        suite.add_expectation(
            gxe.ExpectColumnValuesToBeInSet(column=col, value_set=allowed, mostly=0.99)
        )

    # ── 5. Null rate limits ────────────────────────────────────────────────
    # These columns must never be null
    for col in [
        "customer_state",
        "main_seller_state",
        "main_payment_type",
        "order_purchase_timestamp",
        "order_estimated_delivery_date",
    ]:
        suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column=col, mostly=1.0))

    # Numeric columns: at most 5% null
    for col in NUMERIC_COLUMNS:
        suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column=col, mostly=0.95))

    # Coordinates: up to 10% null
    for col in COORDINATE_COLUMNS:
        suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column=col, mostly=0.90))

    # ── 6. Positive counts ─────────────────────────────────────────────────
    for col in ["n_items", "n_products", "n_sellers", "n_payments"]:
        suite.add_expectation(
            gxe.ExpectColumnValuesToBeBetween(column=col, min_value=1, mostly=0.99)
        )

    return suite


def _get_validation_definition():
    """Ephemeral context + suite + validation definition, built on first use."""
    global _validation_definition
    if _validation_definition is None:
        context = gx.get_context(mode="ephemeral")
        # No tqdm bars in the service log
        context.variables.progress_bars = ProgressBarsConfig(
            globally=False, metric_calculations=False
        )
        data_source = context.data_sources.add_pandas("inference_pandas")
        data_asset = data_source.add_dataframe_asset(name="orders")
        batch_def = data_asset.add_batch_definition_whole_dataframe("full_batch")
        suite = _build_suite(context)
        _validation_definition = context.validation_definitions.add(
            gx.ValidationDefinition(name="order_input_validation", data=batch_def, suite=suite)
        )
        log.info("Great Expectations suite ready: %d expectations", len(suite.expectations))
    return _validation_definition


def warm_up() -> None:
    """Build the suite now (at service startup) instead of on the first batch."""
    with _lock:
        _get_validation_definition()


def validate_orders_ge(df: pd.DataFrame) -> dict:
    """
    Validate a DataFrame of orders using Great Expectations.

    Parameters
    ----------
    df : DataFrame with raw order columns.

    Returns
    -------
    dict with keys:
        success : bool
        statistics : dict (evaluated, passed, failed, success_percent)
        failed_expectations : list of dicts with failed expectation details
    """
    # ── Run validation ─────────────────────────────────────────────────────
    with _lock:
        results = _get_validation_definition().run(batch_parameters={"dataframe": df})

    # ── Process results ────────────────────────────────────────────────────
    success = results.success
    n_total = len(results.results)
    n_pass = sum(1 for r in results.results if r.success)
    n_fail = n_total - n_pass

    stats = {
        "evaluated": n_total,
        "passed": n_pass,
        "failed": n_fail,
        "success_percent": round(100 * n_pass / max(n_total, 1), 1),
    }

    failed = []
    for r in results.results:
        if not r.success:
            exp_kwargs = {k: v for k, v in r.expectation_config.kwargs.items() if k != "batch_id"}
            failed.append(
                {
                    "expectation_type": r.expectation_config.type,
                    "kwargs": exp_kwargs,
                }
            )

    # ── Leakage check (not a GE expectation — checked separately) ─────────
    leakage = [col for col in BLOCKED_COLUMNS if col in df.columns]
    if leakage:
        success = False
        failed.append(
            {
                "expectation_type": "leakage_check",
                "kwargs": {"blocked_columns": leakage},
            }
        )

    if not success:
        log.warning(
            "GE validation failed: %d/%d expectations passed, %d failed",
            n_pass,
            n_total,
            n_fail,
        )
    else:
        log.info("GE validation passed: %d/%d expectations OK", n_pass, n_total)

    return {
        "success": success,
        "statistics": stats,
        "failed_expectations": failed,
    }


def describe_failures(result: dict) -> list[str]:
    """One readable line per failed expectation, for API responses and the log."""
    lines = []
    for f in result["failed_expectations"]:
        kwargs = f["kwargs"]
        target = kwargs.get("column") or kwargs.get("blocked_columns")
        lines.append(f"batch check failed: {f['expectation_type']} on {target}")
    return lines
