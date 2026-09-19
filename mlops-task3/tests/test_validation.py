"""
Unit tests for src/validation.py

Tests: valid input passes, missing fields caught, bad types caught,
unknown states/payment types caught, fatal vs non-fatal problems,
every real order passes, leakage check.
"""

import pytest

from src.validation import check_no_leakage, check_order, validate_batch, validate_input
from tests.helpers import as_api_orders


class TestValidateInput:
    def test_valid_order_passes(self, sample_order):
        """A complete valid order passes validation."""
        is_valid, errors = validate_input(sample_order)
        assert is_valid, f"Valid order failed: {errors}"
        assert errors == []

    def test_missing_required_fields(self):
        """An empty dict fails with missing-fields error."""
        is_valid, errors = validate_input({})
        assert not is_valid
        assert any("Missing required fields" in e for e in errors)

    def test_unknown_customer_state(self, sample_order):
        """A state not in the allowed list is caught."""
        order = {**sample_order, "customer_state": "XX"}
        is_valid, errors = validate_input(order)
        assert not is_valid
        assert any("customer_state" in e for e in errors)

    def test_unknown_seller_state(self, sample_order):
        """Unknown seller state is caught."""
        order = {**sample_order, "main_seller_state": "ZZ"}
        is_valid, errors = validate_input(order)
        assert not is_valid
        assert any("main_seller_state" in e for e in errors)

    def test_unknown_payment_type(self, sample_order):
        """Unknown payment type is caught."""
        order = {**sample_order, "main_payment_type": "bitcoin"}
        is_valid, errors = validate_input(order)
        assert not is_valid
        assert any("main_payment_type" in e for e in errors)

    def test_negative_freight(self, sample_order):
        """Negative freight is caught."""
        order = {**sample_order, "freight_total": -5.0}
        is_valid, errors = validate_input(order)
        assert not is_valid
        assert any("freight_total" in e for e in errors)

    def test_negative_price(self, sample_order):
        """Negative price is caught."""
        order = {**sample_order, "items_price_total": -100}
        is_valid, errors = validate_input(order)
        assert not is_valid
        assert any("items_price_total" in e for e in errors)

    def test_bad_timestamp(self, sample_order):
        """Unparseable timestamp is caught."""
        order = {**sample_order, "order_purchase_timestamp": "not-a-date"}
        is_valid, errors = validate_input(order)
        assert not is_valid
        assert any("timestamp" in e for e in errors)

    def test_multiple_errors_collected(self):
        """Multiple problems produce multiple error messages."""
        order = {
            "customer_state": "XX",
            "freight_total": -10,
            "order_purchase_timestamp": "bad",
        }
        is_valid, errors = validate_input(order)
        assert not is_valid
        assert len(errors) >= 3

    def test_none_values_accepted(self, sample_order):
        """Optional fields can be None without failing."""
        order = {**sample_order, "customer_lat": None, "seller_lat": None}
        is_valid, errors = validate_input(order)
        assert is_valid, f"None values should be accepted: {errors}"


class TestValidateBatch:
    def test_all_valid(self, sample_order):
        """A batch of valid orders passes."""
        all_valid, errors = validate_batch([sample_order, sample_order])
        assert all_valid
        assert errors == []

    def test_one_bad_in_batch(self, sample_order):
        """One bad order in a batch → batch fails with the index."""
        bad = {**sample_order, "customer_state": "XX"}
        all_valid, errors = validate_batch([sample_order, bad, sample_order])
        assert not all_valid
        assert len(errors) == 1
        assert errors[0]["index"] == 1


class TestLeakageCheck:
    def test_clean_columns_pass(self):
        """A clean feature list raises nothing."""
        check_no_leakage(["distance_km", "promised_days", "freight_total"])

    def test_leakage_detected(self):
        """A blocked column raises ValueError."""
        with pytest.raises(ValueError, match="Leakage detected"):
            check_no_leakage(["distance_km", "order_delivered_customer_date"])

    def test_all_blocked_detected(self):
        """Every blocked column is caught."""
        blocked = [
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_status",
            "days_late",
            "is_late",
        ]
        for col in blocked:
            with pytest.raises(ValueError):
                check_no_leakage([col])


class TestRulesMatchRealData:
    def test_every_real_order_passes(self, parity_orders):
        """No rule rejects a real test-split order (checked on all 96,470 once)."""
        for order in as_api_orders(parity_orders):
            is_valid, errors = validate_input(order)
            assert is_valid, f"{order['order_id']}: {errors}"

    def test_zero_installments_accepted(self, sample_order):
        """installments_max = 0 occurs in the real data (2 orders) — not an error."""
        is_valid, errors = validate_input({**sample_order, "installments_max": 0})
        assert is_valid, errors

    def test_impossible_coordinates(self, sample_order):
        """A latitude beyond 90 is caught."""
        is_valid, errors = validate_input({**sample_order, "customer_lat": 95})
        assert not is_valid
        assert any("customer_lat" in e for e in errors)

    def test_promise_before_purchase(self, sample_order):
        """The promised date must come after the purchase (min in real data: 2 days)."""
        order = {**sample_order, "order_estimated_delivery_date": "2018-05-01"}
        is_valid, errors = validate_input(order)
        assert not is_valid
        assert any("after order_purchase_timestamp" in e for e in errors)


class TestFatalVersusProblems:
    def test_unknown_state_is_not_fatal(self, sample_order):
        """An unknown state can still be scored — a problem, not fatal."""
        fatal, problems = check_order({**sample_order, "customer_state": "XX"})
        assert fatal == []
        assert problems

    def test_bad_date_is_fatal(self, sample_order):
        fatal, _ = check_order({**sample_order, "order_purchase_timestamp": "nope"})
        assert fatal

    def test_missing_date_is_fatal(self, sample_order):
        fatal, _ = check_order({**sample_order, "order_estimated_delivery_date": None})
        assert fatal

    def test_timezone_is_fatal(self, sample_order):
        """Mixing a zoned and a naive timestamp would crash the date arithmetic."""
        order = {**sample_order, "order_purchase_timestamp": "2018-06-01T04:19:11Z"}
        fatal, _ = check_order(order)
        assert fatal

    def test_non_numeric_is_fatal(self, sample_order):
        fatal, _ = check_order({**sample_order, "freight_total": "cheap"})
        assert fatal

    def test_infinity_is_fatal(self, sample_order):
        fatal, _ = check_order({**sample_order, "freight_total": float("inf")})
        assert fatal
