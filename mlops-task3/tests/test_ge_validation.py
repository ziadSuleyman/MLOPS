"""
Tests for the Great Expectations validation layer (src/ge_validation.py).

Covers: valid data passes, invalid ranges caught, unknown categories caught,
leakage columns caught, null handling, real batches pass, east-coast longitudes.
"""

import pandas as pd

from src.ge_validation import (
    _get_validation_definition,
    describe_failures,
    validate_orders_ge,
)
from tests.helpers import RAW_COLUMNS


def _make_order(**overrides) -> dict:
    """A real order (test split, 8a9be36ffd78382f9ac518945e909636), with optional overrides."""
    base = {
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
    base.update(overrides)
    return base


class TestGEValidOrderPasses:
    def test_single_valid_order(self):
        """A valid order passes all GE expectations."""
        df = pd.DataFrame([_make_order()])
        result = validate_orders_ge(df)
        assert result["success"], f"Failed: {result['failed_expectations']}"
        assert result["statistics"]["failed"] == 0

    def test_batch_valid_orders(self):
        """A batch of valid orders all pass."""
        df = pd.DataFrame([_make_order() for _ in range(5)])
        result = validate_orders_ge(df)
        assert result["success"]
        assert result["statistics"]["success_percent"] == 100.0


class TestGERangeViolations:
    def test_negative_freight_caught(self):
        """Negative freight_total violates the range expectation."""
        df = pd.DataFrame([_make_order(freight_total=-10)])
        result = validate_orders_ge(df)
        assert not result["success"]
        types = [f["expectation_type"] for f in result["failed_expectations"]]
        assert "expect_column_values_to_be_between" in types

    def test_extreme_weight_caught(self):
        """weight_g_total > 50000 violates the range expectation."""
        df = pd.DataFrame([_make_order(weight_g_total=100_000)])
        result = validate_orders_ge(df)
        assert not result["success"]


class TestGECategoryViolations:
    def test_unknown_state_caught(self):
        """An unknown customer_state violates set membership."""
        df = pd.DataFrame([_make_order(customer_state="XX")])
        result = validate_orders_ge(df)
        assert not result["success"]
        types = [f["expectation_type"] for f in result["failed_expectations"]]
        assert "expect_column_values_to_be_in_set" in types

    def test_unknown_payment_type_caught(self):
        """An unknown payment type is caught."""
        df = pd.DataFrame([_make_order(main_payment_type="bitcoin")])
        result = validate_orders_ge(df)
        assert not result["success"]


class TestGELeakageDetection:
    def test_leakage_column_caught(self):
        """A blocked column in the DataFrame triggers the leakage check."""
        order = _make_order()
        order["is_late"] = 1
        df = pd.DataFrame([order])
        result = validate_orders_ge(df)
        assert not result["success"]
        types = [f["expectation_type"] for f in result["failed_expectations"]]
        assert "leakage_check" in types

    def test_multiple_leakage_columns(self):
        """Multiple blocked columns are all reported."""
        order = _make_order()
        order["is_late"] = 1
        order["order_status"] = "delivered"
        df = pd.DataFrame([order])
        result = validate_orders_ge(df)
        leaked = [
            f for f in result["failed_expectations"] if f["expectation_type"] == "leakage_check"
        ]
        assert len(leaked) == 1
        assert len(leaked[0]["kwargs"]["blocked_columns"]) == 2


class TestGENullHandling:
    def test_null_coordinates_pass_in_batch(self):
        """Null lat/lng is tolerated when < 10% of the batch (mostly=0.90)."""
        # 9 valid rows + 1 row with null coords = 10% null → passes mostly=0.90
        orders = [_make_order() for _ in range(9)]
        orders.append(_make_order(customer_lat=None, customer_lng=None))
        df = pd.DataFrame(orders)
        result = validate_orders_ge(df)
        assert result["success"], f"Failed: {result['failed_expectations']}"

    def test_statistics_structure(self):
        """The result has the expected statistics keys."""
        df = pd.DataFrame([_make_order()])
        result = validate_orders_ge(df)
        stats = result["statistics"]
        assert "evaluated" in stats
        assert "passed" in stats
        assert "failed" in stats
        assert "success_percent" in stats


class TestGERealData:
    def test_real_batch_passes(self, parity_orders):
        """314 real test-split orders, awkward ones included, pass the suite."""
        result = validate_orders_ge(parity_orders[RAW_COLUMNS[1:]])
        assert result["success"], result["failed_expectations"]

    def test_east_coast_longitude_passes(self):
        """João Pessoa sits at -34.86: east of the old -35 edge, still Brazil."""
        df = pd.DataFrame([_make_order(customer_lng=-34.86, customer_lat=-7.12)] * 5)
        assert validate_orders_ge(df)["success"]

    def test_outside_brazil_caught(self):
        """A whole batch in Portugal (a real Olist geolocation error) fails."""
        df = pd.DataFrame([_make_order(customer_lat=41.15, customer_lng=-8.58)] * 5)
        assert not validate_orders_ge(df)["success"]


class TestGESuite:
    def test_suite_size(self):
        """The suite holds 70 expectations."""
        assert len(_get_validation_definition().suite.expectations) == 70

    def test_describe_failures(self):
        """Failures become one readable line each."""
        result = validate_orders_ge(pd.DataFrame([_make_order(weight_g_total=100_000)]))
        lines = describe_failures(result)
        assert any("weight_g_total" in line for line in lines)
