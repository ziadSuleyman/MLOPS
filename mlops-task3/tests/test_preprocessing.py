"""
Unit tests for src/preprocessing.py

Tests: haversine distance, build_derived column count and names,
edge cases (missing coordinates, zero distance).
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import build_derived, haversine_km

# ── haversine_km ─────────────────────────────────────────────────────────────


class TestHaversine:
    def test_known_distance_sp_to_rj(self):
        """São Paulo (-23.55, -46.63) to Rio (-22.91, -43.17) ≈ 358 km."""
        d = haversine_km(
            pd.Series([-23.55]),
            pd.Series([-46.63]),
            pd.Series([-22.91]),
            pd.Series([-43.17]),
        )
        assert 350 < d.iloc[0] < 370, f"Expected ~358 km, got {d.iloc[0]:.0f}"

    def test_same_point_is_zero(self):
        """Distance from a point to itself is 0."""
        d = haversine_km(
            pd.Series([-23.55]),
            pd.Series([-46.63]),
            pd.Series([-23.55]),
            pd.Series([-46.63]),
        )
        assert d.iloc[0] == pytest.approx(0.0, abs=0.01)

    def test_nan_propagates(self):
        """If one coordinate is NaN, the distance is NaN."""
        d = haversine_km(
            pd.Series([np.nan]),
            pd.Series([-46.63]),
            pd.Series([-22.91]),
            pd.Series([-43.17]),
        )
        assert pd.isna(d.iloc[0])


# ── build_derived ────────────────────────────────────────────────────────────


class TestBuildDerived:
    def test_output_column_count(self, sample_order):
        """build_derived must produce exactly 23 columns."""
        df = pd.DataFrame([sample_order])
        result = build_derived(df)
        assert result.shape[1] == 23, f"Expected 23 columns, got {result.shape[1]}"

    def test_output_has_key_features(self, sample_order):
        """The most important derived features must be present."""
        df = pd.DataFrame([sample_order])
        result = build_derived(df)
        expected = [
            "promised_days",
            "distance_km",
            "promise_per_100km",
            "same_state",
            "customer_geo_missing",
            "freight_ratio",
        ]
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"

    def test_promised_days_positive(self, sample_order):
        """The promise is always in the future → positive days."""
        df = pd.DataFrame([sample_order])
        result = build_derived(df)
        assert result["promised_days"].iloc[0] > 0

    def test_same_state_false_for_different_states(self, sample_order):
        """MG customer, SP seller → same_state = 0."""
        df = pd.DataFrame([sample_order])
        result = build_derived(df)
        assert result["same_state"].iloc[0] == 0

    def test_same_state_true_when_equal(self, sample_order):
        """If both are SP → same_state = 1."""
        order = {**sample_order, "customer_state": "SP", "main_seller_state": "SP"}
        df = pd.DataFrame([order])
        result = build_derived(df)
        assert result["same_state"].iloc[0] == 1

    def test_geo_missing_indicator(self, sample_order):
        """When coordinates are present, geo_missing = 0."""
        df = pd.DataFrame([sample_order])
        result = build_derived(df)
        assert result["customer_geo_missing"].iloc[0] == 0
        assert result["seller_geo_missing"].iloc[0] == 0

    def test_geo_missing_when_null(self, sample_order):
        """When coordinates are None, geo_missing = 1."""
        order = {**sample_order, "customer_lat": None, "customer_lng": None}
        df = pd.DataFrame([order])
        result = build_derived(df)
        assert result["customer_geo_missing"].iloc[0] == 1

    def test_single_row_index_preserved(self, sample_order):
        """The output index matches the input index."""
        df = pd.DataFrame([sample_order], index=[42])
        result = build_derived(df)
        assert result.index.tolist() == [42]
