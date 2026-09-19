"""
Unit tests for src/features.py

Tests: feature count, column names match the contract, no NaN after pipeline,
no leakage columns, transformers load correctly, infinite inputs handled.
"""

import json
import logging

import numpy as np
import pandas as pd

from src.config import model as model_cfg
from src.features import build_features, get_feature_columns


class TestLoadTransformers:
    def test_transformers_load(self, transformers):
        """Transformers file loads without error."""
        assert transformers is not None

    def test_transformers_keys(self, transformers):
        """All expected keys are present."""
        expected = [
            "kept_categories",
            "target_encoders",
            "onehot_payment",
            "imputer",
            "feature_columns",
            "scaler",
            "metadata",
        ]
        for key in expected:
            assert key in transformers, f"Missing key: {key}"

    def test_feature_columns_count(self):
        """Feature list has exactly 31 features."""
        cols = get_feature_columns()
        assert len(cols) == 31, f"Expected 31, got {len(cols)}"

    def test_feature_columns_match_json(self):
        """Feature columns match 05_feature_list.json exactly."""
        with open(model_cfg.feature_list_path, encoding="utf-8") as f:
            expected = json.load(f)["features"]
        assert get_feature_columns() == expected


class TestBuildFeatures:
    def test_output_shape(self, sample_order):
        """build_features returns exactly 31 columns for one order."""
        df = pd.DataFrame([sample_order])
        features = build_features(df)
        assert features.shape == (1, 31), f"Expected (1, 31), got {features.shape}"

    def test_output_columns_match(self, sample_order):
        """Column names and order match the feature contract."""
        df = pd.DataFrame([sample_order])
        features = build_features(df)
        assert list(features.columns) == get_feature_columns()

    def test_no_nan(self, sample_order):
        """No NaN values after the full feature pipeline."""
        df = pd.DataFrame([sample_order])
        features = build_features(df)
        nan_count = int(features.isna().sum().sum())
        assert nan_count == 0, f"Found {nan_count} NaN values"

    def test_no_inf(self, sample_order):
        """No infinite values after the full feature pipeline."""
        df = pd.DataFrame([sample_order])
        features = build_features(df)
        assert np.isfinite(features.to_numpy()).all(), "Found infinite values"

    def test_no_leakage_columns(self, sample_order):
        """Blocked columns must never appear in the output."""
        blocked = [
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_status",
            "days_late",
            "is_late",
            "is_late_ts",
            "is_late_cal",
        ]
        df = pd.DataFrame([sample_order])
        features = build_features(df)
        leaked = [col for col in features.columns if col in blocked]
        assert leaked == [], f"Leakage detected: {leaked}"

    def test_target_encoding_in_range(self, sample_order):
        """Target-encoded columns should be between 0 and 1 (they are rates)."""
        df = pd.DataFrame([sample_order])
        features = build_features(df)
        for col in ["customer_state_te", "main_seller_state_te", "main_category_te"]:
            val = features[col].iloc[0]
            assert 0 <= val <= 1, f"{col} = {val}, expected [0, 1]"

    def test_batch_preserves_order(self, sample_order):
        """Two identical orders produce identical features."""
        df = pd.DataFrame([sample_order, sample_order])
        features = build_features(df)
        assert features.iloc[0].equals(features.iloc[1])

    def test_unknown_category_handled(self, sample_order):
        """A category never seen in training → 'rare' → still produces features."""
        order = {**sample_order, "main_category": "never_seen_before_xyz"}
        df = pd.DataFrame([order])
        features = build_features(df)
        assert features.shape == (1, 31)
        assert features.isna().sum().sum() == 0

    def test_unknown_state_handled(self, sample_order):
        """An unknown state → target encoding falls back to the prior."""
        order = {**sample_order, "customer_state": "ZZ"}
        df = pd.DataFrame([order])
        features = build_features(df)
        assert features.shape == (1, 31)
        assert features.isna().sum().sum() == 0


class TestInfiniteValues:
    def test_inf_treated_as_missing(self, sample_order):
        """An infinite input becomes the train median instead of crashing the imputer."""
        inf = build_features(pd.DataFrame([{**sample_order, "freight_total": np.inf}]))
        missing = build_features(pd.DataFrame([{**sample_order, "freight_total": None}]))
        assert np.isfinite(inf.to_numpy()).all()
        pd.testing.assert_frame_equal(inf, missing)

    def test_inf_is_counted(self, sample_order, caplog):
        """The warning reports the real number of infinite values (it used to be negative)."""
        logger = logging.getLogger("olist")
        logger.addHandler(caplog.handler)
        try:
            build_features(pd.DataFrame([{**sample_order, "freight_total": np.inf}]))
        finally:
            logger.removeHandler(caplog.handler)
        # freight_total and freight_ratio are both infinite
        assert "Infinite values found: 2" in caplog.text
