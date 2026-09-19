"""
Unit tests for src/predict.py

Tests: model loads, output keys and ranges, calibration, threshold behaviour, model info.
"""

import pandas as pd

from src.features import build_features
from src.predict import get_model_info, predict, predict_single


class TestLoadModel:
    def test_model_loads(self, model_bundle):
        """The model file loads without error."""
        assert model_bundle is not None

    def test_model_type(self, model_bundle):
        """The model is a LogisticRegression."""
        assert model_bundle["model_type"] == "LogisticRegression"

    def test_model_has_predict_proba(self, model_bundle):
        """The model object has a predict_proba method."""
        assert hasattr(model_bundle["model"], "predict_proba")

    def test_feature_count(self, model_bundle):
        """The model expects 31 features."""
        assert len(model_bundle["features"]) == 31

    def test_requires_scaling(self, model_bundle):
        """Logistic regression requires scaling."""
        assert model_bundle["requires_scaling"] is True


class TestPredict:
    def _features(self, *orders):
        return build_features(pd.DataFrame(list(orders)))

    def test_single_prediction_keys(self, sample_order, transformers):
        """predict_single returns every output field."""
        result = predict_single(self._features(sample_order), transformers)
        for key in [
            "prediction",
            "probability",
            "score",
            "risk_percentile",
            "alert_threshold",
            "model_version",
        ]:
            assert key in result

    def test_prediction_values(self, sample_order, transformers):
        """prediction is either 'late' or 'on_time'."""
        result = predict_single(self._features(sample_order), transformers)
        assert result["prediction"] in ("late", "on_time")

    def test_probability_range(self, sample_order, transformers):
        """Probability is between 0 and 1, percentile between 0 and 100."""
        result = predict_single(self._features(sample_order), transformers)
        assert 0 <= result["probability"] <= 1
        assert 0 <= result["risk_percentile"] <= 100

    def test_threshold_decides_prediction(self, sample_order, transformers):
        """The same score is 'late' under a low threshold and 'on_time' under a high one."""
        features = self._features(sample_order)
        assert predict_single(features, transformers, threshold=0.0)["prediction"] == "late"
        assert predict_single(features, transformers, threshold=1.01)["prediction"] == "on_time"

    def test_default_threshold_is_not_half(self, sample_order, transformers):
        """With an empty window the validation-split threshold applies, not 0.5."""
        result = predict_single(self._features(sample_order), transformers)
        assert result["alert_threshold"] > 0.7

    def test_batch_prediction_count(self, sample_order, transformers):
        """Batch prediction returns one result per order."""
        results = predict(self._features(*[sample_order] * 5), transformers)
        assert len(results) == 5

    def test_identical_inputs_same_output(self, sample_order, transformers):
        """Two identical orders get the same prediction."""
        results = predict(self._features(sample_order, sample_order), transformers)
        assert results[0]["probability"] == results[1]["probability"]

    def test_probability_order_follows_score(self, parity_orders, transformers):
        """Calibration is monotone: a higher score never gets a lower probability."""
        from tests.helpers import RAW_COLUMNS

        results = predict(build_features(parity_orders[RAW_COLUMNS]), transformers)
        ranked = sorted(results, key=lambda r: r["score"])
        probs = [r["probability"] for r in ranked]
        assert probs == sorted(probs)


class TestModelInfo:
    def test_model_info_keys(self):
        """get_model_info returns all expected keys."""
        info = get_model_info()
        expected = [
            "model_type",
            "model_version",
            "service_version",
            "n_features",
            "features",
            "hyperparameters",
            "label_definition",
            "requires_scaling",
            "metrics",
            "artifacts_md5",
            "calibration",
            "alerting",
        ]
        for key in expected:
            assert key in info, f"Missing key: {key}"

    def test_model_info_values(self):
        """Spot-check model info values."""
        info = get_model_info()
        assert info["model_type"] == "LogisticRegression"
        assert info["n_features"] == 31
        assert info["requires_scaling"] is True
        assert info["alerting"]["threshold_source"] == "validation split"
