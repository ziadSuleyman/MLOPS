"""
Tests for src/alerting.py — the alert budget.

Covers: fallback before the window fills, the rolling quantile after, the budget
being respected, and that the fallback is the validation split's threshold.
"""

import numpy as np
import pytest

from src.alerting import AlertPolicy, get_alert_policy
from src.config import alerting as alert_cfg
from src.reference import load_reference, reference_threshold


def _policy(budget=0.05, window=1000, min_window=200, fallback=0.75):
    return AlertPolicy(budget, window, min_window, fallback)


class TestThreshold:
    def test_fallback_until_min_window(self):
        """Below min_window the fallback threshold applies."""
        policy = _policy()
        policy.observe(np.linspace(0, 1, 199))
        assert policy.threshold() == 0.75
        assert policy.source() == "validation split"

    def test_rolling_quantile_after_min_window(self):
        """From min_window on, the threshold is the (1 - budget) quantile."""
        policy = _policy()
        scores = np.linspace(0, 1, 1000)
        policy.observe(scores)
        assert policy.threshold() == pytest.approx(np.quantile(scores, 0.95))
        assert policy.source() == "rolling window"

    def test_budget_respected_under_shift(self):
        """Even when every score rises, about `budget` of orders are flagged."""
        policy = _policy()
        rng = np.random.default_rng(0)
        policy.observe(rng.uniform(0.2, 0.6, 1000))
        shifted = rng.uniform(0.5, 0.9, 5000)
        flagged = []
        for s in shifted:
            flagged.append(s >= policy.threshold())
            policy.observe([s])
        assert np.mean(flagged[-2000:]) == pytest.approx(0.05, abs=0.01)

    def test_window_forgets_old_scores(self):
        """The window holds only the most recent `window` scores."""
        policy = _policy(window=300)
        policy.observe(np.zeros(1000))
        policy.observe(np.ones(300))
        assert len(policy) == 300
        assert policy.threshold() == 1.0

    def test_reset(self):
        """reset() empties the window."""
        policy = _policy()
        policy.observe(np.ones(500))
        policy.reset()
        assert len(policy) == 0

    def test_invalid_budget(self):
        """A budget outside (0, 1) is refused."""
        with pytest.raises(ValueError):
            _policy(budget=1.5)


class TestServicePolicy:
    def test_uses_config(self):
        """The service policy is built from settings.yaml."""
        policy = get_alert_policy()
        assert policy.budget == alert_cfg.budget
        assert policy.min_window == alert_cfg.min_window

    def test_fallback_is_validation_threshold(self):
        """The fallback flags exactly `budget` of the validation orders."""
        pct = load_reference()["percentiles"]
        assert get_alert_policy().fallback_threshold == reference_threshold(alert_cfg.budget)
        assert reference_threshold(0.05) == pytest.approx(pct[95])
