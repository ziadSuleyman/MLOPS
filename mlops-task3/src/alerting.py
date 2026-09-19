"""
Alert policy — which orders are flagged "late".

Notebook 6 measured it: a numeric threshold does not survive the move to a new
period, because the late rate keeps moving (7.95% train → 5.53% val → 3.61% test).
The threshold tuned on validation flags 8.7% of test orders instead of 5%.
What carries over is the budget: "flag the riskiest 5%".

So the threshold is the (1 - budget) quantile of the most recent scores. Until the
window holds `min_window` scores, the validation-split threshold stands in.
"""

from __future__ import annotations

import threading
from collections import deque

import numpy as np

from src.config import alerting as alert_cfg
from src.reference import reference_threshold


class AlertPolicy:
    def __init__(self, budget: float, window: int, min_window: int, fallback_threshold: float):
        if not 0 < budget < 1:
            raise ValueError(f"alert budget must be between 0 and 1, got {budget}")
        self.budget = budget
        self.min_window = min_window
        self.fallback_threshold = fallback_threshold
        self._recent: deque[float] = deque(maxlen=window)
        self._lock = threading.Lock()

    def threshold(self) -> float:
        """Current score threshold: scores at or above it are flagged."""
        with self._lock:
            if len(self._recent) < self.min_window:
                return self.fallback_threshold
            return float(np.quantile(np.fromiter(self._recent, float), 1 - self.budget))

    def source(self) -> str:
        """Where the current threshold comes from."""
        with self._lock:
            n = len(self._recent)
        return "rolling window" if n >= self.min_window else "validation split"

    def observe(self, scores: np.ndarray) -> None:
        """Add served scores to the rolling window."""
        with self._lock:
            self._recent.extend(float(s) for s in scores)

    def reset(self) -> None:
        with self._lock:
            self._recent.clear()

    def __len__(self) -> int:
        return len(self._recent)


_policy: AlertPolicy | None = None


def get_alert_policy() -> AlertPolicy:
    """The process-wide policy, built on first use from config + the serving reference."""
    global _policy
    if _policy is None:
        _policy = AlertPolicy(
            budget=alert_cfg.budget,
            window=alert_cfg.window,
            min_window=alert_cfg.min_window,
            fallback_threshold=reference_threshold(alert_cfg.budget),
        )
    return _policy
