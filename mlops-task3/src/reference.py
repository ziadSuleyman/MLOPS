"""
Serving reference — what the service needs besides the model itself.

Built once, offline, by scripts/build_serving_reference.py from the VALIDATION split
of Task 2, and saved as plain JSON (models/07_serving_reference.json):

- Platt calibration: the model was trained with class_weight="balanced", so its raw
  score averages ~0.49 while the real late rate is 3-8%. The score ranks orders well
  (ROC-AUC 0.71) but is not a probability. Calibration maps it to one.
- The validation score distribution (percentiles 0..100): gives each order a risk
  percentile, the alert threshold used before enough live traffic exists, and the
  reference that PSI drift is measured against.

Nothing here is fitted at inference.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from src.artifacts import file_md5
from src.config import model as model_cfg
from src.logger import log

_EPS = 1e-9
_reference: dict[str, Any] | None = None


def load_reference() -> dict[str, Any]:
    """Load the reference once and check it was built for the model being served."""
    global _reference
    if _reference is not None:
        return _reference

    path = model_cfg.reference_path
    with open(path, encoding="utf-8") as f:
        ref = json.load(f)

    served_md5 = file_md5(model_cfg.artifact_path)
    if ref["model_md5"] != served_md5:
        raise RuntimeError(
            f"{path.name} was built for model md5 {ref['model_md5']} but the served model "
            f"is {served_md5}. Rebuild it: python scripts/build_serving_reference.py"
        )

    ref["percentiles"] = np.asarray(ref["score_percentiles"], dtype=float)
    _reference = ref
    log.info(
        "Serving reference loaded: built on %s (%d orders, late rate %.2f%%)",
        ref["built_on"]["split"],
        ref["built_on"]["n_orders"],
        100 * ref["built_on"]["late_rate"],
    )
    return _reference


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def calibrate(scores: np.ndarray) -> np.ndarray:
    """Raw model score → calibrated probability of a late delivery."""
    cal = load_reference()["calibration"]
    z = cal["coef"] * _logit(np.asarray(scores, dtype=float)) + cal["intercept"]
    return 1 / (1 + np.exp(-z))


def risk_percentile(scores: np.ndarray) -> np.ndarray:
    """Share (0-100) of validation orders that scored lower than each order."""
    pct = load_reference()["percentiles"]
    return np.interp(np.asarray(scores, dtype=float), pct, np.arange(len(pct), dtype=float))


def reference_threshold(budget: float) -> float:
    """The score that flags `budget` of the validation orders."""
    pct = load_reference()["percentiles"]
    return float(np.interp(100 * (1 - budget), np.arange(len(pct), dtype=float), pct))


def psi_bins() -> tuple[np.ndarray, np.ndarray]:
    """Decile edges of the validation scores and the share expected in each bin (10%)."""
    pct = load_reference()["percentiles"]
    edges = pct[::10].copy()  # 0th, 10th, ..., 100th percentile → 11 edges, 10 bins
    edges[0], edges[-1] = -np.inf, np.inf
    expected = np.full(len(edges) - 1, 1 / (len(edges) - 1))
    return edges, expected
