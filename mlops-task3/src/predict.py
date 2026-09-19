"""
Model loading and prediction.

Loads the trained model (logistic regression) from 06_model.joblib, applies scaling,
and turns each raw score into:
    probability      — calibrated P(late), see src/reference.py
    risk_percentile  — where the score falls among validation orders
    prediction       — "late" if the order is inside the alert budget, see src/alerting.py

Ported from: notebooks/06_train_evaluate.ipynb.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.alerting import get_alert_policy
from src.artifacts import artifact_manifest, load_pickle, model_version
from src.config import alerting as alert_cfg
from src.config import model as model_cfg
from src.config import project_version
from src.logger import log
from src.reference import calibrate, load_reference, risk_percentile
from src.registry import current_source

# ── Module-level cache ───────────────────────────────────────────────────────
_model_bundle: dict[str, Any] | None = None


def load_model() -> dict[str, Any]:
    """Load the model bundle from disk. Cached after the first call."""
    global _model_bundle

    if _model_bundle is not None:
        return _model_bundle

    path = model_cfg.artifact_path
    log.info("Loading model from %s", path)
    _model_bundle = load_pickle(path)

    log.info(
        "Model loaded: %s  |  %d features  |  requires_scaling=%s",
        _model_bundle["model_type"],
        len(_model_bundle["features"]),
        _model_bundle["requires_scaling"],
    )
    return _model_bundle


def score(features: pd.DataFrame, transformers: dict[str, Any]) -> np.ndarray:
    """Raw model scores (a ranking, not a probability — the model was class-balanced)."""
    bundle = load_model()

    # Scale if the model requires it (logistic regression does)
    if bundle["requires_scaling"]:
        x_input = transformers["scaler"].transform(features)
    else:
        x_input = features.to_numpy()

    return bundle["model"].predict_proba(x_input)[:, 1]


def predict(
    features: pd.DataFrame,
    transformers: dict[str, Any],
    threshold: float | None = None,
) -> list[dict]:
    """
    Run the model on a feature DataFrame.

    Parameters
    ----------
    features : DataFrame with exactly the 31 model features.
    transformers : The fitted transformers dict (contains the scaler).
    threshold : Score threshold for "late". None → the alert policy's current threshold.

    Returns
    -------
    List of dicts, one per row, with prediction, probability, score,
    risk_percentile, alert_threshold and model_version.
    """
    scores = score(features, transformers)
    thr = get_alert_policy().threshold() if threshold is None else threshold
    probabilities = calibrate(scores)
    percentiles = risk_percentile(scores)
    version = model_version()

    results = [
        {
            "prediction": "late" if s >= thr else "on_time",
            "probability": round(float(p), 4),
            "score": round(float(s), 4),
            "risk_percentile": round(float(q), 1),
            "alert_threshold": round(float(thr), 4),
            "model_version": version,
        }
        for s, p, q in zip(scores, probabilities, percentiles, strict=True)
    ]

    log.debug(
        "Predicted %d orders: %d late, %d on_time (threshold %.4f)",
        len(results),
        sum(1 for r in results if r["prediction"] == "late"),
        sum(1 for r in results if r["prediction"] == "on_time"),
        thr,
    )

    return results


def predict_single(
    features: pd.DataFrame, transformers: dict[str, Any], threshold: float | None = None
) -> dict:
    """Convenience wrapper for a single order."""
    return predict(features, transformers, threshold)[0]


def get_model_info() -> dict:
    """Return model metadata for the /model/info endpoint."""
    bundle = load_model()
    ref = load_reference()
    policy = get_alert_policy()
    return {
        "model_type": bundle["model_type"],
        "model_version": model_version(),
        "service_version": project_version,
        "n_features": len(bundle["features"]),
        "features": bundle["features"],
        "hyperparameters": bundle["hyperparameters"],
        "label_definition": bundle.get("label_definition", model_cfg.label_definition),
        "requires_scaling": bundle["requires_scaling"],
        "metrics": bundle.get("metrics", {}),
        "artifacts_md5": artifact_manifest(),
        "model_source": current_source(),
        "calibration": {
            "method": ref["calibration"]["method"],
            "fitted_on": ref["built_on"]["split"],
            "fitted_on_late_rate": ref["built_on"]["late_rate"],
        },
        "alerting": {
            "budget": alert_cfg.budget,
            "current_threshold": round(policy.threshold(), 4),
            "threshold_source": policy.source(),
            "window_size": len(policy),
        },
    }
