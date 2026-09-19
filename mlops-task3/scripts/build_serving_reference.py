"""
Build models/07_serving_reference.json — fitted offline, read-only at serving time.

What it holds (see src/reference.py for how the service uses it):
    calibration        Platt scaling fitted on the VALIDATION split
    score_percentiles  percentiles 0..100 of the validation scores
    model_md5          the model it was built for (the service refuses a mismatch)
    evaluation         how calibration and the alert policy behave on the TEST split

The test split is only evaluated here, never fitted on.

Usage:
    python scripts/build_serving_reference.py
    python scripts/build_serving_reference.py --task2-artifacts ../mlops-task2/artifacts
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import sklearn  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import brier_score_loss, roc_auc_score  # noqa: E402

from src.alerting import AlertPolicy  # noqa: E402
from src.artifacts import file_md5  # noqa: E402
from src.config import alerting as alert_cfg  # noqa: E402
from src.config import model as model_cfg  # noqa: E402
from src.features import get_feature_columns, load_transformers  # noqa: E402
from src.predict import score  # noqa: E402

DEFAULT_TASK2 = PROJECT_ROOT.parent / "mlops-task2" / "artifacts"


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def _split(task2: Path, name: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Scores and labels of one split, in purchase-time order."""
    feats = pd.read_parquet(task2 / f"05_features_{name}.parquet")
    raw = pd.read_parquet(task2 / f"03_{name}.parquet")[["order_id", "order_purchase_timestamp"]]
    feats = feats.merge(raw, on="order_id", validate="one_to_one")
    feats = feats.sort_values("order_purchase_timestamp", kind="stable").reset_index(drop=True)
    scores = score(feats[get_feature_columns()], load_transformers())
    return scores, feats["is_late"].astype(int).to_numpy(), feats


def _simulate_policy(scores: np.ndarray, y: np.ndarray, fallback: float) -> dict:
    """Serve the split order by order, as the live service would, and count the alerts."""
    policy = AlertPolicy(alert_cfg.budget, alert_cfg.window, alert_cfg.min_window, fallback)
    flagged = np.zeros(len(scores), dtype=bool)
    for i, s in enumerate(scores):
        flagged[i] = s >= policy.threshold()
        policy.observe([s])
    tp = int((flagged & (y == 1)).sum())
    return {
        "flagged_share": round(float(flagged.mean()), 4),
        "precision": round(tp / max(int(flagged.sum()), 1), 4),
        "recall": round(tp / max(int(y.sum()), 1), 4),
    }


def main(task2: Path) -> None:
    val_scores, val_y, val = _split(task2, "val")
    test_scores, test_y, _ = _split(task2, "test")

    # ── Platt scaling on validation ──────────────────────────────────────────
    platt = LogisticRegression(C=1e12, max_iter=1000)
    platt.fit(_logit(val_scores).reshape(-1, 1), val_y)
    coef, intercept = float(platt.coef_[0, 0]), float(platt.intercept_[0])

    def calibrated(s: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-(coef * _logit(s) + intercept)))

    percentiles = np.percentile(val_scores, np.arange(101))
    fallback = float(np.interp(100 * (1 - alert_cfg.budget), np.arange(101), percentiles))
    fixed = test_scores >= fallback
    fixed_tp = int((fixed & (test_y == 1)).sum())

    reference = {
        "description": "Calibration and score distribution for serving. Built by "
        "scripts/build_serving_reference.py; see src/reference.py.",
        "model_md5": file_md5(model_cfg.artifact_path),
        "sklearn_version": sklearn.__version__,
        "built_on": {
            "split": "validation",
            "from": str(val["order_purchase_timestamp"].min().date()),
            "to": str(val["order_purchase_timestamp"].max().date()),
            "n_orders": len(val_y),
            "late_rate": round(float(val_y.mean()), 4),
        },
        "calibration": {"method": "platt", "coef": coef, "intercept": intercept},
        "score_percentiles": [round(float(p), 8) for p in percentiles],
        "evaluation": {
            "split": "test",
            "n_orders": len(test_y),
            "late_rate": round(float(test_y.mean()), 4),
            "roc_auc": round(float(roc_auc_score(test_y, test_scores)), 4),
            "mean_raw_score": round(float(test_scores.mean()), 4),
            "mean_calibrated_probability": round(float(calibrated(test_scores).mean()), 4),
            "brier_raw_score": round(float(brier_score_loss(test_y, test_scores)), 4),
            "brier_calibrated": round(float(brier_score_loss(test_y, calibrated(test_scores))), 4),
            "threshold_0_5_flagged_share": round(float((test_scores >= 0.5).mean()), 4),
            "alert_budget": alert_cfg.budget,
            "fixed_validation_threshold": {
                "threshold": round(fallback, 4),
                "flagged_share": round(float(fixed.mean()), 4),
                "precision": round(fixed_tp / max(int(fixed.sum()), 1), 4),
                "recall": round(fixed_tp / max(int(test_y.sum()), 1), 4),
            },
            "rolling_window_policy": _simulate_policy(test_scores, test_y, fallback),
        },
    }

    out = model_cfg.reference_path
    with open(out, "w", encoding="utf-8") as f:
        json.dump(reference, f, indent=2)
        f.write("\n")

    print(f"Saved {out.relative_to(PROJECT_ROOT)}")
    print(json.dumps(reference["built_on"], indent=2))
    print(json.dumps(reference["calibration"], indent=2))
    print(json.dumps(reference["evaluation"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the serving reference")
    parser.add_argument("--task2-artifacts", type=Path, default=DEFAULT_TASK2)
    args = parser.parse_args()
    main(args.task2_artifacts)
