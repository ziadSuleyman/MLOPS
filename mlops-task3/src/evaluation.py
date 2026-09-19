"""
Scoring logged predictions against what actually happened.

Every prediction is logged with its order_id (src/monitoring.py). Once the order is
delivered, the orders table says whether it was late, and the prediction can be scored.

The label is Task 2's primary one — the calendar-day rule (notebook 02_labels):
late if delivered on a later DATE than promised. The promise is a date, not an instant.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def load_prediction_log(path: Path, hours: float | None = None) -> pd.DataFrame:
    """Read the JSONL prediction log; skip unreadable lines. `hours` keeps only recent ones."""
    if not Path(path).exists():
        return pd.DataFrame()

    cutoff = datetime.now(UTC) - timedelta(hours=hours) if hours is not None else None
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if cutoff is not None and datetime.fromisoformat(rec["timestamp"]) < cutoff:
                    continue
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
            records.append(rec)
    return pd.DataFrame(records)


def label_calendar_day(delivered: pd.Series, estimated: pd.Series) -> pd.Series:
    """Late = delivered on a later calendar day than the promised date."""
    delivered = pd.to_datetime(delivered, format="mixed")
    estimated = pd.to_datetime(estimated, format="mixed")
    return delivered.dt.normalize() > estimated.dt.normalize()


def evaluate(predictions: pd.DataFrame, outcomes: pd.DataFrame) -> dict:
    """
    Join predictions to outcomes by order_id and score them.

    Parameters
    ----------
    predictions : the prediction log (order_id, prediction, probability, score).
                  An order predicted twice counts once — the latest prediction.
    outcomes : order_id, order_delivered_customer_date, order_estimated_delivery_date.

    Returns
    -------
    dict of counts and metrics; metrics are None when they cannot be computed.
    """
    n_logged = len(predictions)
    report = {
        "n_logged": n_logged,
        "n_with_order_id": 0,
        "n_delivered": 0,
        "late_rate": None,
        "roc_auc": None,
        "pr_auc": None,
        "flagged_share": None,
        "precision": None,
        "recall": None,
        "mean_probability": None,
        "brier": None,
    }
    if n_logged == 0 or "order_id" not in predictions:
        return report

    with_id = predictions[predictions["order_id"].notna()]
    report["n_with_order_id"] = len(with_id)
    latest = with_id.drop_duplicates("order_id", keep="last")

    delivered = outcomes[outcomes["order_delivered_customer_date"].notna()].copy()
    delivered["is_late"] = label_calendar_day(
        delivered["order_delivered_customer_date"], delivered["order_estimated_delivery_date"]
    )
    joined = latest.merge(delivered[["order_id", "is_late"]], on="order_id", how="inner")
    report["n_delivered"] = len(joined)
    if joined.empty:
        return report

    y = joined["is_late"].astype(int).to_numpy()
    flagged = (joined["prediction"] == "late").to_numpy()
    tp = int((flagged & (y == 1)).sum())

    report["late_rate"] = round(float(y.mean()), 4)
    report["flagged_share"] = round(float(flagged.mean()), 4)
    report["precision"] = round(tp / flagged.sum(), 4) if flagged.any() else None
    report["recall"] = round(tp / y.sum(), 4) if y.any() else None
    report["mean_probability"] = round(float(joined["probability"].mean()), 4)
    report["brier"] = round(float(brier_score_loss(y, joined["probability"])), 4)
    if 0 < y.sum() < len(y):
        report["roc_auc"] = round(float(roc_auc_score(y, joined["score"])), 4)
        report["pr_auc"] = round(float(average_precision_score(y, joined["score"])), 4)
    return report


def latency_percentiles(latencies: pd.Series) -> dict:
    """p50/p95/p99 of latency in ms."""
    values = np.asarray(latencies, dtype=float)
    return {f"p{q}": round(float(np.percentile(values, q)), 1) for q in (50, 95, 99)}
