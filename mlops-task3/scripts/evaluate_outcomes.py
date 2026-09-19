"""
Score logged predictions against real deliveries.

Reads logs/predictions.jsonl, looks the order_ids up in the orders table, applies the
Task 2 label (calendar-day rule) and prints how the model did in production:
ROC-AUC, alert precision/recall, and whether the calibrated probabilities match the
real late rate. This is the check that tells you when to recalibrate or retrain.

Usage:
    python scripts/evaluate_outcomes.py            # last 24h of predictions
    python scripts/evaluate_outcomes.py --all      # everything logged

Requires the Olist database (DB_* settings in .env, default localhost:5433).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import create_engine, text  # noqa: E402

from src.config import db as db_cfg  # noqa: E402
from src.config import monitoring as mon_cfg  # noqa: E402
from src.evaluation import evaluate, load_prediction_log  # noqa: E402

QUERY = text(
    """
    SELECT order_id, order_delivered_customer_date, order_estimated_delivery_date
    FROM orders
    WHERE order_id = ANY(:ids)
    """
)


def load_outcomes(order_ids: list[str]) -> pd.DataFrame:
    engine = create_engine(db_cfg.url)
    with engine.connect() as conn:
        return pd.read_sql(QUERY, conn, params={"ids": order_ids})


def main(hours: float | None) -> None:
    preds = load_prediction_log(mon_cfg.prediction_log, hours=hours)
    if preds.empty or "order_id" not in preds:
        print("No predictions with an order_id to evaluate.")
        return

    ids = preds["order_id"].dropna().unique().tolist()
    outcomes = load_outcomes(ids)
    report = evaluate(preds, outcomes)

    print("=" * 50)
    print("  PRODUCTION OUTCOMES")
    print("=" * 50)
    print(json.dumps(report, indent=2))
    missing = report["n_logged"] - report["n_with_order_id"]
    if missing:
        print(f"\n  {missing} prediction(s) had no order_id and cannot be scored.")
    if report["mean_probability"] is not None and report["late_rate"] is not None:
        gap = report["mean_probability"] - report["late_rate"]
        print(f"  Calibration gap (mean probability - real late rate): {gap:+.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score logged predictions against outcomes")
    parser.add_argument("--hours", type=float, default=24, help="Hours to look back")
    parser.add_argument("--all", action="store_true", help="Evaluate every logged prediction")
    args = parser.parse_args()
    main(None if args.all else args.hours)
