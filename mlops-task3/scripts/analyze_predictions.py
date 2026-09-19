"""
Analyze the JSONL prediction log.

Usage:
    python scripts/analyze_predictions.py                  # last 24h
    python scripts/analyze_predictions.py --hours 168      # last 7 days
    python scripts/analyze_predictions.py --all            # everything

Prints:
    - Total predictions, and how many carry an order_id (scorable later)
    - Share flagged "late" (should sit near the alert budget)
    - Mean calibrated probability
    - Latency percentiles (p50, p95, p99)
    - Model version distribution
    - Drift: PSI of the logged scores against the validation split

Drift is judged on the score distribution, not on the share predicted late: that share
is set by the alert budget, and the real late rate moves between periods anyway.
To compare predictions with real deliveries, use scripts/evaluate_outcomes.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.config import alerting as alert_cfg  # noqa: E402
from src.config import monitoring as mon_cfg  # noqa: E402
from src.evaluation import latency_percentiles, load_prediction_log  # noqa: E402
from src.monitoring import compute_psi, psi_status  # noqa: E402
from src.reference import psi_bins  # noqa: E402


def analyze(records: pd.DataFrame) -> None:
    """Print analysis of prediction records."""
    if "score" in records:
        # Lines written before v1.1.0 carry no score, order_id or batch_size
        records = records[records["score"].notna()]
    if records.empty or "score" not in records:
        print("No predictions to analyze (records written before v1.1.0 carry no score).")
        return

    n = len(records)
    flagged = float((records["prediction"] == "late").mean())
    with_id = int(records["order_id"].notna().sum()) if "order_id" in records else 0
    # Latency belongs to requests, not orders: the orders of one batch share a timestamp
    requests = records.drop_duplicates("timestamp")

    print("=" * 50)
    print("  PREDICTION LOG ANALYSIS")
    print("=" * 50)
    print(f"\n  Total predictions:   {n}  (in {len(requests)} requests)")
    print(f"  With order_id:       {with_id}  ({with_id / n:.1%}) - scorable later")
    print(f"  Flagged late:        {flagged:.1%}  (alert budget {alert_cfg.budget:.0%})")
    print(f"  Mean probability:    {records['probability'].mean():.2%}")
    for kind, rows in [
        ("single", requests[requests["batch_size"] == 1]),
        ("batch", requests[requests["batch_size"] > 1]),
    ]:
        if len(rows):
            lat = latency_percentiles(rows["latency_ms"])
            print(
                f"\n  Latency, {kind:6s} ({len(rows)} requests): "
                f"p50 {lat['p50']:.1f} ms  p95 {lat['p95']:.1f} ms  p99 {lat['p99']:.1f} ms"
            )

    print("\n  Model versions:")
    for v, count in records["model_version"].value_counts().sort_index().items():
        print(f"    {v}: {count} predictions")

    if n < mon_cfg.drift_min_window:
        print(f"\n  Drift: needs at least {mon_cfg.drift_min_window} predictions, have {n}")
    else:
        psi = compute_psi(records["score"].to_numpy(), *psi_bins())
        status = psi_status(psi)
        print(f"\n  Score PSI vs validation split: {psi:.3f}  -> {status}")
        print(f"  (warn >= {mon_cfg.psi_warn}, alert >= {mon_cfg.psi_alert})")
        if status == "alert":
            print("\n  ** DRIFT ALERT ** The inputs no longer look like the validation period.")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze prediction logs")
    parser.add_argument("--hours", type=float, default=24, help="Hours to look back")
    parser.add_argument("--all", action="store_true", help="Analyze all records")
    args = parser.parse_args()

    hours = None if args.all else args.hours
    analyze(load_prediction_log(mon_cfg.prediction_log, hours=hours))


if __name__ == "__main__":
    main()
