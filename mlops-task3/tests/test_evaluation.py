"""
Tests for src/evaluation.py — scoring logged predictions against real deliveries.

Covers: the calendar-day label, the join by order_id, the latest prediction per
order, unscorable predictions, and reading the log.
"""

import json

import pandas as pd
import pytest

from src.evaluation import evaluate, label_calendar_day, load_prediction_log


def _preds(rows):
    return pd.DataFrame(rows, columns=["order_id", "prediction", "probability", "score"])


def _outcomes(rows):
    return pd.DataFrame(
        rows,
        columns=["order_id", "order_delivered_customer_date", "order_estimated_delivery_date"],
    )


class TestLabel:
    def test_same_day_is_on_time(self):
        """Delivered at 18:00 on the promised date is NOT late (the midnight problem)."""
        late = label_calendar_day(
            pd.Series(["2018-06-10 18:00:00"]), pd.Series(["2018-06-10 00:00:00"])
        )
        assert not late.iloc[0]

    def test_next_day_is_late(self):
        late = label_calendar_day(
            pd.Series(["2018-06-11 08:00:00"]), pd.Series(["2018-06-10 00:00:00"])
        )
        assert late.iloc[0]


class TestEvaluate:
    OUTCOMES = _outcomes(
        [
            ("a", "2018-06-12", "2018-06-10"),  # late
            ("b", "2018-06-09", "2018-06-10"),  # on time
            ("c", "2018-06-10 20:00", "2018-06-10"),  # on time (same day)
            ("d", None, "2018-06-10"),  # not delivered yet
        ]
    )

    def test_metrics(self):
        """Perfect ranking and one correct alert."""
        preds = _preds(
            [
                ("a", "late", 0.30, 0.90),
                ("b", "on_time", 0.02, 0.20),
                ("c", "on_time", 0.03, 0.30),
            ]
        )
        report = evaluate(preds, self.OUTCOMES)
        assert report["n_delivered"] == 3
        assert report["late_rate"] == pytest.approx(1 / 3, abs=1e-4)
        assert report["roc_auc"] == 1.0
        assert report["precision"] == 1.0
        assert report["recall"] == 1.0

    def test_undelivered_and_unknown_ids_excluded(self):
        """Orders not delivered yet, or not in the table, are not scored."""
        preds = _preds([("d", "late", 0.5, 0.9), ("zzz", "late", 0.5, 0.9)])
        report = evaluate(preds, self.OUTCOMES)
        assert report["n_delivered"] == 0
        assert report["roc_auc"] is None

    def test_latest_prediction_wins(self):
        """An order predicted twice is scored once, on its latest prediction."""
        preds = _preds([("a", "on_time", 0.01, 0.1), ("a", "late", 0.3, 0.9)])
        report = evaluate(preds, self.OUTCOMES)
        assert report["n_delivered"] == 1
        assert report["flagged_share"] == 1.0

    def test_missing_order_ids_counted(self):
        """Predictions without an order_id are counted but cannot be scored."""
        preds = _preds([(None, "late", 0.3, 0.9), ("b", "on_time", 0.02, 0.2)])
        report = evaluate(preds, self.OUTCOMES)
        assert report["n_logged"] == 2
        assert report["n_with_order_id"] == 1

    def test_empty(self):
        report = evaluate(pd.DataFrame(), self.OUTCOMES)
        assert report["n_logged"] == 0


class TestLoadLog:
    def test_skips_bad_lines(self, tmp_path):
        """Corrupt lines are skipped, good ones kept."""
        path = tmp_path / "p.jsonl"
        good = {"timestamp": "2099-01-01T00:00:00+00:00", "order_id": "a"}
        path.write_text(json.dumps(good) + "\n{not json\n\n", encoding="utf-8")
        assert load_prediction_log(path)["order_id"].tolist() == ["a"]

    def test_missing_file(self, tmp_path):
        assert load_prediction_log(tmp_path / "none.jsonl").empty
