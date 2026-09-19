"""Small helpers shared by test modules (no fixtures, no side effects)."""

import json
from pathlib import Path

import pandas as pd

from src.validation import REQUIRED_COLUMNS

RAW_COLUMNS = ["order_id", *REQUIRED_COLUMNS, "main_category"]


def as_api_orders(frame: pd.DataFrame) -> list[dict]:
    """Rows of a raw-order frame as the API would receive them (strings and None)."""
    raw = frame[RAW_COLUMNS]
    records = raw.astype(object).where(raw.notna(), None).to_dict("records")
    for rec in records:
        for key, value in rec.items():
            if isinstance(value, pd.Timestamp):
                rec[key] = str(value)
    return records


def read_log(path: Path) -> list[dict]:
    """The JSONL prediction log as a list of records ([] if absent)."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
