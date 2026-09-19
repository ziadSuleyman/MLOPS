"""
Build tests/fixtures/parity_orders.parquet — real test-split orders with the features
Notebook 5 computed for them.

tests/test_parity.py feeds the raw columns through the service and asserts it produces
the notebook's features exactly. The fixture is small enough to commit, so the check
runs in CI without the Task 2 artifacts.

Rows: the conftest sample order, a random sample of the test split, and the awkward cases (missing coordinates,
missing category, missing photos, east-coast longitudes, coordinates outside Brazil).

Usage:
    python scripts/make_parity_fixture.py
    python scripts/make_parity_fixture.py --task2-artifacts ../mlops-task2/artifacts --n 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features import get_feature_columns  # noqa: E402
from src.validation import REQUIRED_COLUMNS  # noqa: E402

DEFAULT_TASK2 = PROJECT_ROOT.parent / "mlops-task2" / "artifacts"
OUT = PROJECT_ROOT / "tests" / "fixtures" / "parity_orders.parquet"
RAW_COLUMNS = ["order_id", *REQUIRED_COLUMNS, "main_category"]
# The sample order in tests/conftest.py — its features are checked against the notebook too
ALWAYS_INCLUDE = ["8a9be36ffd78382f9ac518945e909636"]


def main(task2: Path, n: int) -> None:
    raw = pd.read_parquet(task2 / "03_test.parquet")[RAW_COLUMNS]
    nb = pd.read_parquet(task2 / "05_features_test.parquet")
    nb = nb[["order_id", *get_feature_columns()]].rename(
        columns={c: f"nb__{c}" for c in get_feature_columns()}
    )
    both = raw.merge(nb, on="order_id", validate="one_to_one")

    edge_cases = {
        "missing customer coordinates": lambda d: d["customer_lat"].isna(),
        "missing seller coordinates": lambda d: d["seller_lat"].isna(),
        "missing category": lambda d: d["main_category"].isna(),
        "missing photos": lambda d: d["photos_avg"].isna(),
        "east-coast longitude": lambda d: d["customer_lng"] > -35,
        "coordinates outside Brazil": lambda d: d["customer_lat"] > 6,
    }
    edges = pd.concat(
        [both[both["order_id"].isin(ALWAYS_INCLUDE)]]
        + [both[is_case(both)].head(3) for is_case in edge_cases.values()]
    )
    sample = both.sample(n=n, random_state=0)
    fixture = pd.concat([edges, sample]).drop_duplicates("order_id").reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fixture.to_parquet(OUT, index=False)
    print(f"Saved {OUT.relative_to(PROJECT_ROOT)}: {len(fixture)} orders")
    for name, is_case in edge_cases.items():
        print(f"  {name:30s} {int(is_case(fixture).sum())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the parity test fixture")
    parser.add_argument("--task2-artifacts", type=Path, default=DEFAULT_TASK2)
    parser.add_argument("--n", type=int, default=300)
    args = parser.parse_args()
    main(args.task2_artifacts, args.n)
