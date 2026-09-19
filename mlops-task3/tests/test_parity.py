"""
Parity with the notebooks — the one guarantee Task 3 exists to keep.

Real test-split orders go through the service; what comes out must equal what
Notebook 5 computed (features) and what the Notebook 6 model gives on those
features (scores). Runs on the committed fixture, so CI checks it too.

If the full Task 2 artifacts are present (../mlops-task2/artifacts), one more test
serves the entire test split and checks the headline number: ROC-AUC 0.7107.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.features import build_features, get_feature_columns
from src.pipeline import run_batch_inference
from src.predict import score
from tests.helpers import RAW_COLUMNS, as_api_orders

TASK2_TEST_SPLIT = (
    Path(__file__).resolve().parents[2] / "mlops-task2" / "artifacts" / "03_test.parquet"
)


def _notebook_features(parity_orders: pd.DataFrame) -> pd.DataFrame:
    cols = get_feature_columns()
    nb = parity_orders[[f"nb__{c}" for c in cols]]
    nb.columns = cols
    return nb


class TestFeatureParity:
    def test_features_equal_notebook(self, parity_orders):
        """Every feature of every fixture order equals Notebook 5's value."""
        service = build_features(parity_orders[RAW_COLUMNS])
        notebook = _notebook_features(parity_orders)
        diff = (service - notebook).abs().max()
        assert diff.max() < 1e-9, f"Features differ from the notebook:\n{diff[diff > 1e-9]}"

    def test_fixture_covers_awkward_cases(self, parity_orders):
        """The fixture really contains the cases most likely to break parity."""
        assert parity_orders["customer_lat"].isna().any()
        assert parity_orders["main_category"].isna().any()
        assert (parity_orders["customer_lng"] > -35).any()
        assert (parity_orders["customer_lat"] > 6).any()

    def test_sample_order_is_real(self, parity_orders, sample_order):
        """conftest's sample order is a real order, with the notebook's features."""
        row = parity_orders[parity_orders["order_id"] == sample_order["order_id"]]
        assert len(row) == 1
        service = build_features(pd.DataFrame([sample_order]))
        notebook = _notebook_features(row).reset_index(drop=True)
        assert np.allclose(service.to_numpy(), notebook.to_numpy(), rtol=0, atol=1e-9)


class TestScoreParity:
    def test_scores_equal_model_on_notebook_features(self, parity_orders, transformers):
        """Service scores = the Notebook 6 model applied to Notebook 5 features."""
        service = score(build_features(parity_orders[RAW_COLUMNS]), transformers)
        notebook = score(_notebook_features(parity_orders), transformers)
        assert np.abs(service - notebook).max() < 1e-12

    def test_api_path_equals_notebook(self, parity_orders, transformers, prediction_log):
        """Through validation, JSON-style input and logging, scores still match."""
        results = run_batch_inference(as_api_orders(parity_orders))
        served = np.array([r["score"] for r in results])
        notebook = score(_notebook_features(parity_orders), transformers)
        assert np.abs(served - notebook).max() < 1e-4  # scores are rounded to 4 places
        assert [r["order_id"] for r in results] == parity_orders["order_id"].tolist()


# The parquet files are gitignored: on GitHub the artifacts folder exists without them
@pytest.mark.skipif(not TASK2_TEST_SPLIT.exists(), reason="Task 2 test split not available (CI)")
class TestFullTestSplit:
    def test_roc_auc_matches_notebook_6(self, transformers):
        """The whole test split through the service reproduces ROC-AUC 0.7107."""
        test = pd.read_parquet(TASK2_TEST_SPLIT)
        scores = score(build_features(test[RAW_COLUMNS]), transformers)
        assert round(roc_auc_score(test["is_late"].astype(int), scores), 4) == 0.7107
