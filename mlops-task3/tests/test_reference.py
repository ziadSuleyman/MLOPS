"""
Tests for src/reference.py — calibration and the validation score distribution.

Covers: calibration is monotone and lands on the validation late rate, risk
percentiles, PSI bins, and the guard against a reference built for another model.
"""

import json

import numpy as np
import pytest

import src.reference as reference
from src.config import model as model_cfg
from src.reference import calibrate, load_reference, psi_bins, risk_percentile


class TestCalibration:
    def test_monotone(self):
        """A higher score always means a higher probability."""
        scores = np.linspace(0.01, 0.99, 200)
        assert np.all(np.diff(calibrate(scores)) > 0)

    def test_matches_validation_late_rate(self):
        """Averaged over the validation distribution, it gives the validation late rate."""
        ref = load_reference()
        probs = calibrate(ref["percentiles"])
        assert probs.mean() == pytest.approx(ref["built_on"]["late_rate"], abs=0.01)

    def test_far_below_raw_scores(self):
        """The class-balanced score of a median order is ~0.5; its probability is not."""
        median_score = load_reference()["percentiles"][50]
        assert 0.4 < median_score < 0.6
        assert calibrate(np.array([median_score]))[0] < 0.1


class TestRiskPercentile:
    def test_bounds(self):
        """Scores below/above everything seen map to 0 and 100."""
        assert risk_percentile(np.array([0.0]))[0] == 0
        assert risk_percentile(np.array([1.0]))[0] == 100

    def test_median(self):
        """The validation median sits at the 50th percentile."""
        median_score = load_reference()["percentiles"][50]
        assert risk_percentile(np.array([median_score]))[0] == pytest.approx(50)


class TestPSIBins:
    def test_deciles(self):
        """11 edges, open-ended, 10% expected in each bin."""
        edges, expected = psi_bins()
        assert len(edges) == 11
        assert edges[0] == -np.inf and edges[-1] == np.inf
        assert expected.sum() == pytest.approx(1)


class TestModelGuard:
    def test_reference_for_another_model_is_refused(self, tmp_path, monkeypatch):
        """A reference built for a different model file is not silently used."""
        wrong = json.loads(model_cfg.reference_path.read_text(encoding="utf-8"))
        wrong["model_md5"] = "0" * 32
        path = tmp_path / "reference.json"
        path.write_text(json.dumps(wrong), encoding="utf-8")

        monkeypatch.setattr(model_cfg, "reference_path", path)
        monkeypatch.setattr(reference, "_reference", None)
        with pytest.raises(RuntimeError, match="Rebuild it"):
            load_reference()
