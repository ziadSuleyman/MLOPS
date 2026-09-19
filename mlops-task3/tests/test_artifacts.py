"""
Tests for src/artifacts.py — what exactly is being served.

Covers: the model version is a hash of the artifacts and changes with them,
the artifacts match their DVC pointers, and a scikit-learn version mismatch
refuses to load.
"""

import re
import warnings

import pytest
import yaml
from sklearn.exceptions import InconsistentVersionWarning

import src.artifacts as artifacts
from src.artifacts import (
    artifact_manifest,
    compute_model_version,
    dvc_mismatches,
    file_md5,
    load_pickle,
    model_version,
    serving_artifacts,
)
from src.config import model as model_cfg


class TestModelVersion:
    def test_format(self):
        """12 hex characters."""
        assert re.fullmatch(r"[0-9a-f]{12}", model_version())

    def test_stable(self):
        """Same files → same version."""
        assert compute_model_version(artifact_manifest()) == model_version()

    def test_changes_with_any_artifact(self):
        """Changing any one artifact changes the version."""
        manifest = artifact_manifest()
        for name in manifest:
            changed = {**manifest, name: "0" * 32}
            assert compute_model_version(changed) != model_version()

    def test_covers_all_serving_files(self):
        """Model, transformers, feature list and reference are all hashed."""
        assert set(serving_artifacts()) == {"model", "transformers", "feature_list", "reference"}


class TestDVC:
    def test_artifacts_match_dvc_pointers(self):
        """Every served file is exactly the version DVC recorded."""
        assert dvc_mismatches() == []

    def test_model_md5_is_dvc_md5(self):
        """The md5 we report is the md5 DVC stores."""
        pointer = model_cfg.artifact_path.with_name(model_cfg.artifact_path.name + ".dvc")
        recorded = yaml.safe_load(pointer.read_text(encoding="utf-8"))["outs"][0]["md5"]
        assert recorded == file_md5(model_cfg.artifact_path)


class TestLoadPickle:
    def test_loads_current_artifacts(self):
        """The Task 2 artifacts load with the pinned scikit-learn."""
        assert load_pickle(model_cfg.artifact_path)["model_type"] == "LogisticRegression"

    def test_version_mismatch_is_an_error(self, monkeypatch):
        """A pickle from another scikit-learn version raises instead of warning."""

        def fake_load(path):
            warnings.warn(
                InconsistentVersionWarning(
                    estimator_name="LogisticRegression",
                    current_sklearn_version="1.6.1",
                    original_sklearn_version="1.9.0",
                ),
                stacklevel=1,
            )

        monkeypatch.setattr(artifacts.joblib, "load", fake_load)
        with pytest.raises(RuntimeError, match="pickled with scikit-learn 1.9.0"):
            load_pickle(model_cfg.artifact_path)
