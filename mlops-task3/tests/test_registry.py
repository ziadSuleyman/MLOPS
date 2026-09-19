"""
Tests for src/registry.py — registering the serving bundle and loading it back.

The registry tests use a throw-away MLflow store (SQLite in a temp dir), so no server
is needed. They are skipped where MLflow is not installed; it is in
requirements/base.txt, so CI and the Docker image run them.
"""

import pandas as pd
import pytest

import src.alerting
import src.artifacts
import src.features
import src.predict
import src.reference
import src.registry as registry
from src.artifacts import artifact_manifest, file_md5, model_version
from src.config import model as model_cfg
from src.features import build_features, load_transformers
from src.predict import predict_single


@pytest.fixture
def isolated(monkeypatch):
    """Restore artifact paths, the recorded source and every cache after the test."""
    for attr in registry.ARTIFACT_ATTRS.values():
        monkeypatch.setattr(model_cfg, attr, getattr(model_cfg, attr))
    monkeypatch.setattr(registry, "_source", dict(registry._source))
    for module, cache in [
        (src.predict, "_model_bundle"),
        (src.features, "_transformers"),
        (src.features, "_feature_columns"),
        (src.reference, "_reference"),
        (src.artifacts, "_version"),
        (src.alerting, "_policy"),
    ]:
        monkeypatch.setattr(module, cache, getattr(module, cache))
    return monkeypatch


@pytest.fixture
def store(tmp_path, isolated):
    """A fresh MLflow tracking + registry store in a temp dir."""
    mlflow = pytest.importorskip("mlflow")
    previous = mlflow.get_tracking_uri()
    isolated.chdir(tmp_path)  # the default artifact root (./mlruns) lands in tmp
    yield f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(previous)


def _client(uri):
    from mlflow.tracking import MlflowClient

    return MlflowClient(tracking_uri=uri)


class TestRegister:
    def test_alias_stage_and_tags(self, store):
        """The new version is 'production', stage Production, tagged with hash + md5s."""
        result = registry.register_bundle(store)
        mv = _client(store).get_model_version_by_alias(
            model_cfg.registry_name, model_cfg.registry_alias
        )
        assert str(mv.version) == result["version"]
        assert mv.current_stage == model_cfg.registry_stage
        assert mv.tags["model_version"] == model_version()
        for name, md5 in artifact_manifest().items():
            assert mv.tags[f"md5_{name}"] == md5

    def test_if_missing_reuses_the_version(self, store):
        """Registering the same files again with --if-missing adds no version."""
        first = registry.register_bundle(store)
        again = registry.register_bundle(store, if_missing=True)
        assert again["registered"] is False
        assert again["version"] == first["version"]

    def test_metric_names_are_mlflow_safe(self):
        assert registry.metric_name("test_recall@5%") == "test_recall_at_5pct"


class TestResolve:
    def test_downloads_the_exact_files(self, store, tmp_path):
        """The production version's serving/ files are byte-identical to models/."""
        registry.register_bundle(store)
        found = registry.resolve_from_registry(store, tmp_path / "download")
        local = artifact_manifest()
        for name, path in found["paths"].items():
            assert file_md5(path) == local[name]

    def test_tampered_version_is_refused(self, store, tmp_path):
        """If the recorded md5 does not match what the registry serves, loading stops."""
        result = registry.register_bundle(store)
        _client(store).set_model_version_tag(
            model_cfg.registry_name, result["version"], "md5_model", "0" * 32
        )
        with pytest.raises(registry.RegistryIntegrityError):
            registry.resolve_from_registry(store, tmp_path / "download")


class TestSelectSource:
    def test_service_loads_from_registry(self, store, isolated, sample_order):
        """With the registry configured, the service serves the downloaded files — same output."""
        before = predict_single(build_features(pd.DataFrame([sample_order])), load_transformers())
        registry.register_bundle(store)

        isolated.setattr(model_cfg, "source", "registry")
        isolated.setenv("MLFLOW_TRACKING_URI", store)
        for module, cache in [
            (src.predict, "_model_bundle"),
            (src.features, "_transformers"),
            (src.features, "_feature_columns"),
            (src.reference, "_reference"),
            (src.artifacts, "_version"),
            (src.alerting, "_policy"),
        ]:
            isolated.setattr(module, cache, None)

        chosen = registry.select_model_source()
        assert chosen["source"] == "registry"
        assert model_cfg.artifact_path.parent != model_cfg.local_dir

        after = predict_single(build_features(pd.DataFrame([sample_order])), load_transformers())
        for key in ("score", "probability", "model_version"):
            assert after[key] == before[key]

    def test_no_tracking_uri_means_local(self, isolated):
        isolated.setattr(model_cfg, "source", "registry")
        isolated.delenv("MLFLOW_TRACKING_URI", raising=False)
        assert registry.select_model_source()["source"] == "local"

    def test_unreachable_registry_falls_back(self, isolated):
        """MLflow down → the local files (the same DVC-tracked bytes), not a crash."""
        isolated.setattr(model_cfg, "source", "registry")
        isolated.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:9")
        chosen = registry.select_model_source()
        assert chosen["source"] == "local"
        assert "unavailable" in chosen["detail"]

    def test_fallback_can_be_disabled(self, isolated):
        isolated.setattr(model_cfg, "source", "registry")
        isolated.setattr(model_cfg, "registry_fallback_to_local", False)
        isolated.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:9")
        with pytest.raises(OSError):
            registry.select_model_source()
