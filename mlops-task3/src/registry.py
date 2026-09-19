"""
Model registry — registering the serving bundle, and loading it back.

Registration (scripts/register_model.py, and the `register` step of docker compose):
the four serving artifacts are logged to one MLflow run under `serving/`, the sklearn
model is registered, and the new version gets the alias `production` (plus the old-style
stage "Production"). Every version carries the md5 of each file and the artifact hash
(`model_version`) as tags.

Loading (app startup, `model.source: registry`): the version behind the alias is looked
up, its `serving/` files are downloaded, and every md5 is checked against the version's
tags and against the DVC pointers in models/. Only then do the loaders see them.

    registry unreachable / not configured  → local files (same DVC-tracked bytes), logged
    registry answers with different bytes  → the service refuses to start

MLflow is imported lazily: the service runs (from local files) without it installed.
"""

from __future__ import annotations

import os
import re
import tempfile
import urllib.request
import warnings
from pathlib import Path
from typing import Any

from src.artifacts import artifact_manifest, dvc_mismatches, file_md5, model_version
from src.config import model as model_cfg
from src.config import project_name, project_version
from src.logger import log

SERVING_DIR = "serving"  # artifact folder of the run that holds the four files
ARTIFACT_ATTRS = {
    "model": "artifact_path",
    "transformers": "transformers_path",
    "feature_list": "feature_list_path",
    "reference": "reference_path",
}

_source: dict[str, Any] = {"source": "local", "detail": "not selected yet"}


class RegistryIntegrityError(RuntimeError):
    """The registry served files that are not the ones recorded for that version."""


# ── Helpers ──────────────────────────────────────────────────────────────────


def _tracking_uri() -> str | None:
    return os.getenv("MLFLOW_TRACKING_URI") or None


def _check_reachable(uri: str, timeout: float = 3.0) -> None:
    """Fail fast on an HTTP tracking server that is down (the client would retry for minutes)."""
    if uri.startswith(("http://", "https://")):
        with urllib.request.urlopen(uri.rstrip("/") + "/health", timeout=timeout) as resp:
            if resp.status != 200:
                raise ConnectionError(f"MLflow health check returned {resp.status}")


def _client(uri: str):
    from mlflow.tracking import MlflowClient

    return MlflowClient(tracking_uri=uri)


def _local_paths() -> dict[str, Path]:
    return {name: getattr(model_cfg, attr) for name, attr in ARTIFACT_ATTRS.items()}


def _has_serving_bundle(client, run_id: str) -> bool:
    return any(True for _ in client.list_artifacts(run_id, SERVING_DIR))


def _point_at_production(client, version: str) -> None:
    """Alias (what the service reads) + deprecated stage (what the task sheet names)."""
    name = model_cfg.registry_name
    client.set_registered_model_alias(name, model_cfg.registry_alias, version)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # stages are deprecated since MLflow 2.9
        client.transition_model_version_stage(
            name, version, model_cfg.registry_stage, archive_existing_versions=True
        )


# ── Registration ─────────────────────────────────────────────────────────────


def register_bundle(uri: str, run_name: str | None = None, if_missing: bool = False) -> dict:
    """
    Register the local serving artifacts and point the production alias at them.

    if_missing: reuse a version that already holds these exact artifacts (same
    model_version tag and a serving/ bundle) instead of registering a new one.
    """
    import mlflow
    import numpy as np
    import pandas as pd
    from mlflow.models import infer_signature

    from src.artifacts import load_pickle

    name = model_cfg.registry_name
    served = model_version()
    manifest = artifact_manifest()
    client = _client(uri)
    mlflow.set_tracking_uri(uri)

    if if_missing:
        for mv in client.search_model_versions(f"name='{name}'"):
            if mv.tags.get("model_version") == served and _has_serving_bundle(client, mv.run_id):
                version = str(mv.version)  # int from SQL stores, str over REST
                _point_at_production(client, version)
                log.info("Registry already holds %s as version %s", served, version)
                return {"registered": False, "version": version, "model_version": served}

    bundle = load_pickle(model_cfg.artifact_path)
    features = bundle["features"]
    sample = pd.DataFrame(np.zeros((1, len(features))), columns=features)

    mlflow.set_experiment(project_name)
    with mlflow.start_run(run_name=run_name or f"{project_name}-v{project_version}") as run:
        mlflow.set_tags(
            {
                "project": project_name,
                "version": project_version,
                "model_version": served,
                "label_definition": model_cfg.label_definition,
                "pipeline": "inference-only",
                **{f"md5_{k}": v for k, v in manifest.items()},
            }
        )
        mlflow.log_params(bundle.get("hyperparameters", {}))
        mlflow.log_params(
            {
                "n_features": len(features),
                "requires_scaling": bundle.get("requires_scaling", False),
                "label_definition": bundle.get("label_definition", model_cfg.label_definition),
            }
        )
        for split, values in bundle.get("metrics", {}).items():
            if isinstance(values, dict):
                for key, value in values.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(metric_name(f"{split}_{key}"), float(value))

        # The four files the service needs, together — this is what it downloads
        for path in _local_paths().values():
            mlflow.log_artifact(str(path), artifact_path=SERVING_DIR)
        mlflow.log_dict(manifest, f"{SERVING_DIR}_manifest.json")

        info = mlflow.sklearn.log_model(
            sk_model=bundle["model"],
            name="sklearn_model",
            signature=infer_signature(sample, np.array([0])),
            input_example=sample,
            registered_model_name=name,
        )
        run_id = run.info.run_id

    version = str(info.registered_model_version)
    for key, value in {
        "model_version": served,
        **{f"md5_{k}": v for k, v in manifest.items()},
    }.items():
        client.set_model_version_tag(name, version, key, value)
    _point_at_production(client, version)
    log.info("Registered %s as %s version %s (run %s)", served, name, version, run_id)
    return {"registered": True, "version": version, "model_version": served, "run_id": run_id}


def metric_name(name: str) -> str:
    """MLflow names allow only letters, digits, _ - . space : / — "recall@5%" is refused."""
    name = name.replace("@", "_at_").replace("%", "pct")
    return re.sub(r"[^\w\-. :/]", "_", name)


# ── Loading ──────────────────────────────────────────────────────────────────


def resolve_from_registry(uri: str, dest: Path) -> dict:
    """Download the production version's serving files into `dest` and verify them."""
    import mlflow

    name, alias = model_cfg.registry_name, model_cfg.registry_alias
    client = _client(uri)
    mv = client.get_model_version_by_alias(name, alias)
    local_dir = Path(
        mlflow.artifacts.download_artifacts(
            run_id=mv.run_id, artifact_path=SERVING_DIR, dst_path=str(dest), tracking_uri=uri
        )
    )
    paths = {key: local_dir / path.name for key, path in _local_paths().items()}

    problems = []
    for key, path in paths.items():
        expected = mv.tags.get(f"md5_{key}")
        if not path.exists():
            problems.append(f"{key}: missing from the registry bundle")
        elif expected != file_md5(path):
            problems.append(f"{key}: md5 differs from the one recorded for version {mv.version}")
    if problems:
        raise RegistryIntegrityError(f"{name}@{alias} (version {mv.version}): {problems}")

    return {"version": str(mv.version), "run_id": mv.run_id, "paths": paths}


def activate(paths: dict[str, Path]) -> None:
    """Point the loaders at these files (call before anything is loaded)."""
    for key, attr in ARTIFACT_ATTRS.items():
        setattr(model_cfg, attr, paths[key])


def select_model_source() -> dict:
    """Decide where the artifacts come from; activate them. Called once at startup."""
    global _source
    uri = _tracking_uri()

    if model_cfg.source == "local" or uri is None:
        why = "model.source is local" if model_cfg.source == "local" else "no MLFLOW_TRACKING_URI"
        _source = {"source": "local", "detail": f"models/ ({why})"}
        log.info("Model source: %s", _source["detail"])
        return _source

    label = f"{model_cfg.registry_name}@{model_cfg.registry_alias}"
    try:
        _check_reachable(uri)
        found = resolve_from_registry(uri, Path(tempfile.mkdtemp(prefix="olist-registry-")))
    except RegistryIntegrityError:
        raise
    except Exception as e:  # unreachable, no such alias, mlflow not installed, ...
        if not model_cfg.registry_fallback_to_local:
            raise
        _source = {"source": "local", "detail": f"models/ (registry {label} unavailable: {e})"}
        log.warning("Model source: %s", _source["detail"])
        return _source

    activate(found["paths"])
    differences = dvc_mismatches(include_missing=False)
    if differences:
        raise RegistryIntegrityError(f"{label} differs from the DVC-recorded files: {differences}")

    _source = {
        "source": "registry",
        "detail": f"{label} (version {found['version']}, run {found['run_id']})",
        "registry_version": found["version"],
    }
    log.info("Model source: %s", _source["detail"])
    return _source


def current_source() -> dict:
    """Where the served artifacts came from (for /health and /model/info)."""
    return dict(_source)
