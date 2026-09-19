"""
Artifact integrity — what exactly is being served.

- load_pickle(): refuses to load a joblib file pickled by a different scikit-learn
  version (the model was fitted with 1.9.0; unpickling it elsewhere is undefined).
- model_version(): a hash of the four serving artifacts. Replace any file and the
  version changes on its own — no hand-edited version number to forget.
- dvc_mismatches(): compares each artifact with the md5 recorded in its .dvc file.
"""

from __future__ import annotations

import hashlib
import warnings
from pathlib import Path
from typing import Any

import joblib
import yaml
from sklearn.exceptions import InconsistentVersionWarning

from src.config import model as model_cfg

_version: str | None = None


def serving_artifacts() -> dict[str, Path]:
    """Every file that changes what the service returns, in a fixed order."""
    return {
        "model": model_cfg.artifact_path,
        "transformers": model_cfg.transformers_path,
        "feature_list": model_cfg.feature_list_path,
        "reference": model_cfg.reference_path,
    }


def file_md5(path: Path) -> str:
    """md5 of a file — the same hash DVC records."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_manifest() -> dict[str, str]:
    """{artifact name: md5} for the /model/info endpoint and MLflow tags."""
    return {name: file_md5(path) for name, path in serving_artifacts().items()}


def compute_model_version(manifest: dict[str, str]) -> str:
    """12 hex characters derived from the artifact hashes."""
    joined = "\n".join(f"{name}:{md5}" for name, md5 in manifest.items())
    return hashlib.sha256(joined.encode()).hexdigest()[:12]


def model_version() -> str:
    """The served model version. Computed once per process."""
    global _version
    if _version is None:
        _version = compute_model_version(artifact_manifest())
    return _version


def dvc_mismatches() -> list[str]:
    """Artifacts whose content differs from the md5 in their .dvc pointer file."""
    problems = []
    for name, path in serving_artifacts().items():
        pointer = path.with_name(path.name + ".dvc")
        if not pointer.exists():
            problems.append(f"{name}: no DVC pointer file ({pointer.name})")
            continue
        with open(pointer, encoding="utf-8") as f:
            recorded = yaml.safe_load(f)["outs"][0]["md5"]
        if recorded != file_md5(path):
            problems.append(f"{name}: file differs from the version recorded in {pointer.name}")
    return problems


def load_pickle(path: Path) -> Any:
    """joblib.load, but a scikit-learn version mismatch is an error, not a warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        try:
            return joblib.load(path)
        except InconsistentVersionWarning as w:
            raise RuntimeError(
                f"{path.name} was pickled with scikit-learn {w.original_sklearn_version} "
                f"but {w.current_sklearn_version} is installed. Install the pinned version "
                f"from requirements/base.txt."
            ) from w
