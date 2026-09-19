"""
Register the trained model + transformers in MLflow.

Usage:
    python scripts/register_model.py                        # defaults
    python scripts/register_model.py --run-name "v1.0.0"    # custom run name

What it does:
    1.  Loads model, transformers, and feature list from models/
    2.  Creates an MLflow experiment "olist-late-predictor"
    3.  Logs:
        - model artifact  (06_model.joblib)
        - transformers    (05_transformers.joblib)
        - feature list    (05_feature_list.json)
        - serving reference (07_serving_reference.json)
        - hyperparameters as params
        - training metrics (from the model bundle)
        - model signature  (input schema + output schema)
    4.  Registers the model in the MLflow Model Registry as
        "olist-late-predictor" (or the name from settings.yaml).

Requires:
    - MLflow server running (docker compose up mlflow)
    - The artifact hash (src/artifacts.model_version) is stored as the tag
      "model_version" on the run and on the registered version, so the registry
      entry can be matched with what /health reports.
    - MLFLOW_TRACKING_URI in .env (default http://localhost:5000)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── project root on sys.path ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import mlflow  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

from src.artifacts import artifact_manifest, load_pickle, model_version  # noqa: E402
from src.config import model as model_cfg  # noqa: E402
from src.config import project_name, project_version  # noqa: E402


def metric_name(name: str) -> str:
    """MLflow names allow only letters, digits, _ - . space : / — "recall@5%" is refused."""
    name = name.replace("@", "_at_").replace("%", "pct")
    return re.sub(r"[^\w\-. :/]", "_", name)


def main(run_name: str | None = None) -> None:
    # ── 1. Resolve tracking URI ────────────────────────────────────────────
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    print(f"[MLflow] Tracking URI: {tracking_uri}")

    # ── 2. Experiment ──────────────────────────────────────────────────────
    experiment_name = project_name
    mlflow.set_experiment(experiment_name)
    print(f"[MLflow] Experiment: {experiment_name}")

    # ── 3. Load artifacts ──────────────────────────────────────────────────
    model_bundle = load_pickle(model_cfg.artifact_path)
    transformers_bundle = load_pickle(model_cfg.transformers_path)
    with open(model_cfg.feature_list_path, encoding="utf-8") as f:
        feature_meta = json.load(f)

    sk_model = model_bundle["model"]
    features = model_bundle["features"]
    metrics = model_bundle.get("metrics", {})
    hyperparams = model_bundle.get("hyperparameters", {})

    served_version = model_version()
    print(f"[MLflow] Model type: {type(sk_model).__name__}")
    print(f"[MLflow] Model version (artifact hash): {served_version}")
    print(f"[MLflow] Features: {len(features)}")
    print(f"[MLflow] Transformers keys: {list(transformers_bundle.keys())}")
    print(f"[MLflow] Feature list metadata: {list(feature_meta.get('metadata', {}).keys())}")

    # ── 4. Build a fake input sample for signature ─────────────────────────
    sample_input = pd.DataFrame(
        np.zeros((1, len(features))),
        columns=features,
    )
    sample_output = np.array([0])  # binary prediction

    signature = infer_signature(sample_input, sample_output)

    # ── 5. Start MLflow run ────────────────────────────────────────────────
    rn = run_name or f"{project_name}-v{project_version}"
    with mlflow.start_run(run_name=rn) as run:
        run_id = run.info.run_id
        print(f"[MLflow] Run ID: {run_id}")
        print(f"[MLflow] Run name: {rn}")

        # Tags
        mlflow.set_tag("project", project_name)
        mlflow.set_tag("version", project_version)
        mlflow.set_tag("model_version", served_version)
        mlflow.set_tag("label_definition", model_cfg.label_definition)
        for name, md5 in artifact_manifest().items():
            mlflow.set_tag(f"md5_{name}", md5)
        mlflow.set_tag("pipeline", "inference-only")

        # Hyperparameters
        for k, v in hyperparams.items():
            mlflow.log_param(k, v)
        mlflow.log_param("n_features", len(features))
        mlflow.log_param("requires_scaling", model_bundle.get("requires_scaling", False))
        mlflow.log_param("label_definition", model_bundle.get("label_definition", "calendar-day"))

        # Metrics — flatten nested dicts (validation/test splits)
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(metric_name(k), v)
            elif isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    if isinstance(sub_v, (int, float)):
                        mlflow.log_metric(metric_name(f"{k}_{sub_k}"), float(sub_v))

        # Artifacts — log the raw files
        mlflow.log_artifact(str(model_cfg.artifact_path), artifact_path="model")
        mlflow.log_artifact(str(model_cfg.transformers_path), artifact_path="transformers")
        mlflow.log_artifact(str(model_cfg.feature_list_path), artifact_path="metadata")
        mlflow.log_artifact(str(model_cfg.reference_path), artifact_path="metadata")

        # Log the sklearn model with signature for MLflow Model Registry
        info = mlflow.sklearn.log_model(
            sk_model=sk_model,
            name="sklearn_model",
            signature=signature,
            registered_model_name=project_name,
            input_example=sample_input,
        )
        MlflowClient().set_model_version_tag(
            project_name, info.registered_model_version, "model_version", served_version
        )

        print("[MLflow] OK  Artifacts logged")
        print(f"[MLflow] OK  Model registered as '{project_name}' v{info.registered_model_version}")
        print(f"[MLflow] OK  View at: {tracking_uri}/#/experiments")

    # ── 6. Summary ─────────────────────────────────────────────────────────
    print("\n-- Registration complete --")
    print(f"   Run ID:        {run_id}")
    print(f"   Experiment:    {experiment_name}")
    print(f"   Model name:    {project_name}")
    print(f"   Version:       {project_version}  (model {served_version})")
    print(f"   Features:      {len(features)}")
    print(f"   Metrics:       {metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register model in MLflow")
    parser.add_argument("--run-name", type=str, default=None, help="Custom MLflow run name")
    args = parser.parse_args()
    main(run_name=args.run_name)
