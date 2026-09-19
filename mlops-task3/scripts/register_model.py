"""
Register the serving bundle in the MLflow Model Registry.

Usage:
    python scripts/register_model.py                    # always a new version
    python scripts/register_model.py --if-missing       # reuse a version with these exact files
    python scripts/register_model.py --run-name "v1.2.0"

What it does (src/registry.register_bundle):
    1.  Logs one run in the experiment "olist-late-predictor" with
        - params: hyperparameters, n_features, requires_scaling, label definition
        - metrics: validation + test metrics from the model bundle
        - tags: model_version (the artifact hash /health reports) and each file's md5
        - artifacts: serving/ = the four files the service loads
          (06_model.joblib, 05_transformers.joblib, 05_feature_list.json,
           07_serving_reference.json)
    2.  Registers the sklearn model (with its signature) as "olist-late-predictor".
    3.  Tags the version with model_version + md5s, gives it the alias "production"
        and the stage "Production".

The API (model.source: registry) loads whatever "production" points at, after
checking every md5 against these tags and against the DVC pointers.

`docker compose up` runs this with --if-missing before the API starts.

Requires:
    - MLflow server running (docker compose up mlflow)
    - MLFLOW_TRACKING_URI in .env (default http://localhost:5000)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import model as model_cfg  # noqa: E402
from src.registry import register_bundle  # noqa: E402


def main(run_name: str | None, if_missing: bool) -> None:
    uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    print(f"[MLflow] Tracking URI: {uri}")

    result = register_bundle(uri, run_name=run_name, if_missing=if_missing)

    name, alias = model_cfg.registry_name, model_cfg.registry_alias
    action = "Registered" if result["registered"] else "Already registered"
    print(f"[MLflow] {action}: {name} version {result['version']}")
    print(f"[MLflow] Model version (artifact hash): {result['model_version']}")
    print(
        f"[MLflow] Alias '{alias}' and stage '{model_cfg.registry_stage}' -> version {result['version']}"
    )
    print(f"[MLflow] UI: {uri}/#/models/{name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register the serving bundle in MLflow")
    parser.add_argument("--run-name", type=str, default=None, help="Custom MLflow run name")
    parser.add_argument(
        "--if-missing",
        action="store_true",
        help="Do not register again if a version with these exact artifacts exists",
    )
    args = parser.parse_args()
    main(args.run_name, args.if_missing)
