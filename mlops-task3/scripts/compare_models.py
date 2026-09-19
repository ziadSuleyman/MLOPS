"""
Compare the current local model against what MLflow has registered.

Usage:
    python scripts/compare_models.py

Prints a side-by-side table:  local artifacts vs. MLflow registry.
Useful before promoting a new version.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.artifacts import load_pickle, model_version  # noqa: E402
from src.config import model as model_cfg  # noqa: E402
from src.config import project_name, project_version  # noqa: E402


def local_info() -> dict:
    """Read info from the local model files."""
    bundle = load_pickle(model_cfg.artifact_path)
    with open(model_cfg.feature_list_path, encoding="utf-8") as f:
        fmeta = json.load(f)

    return {
        "source": "local",
        "service_version": project_version,
        "model_version": model_version(),
        "model_type": type(bundle["model"]).__name__,
        "n_features": len(bundle["features"]),
        "metrics": bundle.get("metrics", {}),
        "hyperparameters": bundle.get("hyperparameters", {}),
        "trained_on": fmeta.get("metadata", {}).get("trained_on_rows", "?"),
        "base_rate": fmeta.get("metadata", {}).get("base_rate", "?"),
    }


def mlflow_info() -> dict | None:
    """Try to get the latest registered model version from MLflow."""
    try:
        import mlflow  # noqa: E402

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient()

        versions = client.search_model_versions(f"name='{project_name}'")
        if not versions:
            return None

        latest = max(versions, key=lambda v: int(v.version))
        run = client.get_run(latest.run_id)

        return {
            "source": "mlflow",
            "version": latest.version,
            "model_version": latest.tags.get("model_version", "untagged"),
            "run_id": latest.run_id,
            "metrics": run.data.metrics,
            "params": run.data.params,
            "tags": {k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.")},
        }
    except Exception as e:
        return {"source": "mlflow", "error": str(e)}


def main() -> None:
    print("=" * 60)
    print("  MODEL COMPARISON - Local vs. MLflow Registry")
    print("=" * 60)

    loc = local_info()
    print(f"\n[local] Model ({model_cfg.artifact_path.name}):")
    for k, v in loc.items():
        if k != "source":
            print(f"   {k:20s}: {v}")

    mlf = mlflow_info()
    if mlf is None:
        print("\n[mlflow] No registered model found.")
        print("   -> Run:  python scripts/register_model.py")
    elif "error" in mlf:
        print(f"\n[mlflow] Could not connect - {mlf['error']}")
        print("   -> Make sure MLflow is running: docker compose up mlflow")
    else:
        print(f"\n[mlflow] Latest version {mlf['version']} (run {mlf['run_id'][:8]}):")
        for k, v in mlf.items():
            if k not in ("source", "run_id"):
                print(f"   {k:20s}: {v}")
        same = mlf["model_version"] == loc["model_version"]
        verdict = "MATCH" if same else "DIFFER FROM"
        print(f"\n   Local artifacts {verdict} the latest registered version.")

    print()


if __name__ == "__main__":
    main()
