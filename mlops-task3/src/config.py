"""
Central configuration loader.

Reads config/settings.yaml for structure, .env for secrets.
Every module imports `settings` from here — no hardcoded values anywhere else.
"""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"

# ── Load .env (secrets) ─────────────────────────────────────────────────────
load_dotenv(PROJECT_ROOT / ".env")


# ── Load YAML (structure) ───────────────────────────────────────────────────
def _load_yaml() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


_cfg = _load_yaml()


def _path_from_env(env_var: str, yaml_value: str) -> Path:
    """An env var (absolute or relative to the project root) overrides the YAML path."""
    return PROJECT_ROOT / os.getenv(env_var, yaml_value)


# ── Typed accessors ─────────────────────────────────────────────────────────
class _DB:
    host: str = os.getenv("DB_HOST", "localhost")
    port: int = int(os.getenv("DB_PORT", "5433"))
    name: str = os.getenv("DB_NAME", "olist")
    user: str = os.getenv("DB_USER", "olist")
    password: str = os.getenv("DB_PASSWORD", "")

    @property
    def url(self) -> str:
        return (
            f"postgresql+psycopg://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"
        )


MODEL_SOURCE_OPTIONS = ("registry", "local")


class _Model:
    source: str = os.getenv("MODEL_SOURCE", _cfg["model"]["source"])
    registry_name: str = _cfg["model"]["registry_name"]
    registry_alias: str = _cfg["model"]["registry_alias"]
    registry_stage: str = _cfg["model"]["registry_stage"]
    registry_fallback_to_local: bool = _cfg["model"]["registry_fallback_to_local"]
    # The artifact paths point at models/ until src/registry.py swaps in a registry download
    local_dir: Path = PROJECT_ROOT / "models"
    artifact_path: Path = PROJECT_ROOT / _cfg["model"]["artifact_path"]
    transformers_path: Path = PROJECT_ROOT / _cfg["model"]["transformers_path"]
    feature_list_path: Path = PROJECT_ROOT / _cfg["model"]["feature_list_path"]
    reference_path: Path = PROJECT_ROOT / _cfg["model"]["reference_path"]
    target: str = _cfg["model"]["target"]
    label_definition: str = _cfg["model"]["label_definition"]


class _Alerting:
    budget: float = _cfg["alerting"]["budget"]
    window: int = _cfg["alerting"]["window"]
    min_window: int = _cfg["alerting"]["min_window"]


class _API:
    title: str = _cfg["api"]["title"]
    host: str = os.getenv("API_HOST", "0.0.0.0")
    port: int = int(os.getenv("API_PORT", "8000"))
    docs_url: str = _cfg["api"]["docs_url"]
    health_path: str = _cfg["api"]["health_path"]


class _Logging:
    level: str = os.getenv("LOG_LEVEL", _cfg["logging"]["level"])
    fmt: str = _cfg["logging"]["format"]
    file: Path = _path_from_env("SERVICE_LOG_FILE", _cfg["logging"]["file"])
    max_bytes: int = _cfg["logging"]["max_bytes"]
    backup_count: int = _cfg["logging"]["backup_count"]


class _Monitoring:
    enable_prometheus: bool = _cfg["monitoring"]["enable_prometheus"]
    metrics_path: str = _cfg["monitoring"]["metrics_path"]
    prediction_log: Path = _path_from_env(
        "PREDICTION_LOG_FILE", _cfg["monitoring"]["prediction_log_file"]
    )
    drift_window: int = _cfg["monitoring"]["drift_window"]
    drift_min_window: int = _cfg["monitoring"]["drift_min_window"]
    psi_warn: float = _cfg["monitoring"]["psi_warn"]
    psi_alert: float = _cfg["monitoring"]["psi_alert"]


ON_FAILURE_OPTIONS = ("reject", "flag")


class _Validation:
    suite_name: str = _cfg["validation"]["suite_name"]
    on_failure: str = _cfg["validation"]["on_failure"]
    ge_on_batch: bool = _cfg["validation"]["ge_on_batch"]
    allowed_states: list[str] = _cfg["validation"]["allowed_states"]
    allowed_payment_types: list[str] = _cfg["validation"]["allowed_payment_types"]
    numeric_ranges: dict[str, tuple[float, float]] = {
        k: tuple(v) for k, v in _cfg["validation"]["numeric_ranges"].items()
    }
    lat_range: tuple[float, float] = tuple(_cfg["validation"]["coordinate_ranges"]["lat"])
    lng_range: tuple[float, float] = tuple(_cfg["validation"]["coordinate_ranges"]["lng"])


class _Inference:
    refit: bool = _cfg["inference"]["refit"]
    batch_max_size: int = _cfg["inference"]["batch_max_size"]


# ── Fail fast on settings the service cannot honour ─────────────────────────
if _Inference.refit:
    raise ValueError(
        "inference.refit is true in config/settings.yaml — this service only applies "
        "the objects fitted in Task 2 and never refits. Set it to false."
    )
if _Model.source not in MODEL_SOURCE_OPTIONS:
    raise ValueError(
        f"model.source (or MODEL_SOURCE) must be one of {MODEL_SOURCE_OPTIONS}, "
        f"got '{_Model.source}'"
    )
if _Validation.on_failure not in ON_FAILURE_OPTIONS:
    raise ValueError(
        f"validation.on_failure must be one of {ON_FAILURE_OPTIONS}, got '{_Validation.on_failure}'"
    )

# ── Public singletons ───────────────────────────────────────────────────────
db = _DB()
model = _Model()
alerting = _Alerting()
api = _API()
logging_cfg = _Logging()
monitoring = _Monitoring()
validation = _Validation()
inference = _Inference()
project_name: str = _cfg["project"]["name"]
project_version: str = _cfg["project"]["version"]
