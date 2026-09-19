"""
Feature engineering — applies the fitted transformers saved in Task 2.

Every statistic (a median, a category list, a late-rate map) was learned on
the training set and saved in 05_transformers.joblib. This module loads them
once and applies them. Nothing is ever re-fitted.

Ported from: notebooks/05_features.ipynb, sections 4–7.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from src.artifacts import load_pickle
from src.config import model as model_cfg
from src.logger import log
from src.preprocessing import build_derived

# ── Module-level cache (loaded once at startup) ─────────────────────────────
_transformers: dict[str, Any] | None = None
_feature_columns: list[str] | None = None


def load_transformers() -> dict[str, Any]:
    """Load the fitted transformers from disk. Cached after the first call."""
    global _transformers, _feature_columns

    if _transformers is not None:
        return _transformers

    path = model_cfg.transformers_path
    log.info("Loading transformers from %s", path)
    _transformers = load_pickle(path)

    # Also load the feature list (the column-name contract)
    with open(model_cfg.feature_list_path, encoding="utf-8") as f:
        fl = json.load(f)
    _feature_columns = fl["features"]

    log.info(
        "Transformers loaded: %s  |  %d features",
        list(_transformers.keys()),
        len(_feature_columns),
    )
    return _transformers


def get_feature_columns() -> list[str]:
    """Return the ordered feature list. Calls load_transformers if needed."""
    if _feature_columns is None:
        load_transformers()
    return _feature_columns  # type: ignore[return-value]


# ── Category helpers (from NB5 section 4) ────────────────────────────────────


def _group_rare(series: pd.Series, kept: list[str]) -> pd.Series:
    """Categories outside the kept list → 'rare', gaps → 'unknown'."""
    return series.where(series.isin(kept), "rare").fillna("unknown")


def _apply_target_encoder(x: pd.Series, encoder: dict) -> pd.Series:
    """Map a category to its smoothed late rate. Unknown → the overall mean."""
    return x.map(encoder["mapping"]).fillna(encoder["prior"]).astype(float)


# ── Main entry point ─────────────────────────────────────────────────────────


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature pipeline: raw order → model-ready feature row.

    Parameters
    ----------
    df : DataFrame with the raw order columns.

    Returns
    -------
    DataFrame with exactly the 31 features in the correct order,
    no NaN, no infinite values, ready for model.predict_proba().
    """
    transformers = load_transformers()
    feature_cols = get_feature_columns()

    # 1) Derived features (pure row-level transforms)
    derived = build_derived(df)

    # 2) Target encoding — customer_state, main_seller_state, main_category
    #    Uses the full-train encoder (for val/test/production).
    kept_categories = transformers["kept_categories"]

    cats = pd.DataFrame(
        {
            "customer_state": df["customer_state"].fillna("unknown"),
            "main_seller_state": df["main_seller_state"].fillna("unknown"),
            "main_category": _group_rare(
                df["main_category"]
                if "main_category" in df.columns
                else pd.Series("unknown", index=df.index),
                kept_categories,
            ),
        }
    )

    for col in ["customer_state", "main_seller_state", "main_category"]:
        encoder = transformers["target_encoders"][col]
        derived[f"{col}_te"] = _apply_target_encoder(cats[col], encoder)

    # 3) One-hot encoding — payment type
    ohe = transformers["onehot_payment"]
    payment_col = (
        df["main_payment_type"].fillna("unknown")
        if "main_payment_type" in df.columns
        else pd.Series("unknown", index=df.index)
    )
    encoded = ohe.transform(payment_col.to_frame())
    ohe_names = [f"pay_{v}" for v in ohe.categories_[0]]
    for i, col_name in enumerate(ohe_names):
        derived[col_name] = encoded[:, i]

    # 4) Infinite values → missing, BEFORE imputing. The imputer rejects infinities,
    #    so this has to happen first; the imputer then fills them with the train medians.
    inf_count = int(np.isinf(derived[feature_cols].to_numpy(dtype=float)).sum())
    if inf_count > 0:
        log.warning("Infinite values found: %d — treated as missing", inf_count)
        derived[feature_cols] = derived[feature_cols].replace([np.inf, -np.inf], np.nan)

    # 5) Impute missing values with the train medians
    imputer = transformers["imputer"]
    filled = imputer.transform(derived[feature_cols])
    features = pd.DataFrame(filled, columns=feature_cols, index=df.index)

    # 6) Sanity checks
    assert list(features.columns) == feature_cols, "Column mismatch!"
    nan_count = int(features.isna().sum().sum())
    if nan_count > 0:
        log.warning("NaN remaining after imputation: %d — filling with 0", nan_count)
        features = features.fillna(0)

    return features
