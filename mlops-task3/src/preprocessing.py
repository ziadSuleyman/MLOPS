"""
Preprocessing — pure row-level transforms.

Every function here is a pure function of a single row (or DataFrame).
No statistics are learned, no fitted objects are used.
Applied identically to train, val, test, and production orders.

Ported from: notebooks/05_features.ipynb, section 2 (build_derived).
"""

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: pd.Series, lon1: pd.Series, lat2: pd.Series, lon2: pd.Series) -> pd.Series:
    """Great-circle distance in km between two points given in degrees."""
    lat1, lon1, lat2, lon2 = (np.radians(s) for s in (lat1, lon1, lat2, lon2))
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def build_derived(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute every feature that comes from the row itself, with no knowledge
    of the rest of the data. Applied literally unchanged to any split or
    to a single production order.

    Parameters
    ----------
    df : DataFrame with the raw order columns (from the DB or from JSON).

    Returns
    -------
    DataFrame with 23 derived columns, same index as df.
    """
    f = pd.DataFrame(index=df.index)
    ts = pd.to_datetime(df["order_purchase_timestamp"])

    # ── Promise and time (all known at the moment of purchase) ──────────
    estimated = pd.to_datetime(df["order_estimated_delivery_date"])
    f["promised_days"] = (estimated - ts).dt.total_seconds() / 86400

    shipping_limit = pd.to_datetime(df["shipping_limit_first"])
    f["shipping_limit_days"] = (shipping_limit - ts).dt.total_seconds() / 86400

    f["purchase_dow"] = ts.dt.dayofweek
    f["purchase_hour"] = ts.dt.hour

    # ── Geography ───────────────────────────────────────────────────────
    f["distance_km"] = haversine_km(
        df["customer_lat"].astype(float),
        df["customer_lng"].astype(float),
        df["seller_lat"].astype(float),
        df["seller_lng"].astype(float),
    )

    # Missingness is itself a signal (EDA finding #3) — built before filling
    f["customer_geo_missing"] = df["customer_lat"].isna().astype(int)
    f["seller_geo_missing"] = df["seller_lat"].isna().astype(int)
    f["same_state"] = (df["customer_state"] == df["main_seller_state"]).astype(int)

    # ── The strongest feature: promise relative to distance ─────────────
    hundreds_of_km = (f["distance_km"] / 100).replace(0, np.nan)
    f["promise_per_100km"] = f["promised_days"] / hundreds_of_km

    # ── Money and freight ───────────────────────────────────────────────
    f["freight_total"] = df["freight_total"].astype(float)
    f["freight_max"] = df["freight_max"].astype(float)
    f["freight_ratio"] = df["freight_total"].astype(float) / (
        df["items_price_total"].astype(float).replace(0, np.nan)
    )
    f["items_price_total"] = df["items_price_total"].astype(float)
    f["items_price_max"] = df["items_price_max"].astype(float)
    f["payment_total"] = df["payment_total"].astype(float)
    f["installments_max"] = df["installments_max"].astype(float)
    f["n_payments"] = df["n_payments"].astype(float)

    # ── Basket and product (weak signal, zero cost) ─────────────────────
    f["n_items"] = df["n_items"].astype(float)
    f["n_products"] = df["n_products"].astype(float)
    f["n_sellers"] = df["n_sellers"].astype(float)
    f["n_categories"] = df["n_categories"].astype(float)
    f["weight_g_total"] = df["weight_g_total"].astype(float)
    f["photos_avg"] = df["photos_avg"].astype(float)

    return f
