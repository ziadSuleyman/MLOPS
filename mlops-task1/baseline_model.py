"""Leakage-aware baseline for the late-delivery problem.

Two scenarios, so the value of each information stage is measurable:
  A. purchase-time  — everything known the moment the order is placed
  B. post-handover  — A plus the carrier pickup timestamp

Split is chronological (train < 2018-04-01 <= test), never random: a random
split would let the model see the future and inflate every score.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sqlalchemy import create_engine, text

warnings.filterwarnings("ignore")

ENGINE = create_engine("postgresql+psycopg://olist:olist123@localhost:5433/olist")
SPLIT_DATE = pd.Timestamp("2018-04-01")

QUERY = text("""
WITH geo AS (
    SELECT geolocation_zip_code_prefix AS zip,
           avg(geolocation_lat) AS lat, avg(geolocation_lng) AS lng
    FROM geolocation GROUP BY 1
),
agg AS (
    SELECT order_id,
           count(*)                       AS n_items,
           count(DISTINCT seller_id)      AS n_sellers,
           sum(price)                     AS total_price,
           sum(freight_value)             AS total_freight,
           min(seller_id)                 AS main_seller,
           min(product_id)                AS main_product
    FROM order_items GROUP BY 1
)
SELECT o.order_id,
       o.order_purchase_timestamp                       AS purchased_at,
       o.order_delivered_carrier_date                   AS handover_at,
       (o.order_delivered_customer_date
        > o.order_estimated_delivery_date)::int         AS is_late,
       EXTRACT(EPOCH FROM (o.order_estimated_delivery_date
                         - o.order_purchase_timestamp))/86400 AS promised_days,
       a.n_items, a.n_sellers, a.total_price, a.total_freight,
       a.main_seller,
       c.customer_state, s.seller_state,
       (c.customer_state = s.seller_state)::int         AS same_state,
       p.product_weight_g, p.product_length_cm, p.product_height_cm,
       p.product_width_cm, p.product_category_name,
       gc.lat AS cust_lat, gc.lng AS cust_lng,
       gs.lat AS sell_lat, gs.lng AS sell_lng
FROM orders o
JOIN agg a         USING (order_id)
JOIN customers c   USING (customer_id)
JOIN sellers s     ON s.seller_id  = a.main_seller
JOIN products p    ON p.product_id = a.main_product
LEFT JOIN geo gc   ON gc.zip = c.customer_zip_code_prefix
LEFT JOIN geo gs   ON gs.zip = s.seller_zip_code_prefix
WHERE o.order_status = 'delivered'
  AND o.order_delivered_customer_date IS NOT NULL
""")


def haversine_km(lat1, lon1, lat2, lon2):
    r1, r2 = np.radians(lat1), np.radians(lat2)
    dlat, dlon = r2 - r1, np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(r1) * np.cos(r2) * np.sin(dlon / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def main() -> None:
    df = pd.read_sql(QUERY, ENGINE)
    print(f"orders loaded: {len(df):,}   late rate: {df.is_late.mean():.2%}\n")

    df["distance_km"] = haversine_km(
        df.cust_lat, df.cust_lng, df.sell_lat, df.sell_lng
    )
    df["volume_cm3"] = (
        df.product_length_cm * df.product_height_cm * df.product_width_cm
    )
    df["purchase_month"] = df.purchased_at.dt.month
    df["purchase_dow"] = df.purchased_at.dt.dayofweek
    df["purchase_hour"] = df.purchased_at.dt.hour
    df["freight_ratio"] = df.total_freight / df.total_price.clip(lower=1)

    train = df[df.purchased_at < SPLIT_DATE].copy()
    test = df[df.purchased_at >= SPLIT_DATE].copy()
    print(f"train: {len(train):,} orders (late {train.is_late.mean():.2%})")
    print(f"test : {len(test):,} orders (late {test.is_late.mean():.2%})\n")

    # Seller history comes from TRAIN ONLY — computing it on the full frame
    # would leak the test period's outcomes back into the features.
    hist = train.groupby("main_seller").is_late.agg(["mean", "count"])
    prior, k = train.is_late.mean(), 20
    smoothed = (hist["mean"] * hist["count"] + prior * k) / (hist["count"] + k)
    for part in (train, test):
        part["seller_late_hist"] = part.main_seller.map(smoothed).fillna(prior)

    for part in (train, test):
        part["customer_state"] = part.customer_state.astype("category")
        part["seller_state"] = part.seller_state.astype("category")
        part["product_category_name"] = part.product_category_name.astype("category")

    base_features = [
        "promised_days", "n_items", "n_sellers", "total_price", "total_freight",
        "freight_ratio", "same_state", "distance_km", "product_weight_g",
        "volume_cm3", "purchase_month", "purchase_dow", "purchase_hour",
        "seller_late_hist", "customer_state", "seller_state",
        "product_category_name",
    ]
    cat_mask_base = [f in ("customer_state", "seller_state",
                           "product_category_name") for f in base_features]

    scenarios = {
        "A. purchase-time only": (base_features, cat_mask_base),
        "B. + carrier handover": (base_features + ["handover_days"],
                                  cat_mask_base + [False]),
    }
    for part in (train, test):
        part["handover_days"] = (
            part.handover_at - part.purchased_at
        ).dt.total_seconds() / 86400

    print(f"{'scenario':<24} {'ROC-AUC':>9} {'PR-AUC':>9}  "
          f"{'recall@top10%':>13}  {'lift':>5}")
    print("-" * 68)

    for name, (features, cat_mask) in scenarios.items():
        model = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_depth=6,
            categorical_features=cat_mask, random_state=42,
        )
        model.fit(train[features], train.is_late)
        proba = model.predict_proba(test[features])[:, 1]

        auc = roc_auc_score(test.is_late, proba)
        ap = average_precision_score(test.is_late, proba)
        cutoff = np.quantile(proba, 0.90)
        flagged = proba >= cutoff
        recall = test.is_late[flagged].sum() / test.is_late.sum()
        lift = test.is_late[flagged].mean() / test.is_late.mean()

        print(f"{name:<24} {auc:>9.4f} {ap:>9.4f}  "
              f"{recall:>12.1%}  {lift:>5.2f}x")

    print("\nBaseline for reference — always predict 'on time':")
    print(f"  accuracy {1 - test.is_late.mean():.2%}, "
          f"but it catches 0 of {int(test.is_late.sum())} late orders.")


if __name__ == "__main__":
    main()
