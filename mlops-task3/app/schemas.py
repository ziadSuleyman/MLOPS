"""
Pydantic schemas — validate request and response payloads.

The schema checks types only (and rejects NaN/Infinity). Business rules — ranges,
allowed states, dates in the right order — live in src/validation.py, so that
`validation.on_failure` decides what happens when they are broken.

Examples are a real order from the test split (8a9be36ffd78382f9ac518945e909636).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from src.config import inference as inference_cfg


class OrderInput(BaseModel):
    """A single order to predict. All fields available at purchase time."""

    model_config = ConfigDict(allow_inf_nan=False)

    # Not a model feature: the key that joins this prediction to the real delivery
    # date later (scripts/evaluate_outcomes.py). Without it the prediction is unscorable.
    order_id: str | None = Field(None, max_length=64, examples=["8a9be36ffd78382f9ac518945e909636"])

    order_purchase_timestamp: str = Field(..., examples=["2018-06-01 04:19:11"])
    order_estimated_delivery_date: str = Field(..., examples=["2018-07-05"])
    shipping_limit_first: str | None = Field(None, examples=["2018-06-11 04:30:37"])

    # Geography
    customer_state: str = Field(..., max_length=8, examples=["MG"])
    main_seller_state: str | None = Field(None, max_length=8, examples=["SP"])
    customer_lat: float | None = Field(None, examples=[-19.5175])
    customer_lng: float | None = Field(None, examples=[-42.6117])
    seller_lat: float | None = Field(None, examples=[-23.6524])
    seller_lng: float | None = Field(None, examples=[-46.7555])

    # Money
    freight_total: float | None = Field(None, examples=[18.43])
    freight_max: float | None = Field(None, examples=[18.43])
    items_price_total: float | None = Field(None, examples=[78.0])
    items_price_max: float | None = Field(None, examples=[78.0])
    payment_total: float | None = Field(None, examples=[96.43])
    installments_max: float | None = Field(None, examples=[4])
    n_payments: float | None = Field(None, examples=[1])
    main_payment_type: str | None = Field(None, max_length=32, examples=["credit_card"])

    # Basket
    n_items: float | None = Field(None, examples=[1])
    n_products: float | None = Field(None, examples=[1])
    n_sellers: float | None = Field(None, examples=[1])
    n_categories: float | None = Field(None, examples=[1])
    weight_g_total: float | None = Field(None, examples=[250])
    photos_avg: float | None = Field(None, examples=[4])

    # Category
    main_category: str | None = Field(None, max_length=64, examples=["relogios_presentes"])


class PredictionOutput(BaseModel):
    """Response for a single prediction."""

    order_id: str | None = Field(None, examples=["8a9be36ffd78382f9ac518945e909636"])
    prediction: str = Field(
        ...,
        examples=["on_time"],
        description='"late" = inside the alert budget (the riskiest 5% of recent orders)',
    )
    probability: float = Field(
        ..., ge=0, le=1, examples=[0.0221], description="Calibrated probability of a late delivery"
    )
    score: float = Field(
        ...,
        ge=0,
        le=1,
        examples=[0.3197],
        description="Raw model score (a ranking, class-balanced)",
    )
    risk_percentile: float = Field(
        ..., ge=0, le=100, examples=[20.8], description="Share of validation orders scoring lower"
    )
    alert_threshold: float = Field(..., examples=[0.7538], description="Score needed for 'late'")
    model_version: str = Field(..., examples=["f9fcc48ef9ef"])
    latency_ms: float | None = Field(None, examples=[44.4])
    validation_warnings: list[str] | None = Field(
        None, description="Rules this order broke; only with validation.on_failure = flag"
    )


class BatchInput(BaseModel):
    """A batch of orders."""

    orders: list[OrderInput] = Field(..., min_length=1, max_length=inference_cfg.batch_max_size)


class BatchOutput(BaseModel):
    """Response for a batch prediction."""

    predictions: list[PredictionOutput]
    count: int


class HealthOutput(BaseModel):
    """Health check response."""

    status: str = Field(..., examples=["healthy"])
    model_loaded: bool
    model_version: str
    service_version: str


class ModelInfoOutput(BaseModel):
    """Model metadata."""

    model_type: str
    model_version: str
    service_version: str
    n_features: int
    features: list[str]
    hyperparameters: dict
    label_definition: str
    requires_scaling: bool
    metrics: dict
    artifacts_md5: dict[str, str]
    calibration: dict
    alerting: dict


class DriftOutput(BaseModel):
    """Drift over the rolling window of recent predictions."""

    window_size: int
    min_window: int
    psi: float | None = Field(None, description="Score PSI vs the validation split")
    psi_status: str = Field(..., examples=["ok"])
    alert_ratio: float | None = Field(None, description="Share of recent orders flagged late")
    mean_probability: float | None = Field(None, description="Mean calibrated probability")


class ErrorOutput(BaseModel):
    """Error response."""

    detail: str
    errors: list | None = None
