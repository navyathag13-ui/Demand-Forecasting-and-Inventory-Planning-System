"""serving/schemas.py — Request/response models for the serving API."""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, Field


class SalesEventIn(BaseModel):
    store_id: int
    week_ending_date: date
    # Sales are dollar-denominated demand, same as train.csv. Negative
    # values are excluded from training as "not demand signals" (returns
    # / reversals) -- the live feed holds to the same definition rather
    # than silently accepting a value the model was never trained to
    # expect from this field.
    sales: float = Field(ge=0)
    is_holiday: bool = False


class SalesEventOut(BaseModel):
    store_id: int
    week_ending_date: date
    sales: float
    is_holiday: bool
    received_at: datetime
    target_week: date  # the week this event's ingestion just updated the forecast for


class ForecastOut(BaseModel):
    store_id: int
    predicted_demand: float
    based_on_week: date       # most recent actual week of data behind this forecast
    target_week: date         # the week being forecasted
    computed_at: datetime     # when the underlying feature vector was last updated


class InventoryOut(BaseModel):
    store_id: int
    demand_used: float
    avg_weekly_demand: float
    safety_stock: float
    reorder_point: float
    eoq: float
    recommended_order_qty: float
    replenishment_alert: bool
    based_on_week: date
    target_week: date


class HealthOut(BaseModel):
    status: str
    model_loaded: bool
    uptime_seconds: float
    trained_at: str
    holdout_mape: float
    total_stores: int
    stores_with_data: int
    last_event_received_at: Optional[datetime]


class MetricsOut(BaseModel):
    total_events_ingested: int
    total_forecasts_served: int
    average_inference_latency_ms: float
