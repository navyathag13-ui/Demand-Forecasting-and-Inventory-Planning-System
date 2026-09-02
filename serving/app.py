"""
serving/app.py — FastAPI serving layer on top of the batch pipeline in src/.

Near-real-time, not streaming: the grain of the data is weekly store-level
sales, so "live" here means "reflects the freshest ingested weekly record,"
not sub-second updates. The model is trained offline (scripts/train_and_persist.py)
and loaded once at startup; this process never retrains.
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.utils import setup_logging
from .feature_state import DuplicateOrOutOfOrderEventError, FeatureStateStore
from .inference import InferenceError, predict_next_week
from .inventory import build_live_inventory_plan
from .metrics import MetricsTracker
from .model_store import UnknownStoreError, load_bundle
from .schemas import (
    ForecastOut,
    HealthOut,
    InventoryOut,
    MetricsOut,
    SalesEventIn,
    SalesEventOut,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    bundle = load_bundle(config.MODEL_PATH)
    app.state.bundle = bundle
    app.state.feature_store = FeatureStateStore(bundle)
    app.state.metrics = MetricsTracker()
    app.state.started_at = datetime.now(timezone.utc)
    logger.info("Serving layer ready: %d stores loaded.", len(bundle.stores))
    yield


app = FastAPI(title="Demand Forecasting — Serving Layer", lifespan=lifespan)


@app.post("/events/sales", response_model=SalesEventOut, status_code=201)
def ingest_sales_event(payload: SalesEventIn) -> SalesEventOut:
    store: FeatureStateStore = app.state.feature_store
    try:
        event, snapshot = store.record_event(
            store_id=payload.store_id,
            week_ending_date=payload.week_ending_date,
            sales=payload.sales,
            is_holiday=payload.is_holiday,
        )
    except UnknownStoreError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except DuplicateOrOutOfOrderEventError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    app.state.metrics.record_event_ingested()

    return SalesEventOut(
        store_id=event.store_id,
        week_ending_date=event.week_ending_date.date(),
        sales=event.sales,
        is_holiday=event.is_holiday,
        received_at=event.received_at,
        target_week=snapshot.target_week.date(),
    )


@app.get("/forecast/{store_id}", response_model=ForecastOut)
def get_forecast(store_id: int) -> ForecastOut:
    bundle = app.state.bundle
    store: FeatureStateStore = app.state.feature_store
    try:
        snapshot = store.latest_snapshot(store_id)
    except UnknownStoreError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if snapshot is None:
        raise HTTPException(
            status_code=404,
            detail=f"Store {store_id} has no sales history yet -- "
                   f"POST /events/sales for it before requesting a forecast.",
        )

    t0 = time.perf_counter()
    try:
        predicted = predict_next_week(bundle, snapshot)
    except InferenceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    app.state.metrics.record_forecast_served(time.perf_counter() - t0)

    return ForecastOut(
        store_id=store_id,
        predicted_demand=round(predicted, 2),
        based_on_week=snapshot.based_on_week.date(),
        target_week=snapshot.target_week.date(),
        computed_at=snapshot.computed_at,
    )


@app.get("/inventory/{store_id}", response_model=InventoryOut)
def get_inventory(store_id: int) -> InventoryOut:
    bundle = app.state.bundle
    store: FeatureStateStore = app.state.feature_store
    try:
        snapshot = store.latest_snapshot(store_id)
    except UnknownStoreError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if snapshot is None:
        raise HTTPException(
            status_code=404,
            detail=f"Store {store_id} has no sales history yet -- "
                   f"POST /events/sales for it before requesting an inventory plan.",
        )

    t0 = time.perf_counter()
    try:
        predicted = predict_next_week(bundle, snapshot)
    except InferenceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    app.state.metrics.record_forecast_served(time.perf_counter() - t0)

    plan = build_live_inventory_plan(
        store_id,
        predicted,
        bundle,
        lead_time_weeks=config.LEAD_TIME_WEEKS,
        z_score=config.Z_SCORE,
        ordering_cost=config.ORDERING_COST_USD,
        unit_value=config.UNIT_VALUE_USD,
        holding_cost_pct=config.UNIT_HOLDING_COST_PCT,
        review_period_weeks=config.REVIEW_PERIOD_WEEKS,
    )

    return InventoryOut(
        store_id=plan.store_id,
        demand_used=plan.demand_used,
        avg_weekly_demand=plan.avg_weekly_demand,
        safety_stock=plan.safety_stock,
        reorder_point=plan.reorder_point,
        eoq=plan.eoq,
        recommended_order_qty=plan.recommended_order_qty,
        replenishment_alert=plan.replenishment_alert,
        based_on_week=snapshot.based_on_week.date(),
        target_week=snapshot.target_week.date(),
    )


@app.get("/health", response_model=HealthOut)
def health() -> HealthOut:
    bundle = app.state.bundle
    store: FeatureStateStore = app.state.feature_store
    uptime = (datetime.now(timezone.utc) - app.state.started_at).total_seconds()

    return HealthOut(
        status="ok",
        model_loaded=True,
        uptime_seconds=round(uptime, 1),
        trained_at=bundle.trained_at,
        holdout_mape=bundle.holdout_mape,
        total_stores=len(bundle.stores),
        stores_with_data=store.stores_with_data(),
        last_event_received_at=store.last_event_received_at(),
    )


@app.get("/metrics", response_model=MetricsOut)
def metrics() -> MetricsOut:
    snap = app.state.metrics.snapshot()
    return MetricsOut(
        total_events_ingested=snap.total_events_ingested,
        total_forecasts_served=snap.total_forecasts_served,
        average_inference_latency_ms=snap.average_inference_latency_ms,
    )
