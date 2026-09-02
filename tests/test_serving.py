"""
tests/test_serving.py — Unit tests for the serving layer's event ingestion
and feature recomputation, plus HTTP-level tests for POST /events/sales.

Uses a small synthetic ModelBundle (not the real trained artifact) so these
tests run fast and don't depend on scripts/train_and_persist.py having been
run -- same philosophy as tests/test_core.py's synthetic DataFrames.

Run with:  python -m pytest tests/test_serving.py -v
"""

import math
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from serving.feature_state import (
    DuplicateOrOutOfOrderEventError,
    FeatureStateStore,
    StoreHistory,
    build_feature_snapshot,
)
from serving.model_store import DemandStats, ModelBundle, StoreInfo, UnknownStoreError


class _DummyModel:
    """Stands in for the fitted LGBMForecaster. Returns a fixed prediction
    (default) so forecast/inventory tests can assert on an exact, known
    output instead of needing the real LightGBM model."""

    def __init__(self, fixed_prediction: float = 1000.0):
        self.fixed_prediction = fixed_prediction
        self.last_X = None  # last DataFrame passed to predict(), for inspection

    def predict(self, X):
        self.last_X = X
        return [self.fixed_prediction] * len(X)


def make_bundle(store_ids=(1, 2), model=None, avg_weekly_demand=1000.0, std_weekly_demand=100.0) -> ModelBundle:
    stores = {
        sid: StoreInfo(store_id=sid, type="A" if sid % 2 else "B", size=150_000.0 + sid)
        for sid in store_ids
    }
    demand_stats = {
        sid: DemandStats(avg_weekly_demand=avg_weekly_demand, std_weekly_demand=std_weekly_demand)
        for sid in store_ids
    }
    return ModelBundle(
        model=model or _DummyModel(),
        feature_columns=[
            "trend", "week_sin", "week_cos", "month_sin", "month_cos",
            "quarter", "is_month_end", "IsHoliday", "type_code", "size_norm",
            "sales_lag_1", "sales_lag_2", "sales_lag_4", "sales_lag_12",
            "sales_roll_4", "sales_roll_12",
        ],
        size_mean=150_000.0,
        size_std=10_000.0,
        min_date=pd.Timestamp("2020-01-05"),
        stores=stores,
        demand_stats=demand_stats,
        trained_at="2026-01-01T00:00:00+00:00",
        holdout_mape=4.1,
        holdout_r2=0.989,
        holdout_weeks=12,
    )


# ===========================================================================
# Feature recomputation
# ===========================================================================

class TestFeatureRecomputation:

    def test_first_event_sets_lag_1_only(self):
        bundle = make_bundle()
        history = StoreHistory(store_id=1)
        event, snapshot = history.record_event(
            week_ending_date="2020-02-07", sales=1000.0, is_holiday=False, bundle=bundle,
        )
        assert snapshot.features["sales_lag_1"] == pytest.approx(1000.0)
        # Not enough history yet for lag_2/4/12 -- NaN, not a fabricated 0.
        assert math.isnan(snapshot.features["sales_lag_2"])
        assert math.isnan(snapshot.features["sales_lag_12"])
        # Rolling mean over a single observation is just that observation.
        assert snapshot.features["sales_roll_4"] == pytest.approx(1000.0)

    def test_lag_chain_after_several_events(self):
        bundle = make_bundle()
        history = StoreHistory(store_id=1)
        weeks = pd.date_range("2020-01-05", periods=5, freq="7D")
        sales_values = [100.0, 200.0, 300.0, 400.0, 500.0]
        snapshot = None
        for wk, sales in zip(weeks, sales_values):
            _, snapshot = history.record_event(wk, sales, False, bundle)

        # lag_1 = most recent (500), lag_2 = one before that (400), lag_4 = 200.
        assert snapshot.features["sales_lag_1"] == pytest.approx(500.0)
        assert snapshot.features["sales_lag_2"] == pytest.approx(400.0)
        assert snapshot.features["sales_lag_4"] == pytest.approx(200.0)
        assert math.isnan(snapshot.features["sales_lag_12"])  # only 5 weeks of history
        assert snapshot.features["sales_roll_4"] == pytest.approx((200 + 300 + 400 + 500) / 4)

    def test_target_week_is_one_week_after_latest(self):
        bundle = make_bundle()
        history = StoreHistory(store_id=1)
        _, snapshot = history.record_event("2020-02-07", 1000.0, False, bundle)
        assert snapshot.based_on_week == pd.Timestamp("2020-02-07")
        assert snapshot.target_week == pd.Timestamp("2020-02-14")

    def test_calendar_features_computed_for_target_week_not_based_on_week(self):
        # 2020-03-29 -> target 2020-04-05: crosses a month boundary, so
        # month_sin/cos should reflect April, not March, if trend is really
        # anchored to the forecasted week.
        bundle = make_bundle()
        history = StoreHistory(store_id=1)
        _, snapshot = history.record_event("2020-03-29", 1000.0, False, bundle)
        expected_month_sin = math.sin(2 * math.pi * 4 / 12)  # April = month 4
        assert snapshot.features["month_sin"] == pytest.approx(expected_month_sin)

    def test_trend_anchored_to_bundle_min_date(self):
        bundle = make_bundle()
        history = StoreHistory(store_id=1)
        # min_date is 2020-01-05; based_on_week 2020-01-12 -> target 2020-01-19
        # -> trend = 14 days / 7 = 2.
        _, snapshot = history.record_event("2020-01-12", 1000.0, False, bundle)
        assert snapshot.features["trend"] == pytest.approx(2.0)

    def test_size_norm_and_type_code_from_store_catalog(self):
        bundle = make_bundle(store_ids=(1,))  # store 1 -> type "A" (odd id)
        history = StoreHistory(store_id=1)
        _, snapshot = history.record_event("2020-01-12", 1000.0, False, bundle)
        assert snapshot.features["type_code"] == pytest.approx(0.0)  # "A" -> 0
        expected_size_norm = (150_001.0 - 150_000.0) / 10_000.0
        assert snapshot.features["size_norm"] == pytest.approx(expected_size_norm)

    def test_build_feature_snapshot_raises_on_empty_history(self):
        bundle = make_bundle()
        with pytest.raises(ValueError):
            build_feature_snapshot(1, [], bundle)


# ===========================================================================
# Out-of-order / duplicate rejection
# ===========================================================================

class TestOutOfOrderRejection:

    def test_duplicate_week_rejected(self):
        bundle = make_bundle()
        history = StoreHistory(store_id=1)
        history.record_event("2020-02-07", 1000.0, False, bundle)
        with pytest.raises(DuplicateOrOutOfOrderEventError):
            history.record_event("2020-02-07", 1200.0, False, bundle)

    def test_earlier_week_rejected(self):
        bundle = make_bundle()
        history = StoreHistory(store_id=1)
        history.record_event("2020-02-14", 1000.0, False, bundle)
        with pytest.raises(DuplicateOrOutOfOrderEventError):
            history.record_event("2020-02-07", 900.0, False, bundle)

    def test_rejected_event_does_not_change_history_or_snapshot(self):
        bundle = make_bundle()
        history = StoreHistory(store_id=1)
        history.record_event("2020-02-14", 1000.0, False, bundle)
        snapshot_before = history.latest_snapshot
        with pytest.raises(DuplicateOrOutOfOrderEventError):
            history.record_event("2020-02-14", 1000.0, False, bundle)
        assert history.event_count == 1
        assert history.latest_snapshot is snapshot_before  # unchanged, same object

    def test_later_week_after_a_rejection_still_works(self):
        bundle = make_bundle()
        history = StoreHistory(store_id=1)
        history.record_event("2020-02-14", 1000.0, False, bundle)
        with pytest.raises(DuplicateOrOutOfOrderEventError):
            history.record_event("2020-02-07", 900.0, False, bundle)
        # A store rejecting a bad event shouldn't leave it unable to accept
        # a subsequent, correctly-ordered one.
        _, snapshot = history.record_event("2020-02-21", 1100.0, False, bundle)
        assert snapshot.features["sales_lag_1"] == pytest.approx(1100.0)


# ===========================================================================
# FeatureStateStore -- unknown store handling
# ===========================================================================

class TestFeatureStateStore:

    def test_unknown_store_rejected(self):
        bundle = make_bundle(store_ids=(1, 2))
        store = FeatureStateStore(bundle)
        with pytest.raises(UnknownStoreError):
            store.record_event(store_id=999, week_ending_date="2020-01-12", sales=100.0, is_holiday=False)

    def test_known_stores_are_independent(self):
        bundle = make_bundle(store_ids=(1, 2))
        store = FeatureStateStore(bundle)
        store.record_event(1, "2020-01-12", 100.0, False)
        store.record_event(2, "2020-01-12", 999.0, False)
        assert store.latest_snapshot(1).features["sales_lag_1"] == pytest.approx(100.0)
        assert store.latest_snapshot(2).features["sales_lag_1"] == pytest.approx(999.0)

    def test_store_with_no_events_has_no_snapshot(self):
        bundle = make_bundle(store_ids=(1, 2))
        store = FeatureStateStore(bundle)
        assert store.latest_snapshot(1) is None


# ===========================================================================
# Concurrency: same-store events arriving close together
# ===========================================================================

class TestConcurrentIngestion:

    def test_concurrent_distinct_weeks_all_recorded_without_loss(self):
        bundle = make_bundle(store_ids=(1,))
        history = StoreHistory(store_id=1)
        weeks = list(pd.date_range("2020-01-05", periods=20, freq="7D"))
        errors = []

        def ingest(wk):
            try:
                history.record_event(wk, 100.0, False, bundle)
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(exc)

        threads = [threading.Thread(target=ingest, args=(wk,)) for wk in weeks]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert history.event_count == len(weeks)
        # The lock serializes appends, so the final snapshot must be based on
        # the chronologically last week, not whichever thread happened to
        # finish last.
        assert history.latest_snapshot.based_on_week == weeks[-1]

    def test_concurrent_same_week_only_one_wins(self):
        bundle = make_bundle(store_ids=(1,))
        history = StoreHistory(store_id=1)
        results = []

        def ingest():
            try:
                history.record_event("2020-05-01", 100.0, False, bundle)
                results.append("ok")
            except DuplicateOrOutOfOrderEventError:
                results.append("rejected")

        threads = [threading.Thread(target=ingest) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Exactly one thread's event should have been recorded; the rest
        # rejected as duplicates of it -- never a torn/double-counted state.
        assert results.count("ok") == 1
        assert results.count("rejected") == 9
        assert history.event_count == 1


# ===========================================================================
# HTTP layer -- POST /events/sales
# ===========================================================================

@pytest.fixture
def client(monkeypatch):
    from fastapi.testclient import TestClient
    from serving import app as app_module

    bundle = make_bundle(store_ids=(1, 2))
    monkeypatch.setattr(app_module, "load_bundle", lambda path: bundle)
    with TestClient(app_module.app) as c:
        yield c


class TestIngestEndpoint:

    def test_ingest_happy_path(self, client):
        resp = client.post("/events/sales", json={
            "store_id": 1, "week_ending_date": "2020-02-07",
            "sales": 1000.0, "is_holiday": False,
        })
        assert resp.status_code == 201
        body = resp.json()
        assert body["store_id"] == 1
        assert body["target_week"] == "2020-02-14"

    def test_ingest_unknown_store_is_404(self, client):
        resp = client.post("/events/sales", json={
            "store_id": 999, "week_ending_date": "2020-02-07", "sales": 1000.0,
        })
        assert resp.status_code == 404

    def test_ingest_duplicate_week_is_409(self, client):
        payload = {"store_id": 1, "week_ending_date": "2020-02-07", "sales": 1000.0}
        assert client.post("/events/sales", json=payload).status_code == 201
        resp = client.post("/events/sales", json=payload)
        assert resp.status_code == 409

    def test_ingest_out_of_order_week_is_409(self, client):
        client.post("/events/sales", json={
            "store_id": 1, "week_ending_date": "2020-02-14", "sales": 1000.0,
        })
        resp = client.post("/events/sales", json={
            "store_id": 1, "week_ending_date": "2020-02-07", "sales": 900.0,
        })
        assert resp.status_code == 409

    def test_ingest_negative_sales_is_422(self, client):
        resp = client.post("/events/sales", json={
            "store_id": 1, "week_ending_date": "2020-02-07", "sales": -50.0,
        })
        assert resp.status_code == 422

    def test_ingest_defaults_is_holiday_false(self, client):
        resp = client.post("/events/sales", json={
            "store_id": 1, "week_ending_date": "2020-02-07", "sales": 1000.0,
        })
        assert resp.json()["is_holiday"] is False


# ===========================================================================
# Pure inference / inventory logic (no HTTP)
# ===========================================================================

class TestInference:

    def test_predict_next_week_returns_model_output(self):
        from serving.inference import predict_next_week

        model = _DummyModel(fixed_prediction=42_000.0)
        bundle = make_bundle(store_ids=(1,), model=model)
        history = StoreHistory(store_id=1)
        _, snapshot = history.record_event("2020-02-07", 1000.0, False, bundle)

        result = predict_next_week(bundle, snapshot)
        assert result == pytest.approx(42_000.0)
        # The DataFrame handed to the model should carry the snapshot's features.
        assert model.last_X.iloc[0]["sales_lag_1"] == pytest.approx(1000.0)


@pytest.mark.skipif(
    not config.MODEL_PATH.exists(),
    reason="Run `python scripts/train_and_persist.py` first to exercise this against the real model.",
)
class TestRealModelIntegration:
    """
    Every other forecast/inventory test uses _DummyModel, which ignores its
    input entirely -- so nothing else here would catch feature_state.py's
    generated dict keys drifting from the real LGBM_FEATURES list (the real
    LGBMForecaster._select() silently drops any missing column rather than
    raising, so a drift would degrade predictions quietly, not fail loudly).
    This loads the actual persisted model and runs a real feature snapshot
    through it end to end.
    """

    def test_predict_next_week_against_real_persisted_model(self):
        from serving.inference import predict_next_week
        from serving.model_store import load_bundle

        bundle = load_bundle(config.MODEL_PATH)
        any_store_id = next(iter(bundle.stores))
        history = StoreHistory(store_id=any_store_id)

        weeks = pd.date_range("2020-01-05", periods=13, freq="7D")
        snapshot = None
        for wk in weeks:
            _, snapshot = history.record_event(wk, 1_000_000.0, False, bundle)

        # By week 13, lag_12 and roll_12 both have full history -- no NaN
        # left for fillna(0) to paper over silently.
        assert not math.isnan(snapshot.features["sales_lag_12"])
        assert set(snapshot.features) >= set(bundle.feature_columns)

        prediction = predict_next_week(bundle, snapshot)
        assert math.isfinite(prediction)
        assert prediction >= 0  # LGBMForecaster.predict() clips negatives


class TestLiveInventoryPlan:

    def test_no_alert_when_forecast_near_average(self):
        from serving.inventory import build_live_inventory_plan

        bundle = make_bundle(store_ids=(1,), avg_weekly_demand=1000.0, std_weekly_demand=100.0)
        plan = build_live_inventory_plan(
            1, forecast=1000.0, bundle=bundle,
            lead_time_weeks=2, z_score=1.645, ordering_cost=150,
            unit_value=25, holding_cost_pct=0.2, review_period_weeks=4,
        )
        assert plan.replenishment_alert is False

    def test_alert_fires_above_10_percent_over_average(self):
        from serving.inventory import build_live_inventory_plan

        bundle = make_bundle(store_ids=(1,), avg_weekly_demand=1000.0, std_weekly_demand=100.0)
        plan = build_live_inventory_plan(
            1, forecast=1101.0, bundle=bundle,  # just over the 1.10x threshold
            lead_time_weeks=2, z_score=1.645, ordering_cost=150,
            unit_value=25, holding_cost_pct=0.2, review_period_weeks=4,
        )
        assert plan.replenishment_alert is True

    def test_matches_batch_formulas_directly(self):
        from src.inventory import compute_safety_stock, compute_reorder_point, compute_eoq
        from serving.inventory import build_live_inventory_plan

        bundle = make_bundle(store_ids=(1,), avg_weekly_demand=1000.0, std_weekly_demand=200.0)
        plan = build_live_inventory_plan(
            1, forecast=1200.0, bundle=bundle,
            lead_time_weeks=2, z_score=1.645, ordering_cost=150,
            unit_value=25, holding_cost_pct=0.2, review_period_weeks=4,
        )
        expected_ss = compute_safety_stock(200.0, 2, 1.645)
        expected_rop = compute_reorder_point(1200.0, 200.0, 2, 1.645)
        expected_eoq = compute_eoq(1200.0, 150, 25, 0.2)
        assert plan.safety_stock == pytest.approx(round(expected_ss, 2))
        assert plan.reorder_point == pytest.approx(round(expected_rop, 2))
        assert plan.eoq == pytest.approx(round(expected_eoq, 2))


# ===========================================================================
# HTTP layer -- GET /forecast/{store_id}
# ===========================================================================

def make_client(monkeypatch, bundle):
    from fastapi.testclient import TestClient
    from serving import app as app_module

    monkeypatch.setattr(app_module, "load_bundle", lambda path: bundle)
    return TestClient(app_module.app)


class TestForecastEndpoint:

    def test_forecast_for_store_with_no_history_is_404(self, monkeypatch):
        bundle = make_bundle(store_ids=(1, 2))
        with make_client(monkeypatch, bundle) as client:
            resp = client.get("/forecast/1")
        assert resp.status_code == 404

    def test_forecast_unknown_store_is_404(self, monkeypatch):
        bundle = make_bundle(store_ids=(1, 2))
        with make_client(monkeypatch, bundle) as client:
            resp = client.get("/forecast/999")
        assert resp.status_code == 404

    def test_forecast_after_ingesting_returns_prediction(self, monkeypatch):
        model = _DummyModel(fixed_prediction=55_000.0)
        bundle = make_bundle(store_ids=(1,), model=model)
        with make_client(monkeypatch, bundle) as client:
            client.post("/events/sales", json={
                "store_id": 1, "week_ending_date": "2020-02-07", "sales": 1000.0,
            })
            resp = client.get("/forecast/1")
        assert resp.status_code == 200
        body = resp.json()
        assert body["predicted_demand"] == pytest.approx(55_000.0)
        assert body["based_on_week"] == "2020-02-07"
        assert body["target_week"] == "2020-02-14"

    def test_forecast_reflects_most_recent_event_not_stale_data(self, monkeypatch):
        # Two different fixed predictions depending on what's fed in would
        # need a smarter dummy model; instead, assert on based_on_week/
        # target_week shifting after a second ingest -- that's what proves
        # the endpoint is reading the *current* snapshot, not a cached one
        # from the first event.
        bundle = make_bundle(store_ids=(1,))
        with make_client(monkeypatch, bundle) as client:
            client.post("/events/sales", json={
                "store_id": 1, "week_ending_date": "2020-02-07", "sales": 1000.0,
            })
            first = client.get("/forecast/1").json()
            client.post("/events/sales", json={
                "store_id": 1, "week_ending_date": "2020-02-14", "sales": 1100.0,
            })
            second = client.get("/forecast/1").json()

        assert first["based_on_week"] == "2020-02-07"
        assert second["based_on_week"] == "2020-02-14"
        assert second["target_week"] == "2020-02-21"


# ===========================================================================
# HTTP layer -- GET /inventory/{store_id}
# ===========================================================================

class TestInventoryEndpoint:

    def test_inventory_for_store_with_no_history_is_404(self, monkeypatch):
        bundle = make_bundle(store_ids=(1,))
        with make_client(monkeypatch, bundle) as client:
            resp = client.get("/inventory/1")
        assert resp.status_code == 404

    def test_inventory_after_ingesting_returns_plan(self, monkeypatch):
        model = _DummyModel(fixed_prediction=1200.0)
        bundle = make_bundle(store_ids=(1,), model=model, avg_weekly_demand=1000.0, std_weekly_demand=100.0)
        with make_client(monkeypatch, bundle) as client:
            client.post("/events/sales", json={
                "store_id": 1, "week_ending_date": "2020-02-07", "sales": 1000.0,
            })
            resp = client.get("/inventory/1")
        assert resp.status_code == 200
        body = resp.json()
        assert body["demand_used"] == pytest.approx(1200.0)
        assert body["replenishment_alert"] is True  # 1200 > 1000 * 1.10


# ===========================================================================
# HTTP layer -- GET /health and GET /metrics
# ===========================================================================

class TestHealthAndMetrics:

    def test_health_reports_model_loaded_and_store_counts(self, monkeypatch):
        bundle = make_bundle(store_ids=(1, 2, 3))
        with make_client(monkeypatch, bundle) as client:
            client.post("/events/sales", json={
                "store_id": 1, "week_ending_date": "2020-02-07", "sales": 1000.0,
            })
            resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["model_loaded"] is True
        assert body["total_stores"] == 3
        assert body["stores_with_data"] == 1
        assert body["last_event_received_at"] is not None

    def test_health_with_no_events_yet(self, monkeypatch):
        bundle = make_bundle(store_ids=(1, 2))
        with make_client(monkeypatch, bundle) as client:
            resp = client.get("/health")
        body = resp.json()
        assert body["stores_with_data"] == 0
        assert body["last_event_received_at"] is None

    def test_metrics_count_events_and_forecasts(self, monkeypatch):
        bundle = make_bundle(store_ids=(1,))
        with make_client(monkeypatch, bundle) as client:
            client.post("/events/sales", json={
                "store_id": 1, "week_ending_date": "2020-02-07", "sales": 1000.0,
            })
            client.get("/forecast/1")
            client.get("/forecast/1")
            resp = client.get("/metrics")
        body = resp.json()
        assert body["total_events_ingested"] == 1
        assert body["total_forecasts_served"] == 2
        assert body["average_inference_latency_ms"] >= 0

    def test_metrics_start_at_zero(self, monkeypatch):
        bundle = make_bundle(store_ids=(1,))
        with make_client(monkeypatch, bundle) as client:
            resp = client.get("/metrics")
        body = resp.json()
        assert body["total_events_ingested"] == 0
        assert body["total_forecasts_served"] == 0
        assert body["average_inference_latency_ms"] == 0
