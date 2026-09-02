"""
serving/feature_state.py — Per-store rolling sales history and the feature
recomputation that mirrors src/data_processing.py's lag/rolling/calendar
feature engineering, but incrementally, one event at a time.

No FastAPI imports here -- this is pure, framework-agnostic logic so it can
be unit-tested directly, the same separation src/ already keeps between
model code and main.py's orchestration.

Feature vector semantics
-------------------------
When a sales record for week W arrives, the cached feature vector is built
to predict week W+1 ("next week"), not to describe week W itself:
  - sales_lag_1  = actual sales at W       (the week we just received)
  - sales_lag_2  = actual sales at W-1
  - sales_lag_4  = actual sales at W-3
  - sales_lag_12 = actual sales at W-11
  - sales_roll_4 / sales_roll_12 = trailing means ending at W
  - calendar features (trend, week/month sin-cos, quarter, is_month_end)
    are computed for W+1's date, since that's the week being forecasted
  - IsHoliday for W+1 is NOT knowable from the ingestion feed (a sales
    event only reports whether the week that just happened was a holiday,
    not whether the upcoming one will be) -- it defaults to False. A real
    deployment would source this from a known holiday calendar rather than
    the sales feed; documented here rather than silently guessed.

This exactly mirrors add_lag_features()'s shift(1)/shift(2)/... semantics:
a row's lag_1 there is the PRIOR row's value, so the row being built to
predict W+1 has lag_1 = W's own value -- the same relationship, just
computed incrementally instead of with a groupby().shift() over a full
DataFrame.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from src.data_processing import STORE_TYPE_MAP
from .model_store import ModelBundle

LAGS = (1, 2, 4, 12)
ROLLING_WINDOWS = (4, 12)


@dataclass(frozen=True)
class SalesEvent:
    store_id: int
    week_ending_date: pd.Timestamp
    sales: float
    is_holiday: bool
    received_at: datetime


@dataclass(frozen=True)
class FeatureSnapshot:
    based_on_week: pd.Timestamp     # most recent actual week ingested
    target_week: pd.Timestamp       # the week this feature vector forecasts
    features: dict[str, float]      # LGBM_FEATURES -> value (NaN where history is short)
    computed_at: datetime


class DuplicateOrOutOfOrderEventError(ValueError):
    """A new event's week is not strictly after the store's most recent one."""

    def __init__(self, store_id: int, attempted_week: pd.Timestamp, most_recent_week: pd.Timestamp):
        self.store_id = store_id
        self.attempted_week = attempted_week
        self.most_recent_week = most_recent_week
        super().__init__(
            f"Store {store_id}: week {attempted_week.date()} is not after "
            f"the most recently recorded week {most_recent_week.date()}."
        )


def _lag_value(events: list[SalesEvent], n: int) -> float:
    idx = len(events) - n
    return events[idx].sales if idx >= 0 else math.nan


def _rolling_mean(events: list[SalesEvent], window: int) -> float:
    tail = events[-window:]
    return float(sum(e.sales for e in tail) / len(tail)) if tail else math.nan


def build_feature_snapshot(
    store_id: int,
    events: list[SalesEvent],
    bundle: ModelBundle,
    *,
    now: Optional[datetime] = None,
) -> FeatureSnapshot:
    """
    Build the feature vector for forecasting the week AFTER the most recent
    event in `events`. Pure function of (history, model bundle) -- no
    hidden state -- so it's directly unit-testable without a running store.
    """
    if not events:
        raise ValueError("Cannot build a feature snapshot with no history.")

    now = now or datetime.now(timezone.utc)
    latest = events[-1]
    target_week = latest.week_ending_date + pd.Timedelta(days=7)

    store_info = bundle.store_info(store_id)
    type_code = STORE_TYPE_MAP[store_info.type]
    size_norm = (store_info.size - bundle.size_mean) / bundle.size_std

    week_of_year = target_week.isocalendar()[1]
    trend = int((target_week - bundle.min_date).days / 7)

    features = {
        "trend": float(trend),
        "week_sin": math.sin(2 * math.pi * week_of_year / 52),
        "week_cos": math.cos(2 * math.pi * week_of_year / 52),
        "month_sin": math.sin(2 * math.pi * target_week.month / 12),
        "month_cos": math.cos(2 * math.pi * target_week.month / 12),
        "quarter": float(target_week.quarter),
        "is_month_end": float(bool(target_week.is_month_end)),
        "IsHoliday": 0.0,
        "type_code": float(type_code),
        "size_norm": float(size_norm),
    }
    for n in LAGS:
        features[f"sales_lag_{n}"] = _lag_value(events, n)
    for window in ROLLING_WINDOWS:
        features[f"sales_roll_{window}"] = _rolling_mean(events, window)

    return FeatureSnapshot(
        based_on_week=latest.week_ending_date,
        target_week=target_week,
        features=features,
        computed_at=now,
    )


class StoreHistory:
    """
    One store's rolling sales history plus its latest feature snapshot.

    A single lock guards the full check-then-append-then-recompute
    sequence for this store: two concurrent events for the same store
    arriving close together must not both pass the "is this newer than
    the latest" check before either commits (a classic check-then-act
    race), and a reader must never observe a feature snapshot built from
    a partially-appended history. The snapshot is swapped in with one
    reference assignment as the last step, so a concurrent read (which
    takes no lock -- see `latest_snapshot`) always sees either the old
    snapshot or the fully-built new one, never something in between.
    """

    def __init__(self, store_id: int):
        self.store_id = store_id
        self._lock = threading.Lock()
        self._events: list[SalesEvent] = []
        self._latest_snapshot: Optional[FeatureSnapshot] = None

    @property
    def latest_snapshot(self) -> Optional[FeatureSnapshot]:
        return self._latest_snapshot

    @property
    def event_count(self) -> int:
        return len(self._events)

    @property
    def last_received_at(self) -> Optional[datetime]:
        return self._events[-1].received_at if self._events else None

    def record_event(
        self,
        week_ending_date,
        sales: float,
        is_holiday: bool,
        bundle: ModelBundle,
        *,
        now: Optional[datetime] = None,
    ) -> tuple[SalesEvent, FeatureSnapshot]:
        now = now or datetime.now(timezone.utc)
        week_ending_date = pd.Timestamp(week_ending_date)

        with self._lock:
            if self._events and week_ending_date <= self._events[-1].week_ending_date:
                raise DuplicateOrOutOfOrderEventError(
                    self.store_id, week_ending_date, self._events[-1].week_ending_date
                )
            event = SalesEvent(
                store_id=self.store_id,
                week_ending_date=week_ending_date,
                sales=float(sales),
                is_holiday=bool(is_holiday),
                received_at=now,
            )
            new_events = self._events + [event]  # build fresh; never mutate a list a reader might be iterating
            snapshot = build_feature_snapshot(self.store_id, new_events, bundle, now=now)
            self._events = new_events
            self._latest_snapshot = snapshot   # single reference swap -- atomic under the GIL
            return event, snapshot


class FeatureStateStore:
    """
    Holds one StoreHistory per known store. All store ids are pre-created
    from the model bundle's store catalog at construction time, so ingesting
    an event never inserts a new key into a shared dict at request time --
    that would be its own race (two concurrent first-events for a
    never-before-seen store both trying to create the entry). There is no
    such thing as a "never-before-seen" store here: the catalog is fixed by
    what the model was trained on.
    """

    def __init__(self, bundle: ModelBundle):
        self._bundle = bundle
        self._histories: dict[int, StoreHistory] = {
            store_id: StoreHistory(store_id) for store_id in bundle.stores
        }

    def record_event(self, store_id: int, week_ending_date, sales: float, is_holiday: bool):
        self._bundle.store_info(store_id)  # raises UnknownStoreError if store_id isn't in the catalog
        history = self._histories[store_id]
        return history.record_event(week_ending_date, sales, is_holiday, self._bundle)

    def get_history(self, store_id: int) -> StoreHistory:
        self._bundle.store_info(store_id)
        return self._histories[store_id]

    def latest_snapshot(self, store_id: int) -> Optional[FeatureSnapshot]:
        return self.get_history(store_id).latest_snapshot

    def all_histories(self) -> dict[int, StoreHistory]:
        return dict(self._histories)

    def stores_with_data(self) -> int:
        return sum(1 for h in self._histories.values() if h.event_count > 0)

    def last_event_received_at(self) -> Optional[datetime]:
        """Most recent event across every store, or None if nothing has
        been ingested yet. Used by GET /health's overall freshness check."""
        timestamps = [h.last_received_at for h in self._histories.values() if h.last_received_at]
        return max(timestamps) if timestamps else None
