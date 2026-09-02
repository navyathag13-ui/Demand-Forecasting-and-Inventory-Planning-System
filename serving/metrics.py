"""
serving/metrics.py — Lightweight in-memory counters for GET /metrics.

Plain JSON, not a full Prometheus setup, per the spec. Keeps a running
sum/count for average latency rather than a growing list of every
inference's timing -- correct and O(1) memory for a service meant to run
indefinitely, not just for a demo session.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class MetricsSnapshot:
    total_events_ingested: int
    total_forecasts_served: int
    average_inference_latency_ms: float


class MetricsTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self._events_ingested = 0
        self._forecasts_served = 0
        self._inference_latency_total_s = 0.0

    def record_event_ingested(self) -> None:
        with self._lock:
            self._events_ingested += 1

    def record_forecast_served(self, latency_s: float) -> None:
        with self._lock:
            self._forecasts_served += 1
            self._inference_latency_total_s += latency_s

    def snapshot(self) -> MetricsSnapshot:
        with self._lock:
            avg_latency_ms = (
                (self._inference_latency_total_s / self._forecasts_served) * 1000
                if self._forecasts_served else 0.0
            )
            return MetricsSnapshot(
                total_events_ingested=self._events_ingested,
                total_forecasts_served=self._forecasts_served,
                average_inference_latency_ms=round(avg_latency_ms, 3),
            )
