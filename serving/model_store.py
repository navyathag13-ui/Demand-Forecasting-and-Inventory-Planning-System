"""
serving/model_store.py — Loads the persisted model bundle once at process
startup and holds it in memory. No FastAPI imports here on purpose: this
module only knows about the joblib bundle produced by
scripts/train_and_persist.py, so it can be unit-tested without spinning up
the API.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StoreInfo:
    store_id: int
    type: str
    size: float


@dataclass(frozen=True)
class DemandStats:
    avg_weekly_demand: float
    std_weekly_demand: float


class ModelBundleNotLoadedError(RuntimeError):
    """Raised when serving code is used before load_bundle() has run."""


class UnknownStoreError(KeyError):
    """Raised when a store id isn't in the trained catalog of stores."""

    def __init__(self, store_id: int):
        self.store_id = store_id
        super().__init__(f"Store {store_id} is not a known store.")


@dataclass(frozen=True)
class ModelBundle:
    model: Any                     # fitted LGBMForecaster
    feature_columns: list[str]
    size_mean: float
    size_std: float
    min_date: pd.Timestamp
    stores: dict[int, StoreInfo]
    demand_stats: dict[int, DemandStats]
    trained_at: str
    holdout_mape: float
    holdout_r2: float
    holdout_weeks: int

    def store_info(self, store_id: int) -> StoreInfo:
        try:
            return self.stores[store_id]
        except KeyError:
            raise UnknownStoreError(store_id) from None

    def demand_stats_for(self, store_id: int) -> DemandStats:
        try:
            return self.demand_stats[store_id]
        except KeyError:
            raise UnknownStoreError(store_id) from None


def load_bundle(path: Path) -> ModelBundle:
    """
    Load the joblib bundle written by scripts/train_and_persist.py.

    Raises FileNotFoundError with a clear message if persistence hasn't
    been run yet -- callers should fail startup loudly rather than serve
    with no model.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"No persisted model found at {path}. "
            f"Run `python scripts/train_and_persist.py` first."
        )
    raw = joblib.load(path)
    stores = {
        store_id: StoreInfo(store_id=store_id, type=info["type"], size=info["size"])
        for store_id, info in raw["stores"].items()
    }
    demand_stats = {
        store_id: DemandStats(
            avg_weekly_demand=info["avg_weekly_demand"],
            std_weekly_demand=info["std_weekly_demand"],
        )
        for store_id, info in raw["demand_stats"].items()
    }
    bundle = ModelBundle(
        model=raw["model"],
        feature_columns=raw["feature_columns"],
        size_mean=raw["size_mean"],
        size_std=raw["size_std"],
        min_date=pd.Timestamp(raw["min_date"]),
        stores=stores,
        demand_stats=demand_stats,
        trained_at=raw["trained_at"],
        holdout_mape=raw["holdout_mape"],
        holdout_r2=raw["holdout_r2"],
        holdout_weeks=raw["holdout_weeks"],
    )
    logger.info(
        "Loaded model bundle trained_at=%s holdout_mape=%.2f%% stores=%d",
        bundle.trained_at, bundle.holdout_mape, len(bundle.stores),
    )
    return bundle
