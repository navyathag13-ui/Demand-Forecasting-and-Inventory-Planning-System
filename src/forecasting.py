"""
forecasting.py — Forecasting models for weekly retail demand.

Models implemented (increasing complexity):
  1. NaiveForecaster     – last observed value (benchmark floor)
  2. MovingAverageForecaster – rolling n-week mean
  3. LinearForecaster    – OLS regression on time + store features
  4. LGBMForecaster      – LightGBM with lag / rolling / calendar features

All models follow the same fit(X_train, y_train) / predict(X) interface so
they can be compared interchangeably in main.py.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

logger = logging.getLogger(__name__)

# Feature columns used by the ML models
TIME_FEATURES = [
    "trend", "week_sin", "week_cos", "month_sin", "month_cos",
    "quarter", "is_month_end", "IsHoliday",
]
STORE_FEATURES = ["type_code", "size_norm"]
LAG_FEATURES   = [
    "sales_lag_1", "sales_lag_2", "sales_lag_4", "sales_lag_12",
    "sales_roll_4", "sales_roll_12",
]
LINEAR_FEATURES = TIME_FEATURES + STORE_FEATURES
LGBM_FEATURES   = TIME_FEATURES + STORE_FEATURES + LAG_FEATURES


# ---------------------------------------------------------------------------
# Naive baseline
# ---------------------------------------------------------------------------

class NaiveForecaster:
    """
    Predicts next period's demand as the most recent observed value.

    This is the simplest possible benchmark; any useful model must beat it.
    """

    name = "Naive (Last Value)"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NaiveForecaster":
        # Store the last known sales value per store for prediction
        self._last = (
            X.assign(y=y.values)
            .groupby("Store")["y"]
            .last()
            .to_dict()
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["Store"].map(self._last).fillna(0).values


# ---------------------------------------------------------------------------
# Moving-average baseline
# ---------------------------------------------------------------------------

class MovingAverageForecaster:
    """
    Predicts demand as the rolling mean of the last `window` weeks.

    Uses the pre-computed `sales_roll_{window}` feature so it is
    consistent with the leakage-free feature engineering pipeline.
    """

    name: str

    def __init__(self, window: int = 4):
        self.window = window
        self.name   = f"Moving Average ({window}w)"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MovingAverageForecaster":
        return self  # stateless — relies on pre-computed rolling feature

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        col = f"sales_roll_{self.window}"
        if col not in X.columns:
            raise ValueError(f"Feature '{col}' missing — run add_lag_features() first.")
        return X[col].fillna(0).values


# ---------------------------------------------------------------------------
# Ridge Regression
# ---------------------------------------------------------------------------

class LinearForecaster:
    """
    Ridge regression on calendar and store features.

    Captures trend and seasonality linearly. Good interpretability baseline.
    """

    name = "Ridge Regression"

    def __init__(self, alpha: float = 1.0):
        self.alpha   = alpha
        self._scaler = StandardScaler()
        self._model  = Ridge(alpha=alpha)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LinearForecaster":
        X_train = self._select(X)
        X_scaled = self._scaler.fit_transform(X_train)
        self._model.fit(X_scaled, y)
        logger.info("LinearForecaster fitted on %d rows, %d features.",
                    len(y), X_train.shape[1])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_feat   = self._select(X)
        X_scaled = self._scaler.transform(X_feat)
        return np.maximum(self._model.predict(X_scaled), 0)  # clip negatives

    def _select(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[LINEAR_FEATURES].fillna(0)


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------

class LGBMForecaster:
    """
    Gradient-boosted trees with lag, rolling, calendar, and store features.

    Best performer — captures non-linear interactions and holiday spikes.
    Uses early stopping on a 10% chronological validation split.
    """

    name = "LightGBM"

    def __init__(self, params: Optional[dict] = None):
        from config import LGBM_PARAMS
        self._params = params or LGBM_PARAMS
        self._model: Optional[lgb.LGBMRegressor] = None
        self._feature_importances_: Optional[pd.Series] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LGBMForecaster":
        X_feat = self._select(X)

        # Chronological 10% validation split for early stopping
        n_val        = max(1, int(len(y) * 0.1))
        X_tr, y_tr   = X_feat.iloc[:-n_val], y.iloc[:-n_val]
        X_val, y_val = X_feat.iloc[-n_val:], y.iloc[-n_val:]

        self._model = lgb.LGBMRegressor(**self._params)
        self._model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )
        self._feature_importances_ = pd.Series(
            self._model.feature_importances_,
            index=X_feat.columns,
        ).sort_values(ascending=False)
        logger.info(
            "LGBMForecaster fitted | best_iteration=%d",
            self._model.best_iteration_,
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_feat = self._select(X)
        return np.maximum(self._model.predict(X_feat), 0)  # clip negatives

    def _select(self, X: pd.DataFrame) -> pd.DataFrame:
        available = [f for f in LGBM_FEATURES if f in X.columns]
        return X[available].fillna(0)

    @property
    def feature_importances(self) -> pd.Series:
        if self._feature_importances_ is None:
            raise RuntimeError("Call fit() first.")
        return self._feature_importances_
