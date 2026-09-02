"""
serving/inference.py — Runs the persisted LightGBM model against one cached
feature snapshot. No FastAPI imports; pure function of (bundle, snapshot),
so it's directly unit-testable without the API running.
"""

from __future__ import annotations

import pandas as pd

from .feature_state import FeatureSnapshot
from .model_store import ModelBundle


class InferenceError(RuntimeError):
    """The persisted model failed to produce a prediction for a snapshot
    that otherwise passed all upstream validation. Distinct from the
    request-shape/state errors (UnknownStoreError, no-history-yet) the
    endpoints already handle, so callers can map it to a clean 503
    instead of letting whatever the model raised become a raw 500."""


def predict_next_week(bundle: ModelBundle, snapshot: FeatureSnapshot) -> float:
    """
    Run inference for one store's current feature snapshot.

    Builds a single-row DataFrame from the snapshot's feature dict --
    LGBMForecaster.predict() selects and orders the columns it needs
    internally (see src/forecasting.py's _select()), so a NaN lag/rolling
    feature from short history is handled the same fillna(0) way a
    short-history training row was.
    """
    X = pd.DataFrame([snapshot.features])
    try:
        prediction = bundle.model.predict(X)
        return float(prediction[0])
    except InferenceError:
        raise
    except Exception as exc:
        raise InferenceError(f"Model failed to produce a prediction: {exc}") from exc
