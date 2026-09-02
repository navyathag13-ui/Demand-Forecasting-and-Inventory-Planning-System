"""
scripts/train_and_persist.py — Fit the LightGBM forecaster exactly as
main.py does, then persist it (plus the fitted feature-engineering
constants) to disk via joblib for the serving layer to load.

Usage
-----
    python scripts/train_and_persist.py

This does NOT retrain on a different data slice or with different
hyperparameters than main.py -- same train/holdout split, same
LGBM_PARAMS, same RANDOM_SEED. The point is to persist the exact model
whose 4.1% MAPE is already documented, not to produce a new one.
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.data_processing import (
    load_raw_data,
    build_store_weekly,
    prepare_features,
    train_holdout_split,
    size_normalization_params,
    trend_reference_date,
)
from src.forecasting import LGBMForecaster, LGBM_FEATURES
from src.evaluation import evaluate
from src.inventory import compute_demand_stats
from src.utils import setup_logging, ensure_dirs

TARGET = "Weekly_Sales"


def main() -> None:
    setup_logging()
    log = logging.getLogger(__name__)

    train_raw, stores = load_raw_data(config.DATA_DIR)
    df_weekly = build_store_weekly(train_raw, stores)
    df_features = prepare_features(df_weekly)
    train_df, holdout_df = train_holdout_split(df_features, holdout_weeks=config.HOLDOUT_WEEKS)

    model = LGBMForecaster()
    log.info("Fitting LGBMForecaster on %d training rows ...", len(train_df))
    model.fit(train_df, train_df[TARGET])

    holdout_preds = model.predict(holdout_df)
    holdout_metrics = evaluate(holdout_df[TARGET].values, holdout_preds, model_name="LightGBM")
    log.info(
        "Holdout check: MAPE=%.2f%% (expected ~4.1%%), R2=%.4f",
        holdout_metrics["MAPE"], holdout_metrics["R2"],
    )

    # Feature-engineering constants fitted on the same df_weekly the model
    # trained against, so serving-time feature recomputation normalises
    # size and anchors trend identically to training -- not a second,
    # independently-derived copy of the same statistics.
    size_mean, size_std = size_normalization_params(df_weekly)
    min_date = trend_reference_date(df_weekly)

    stores_lookup = {
        int(row.Store): {"type": row.Type, "size": float(row.Size)}
        for row in stores.itertuples()
    }

    # Historical demand stats (avg/std weekly sales), same as main.py's
    # inventory step -- computed on train_df only, so serving-time
    # inventory calculations use the same "known-good" baseline the batch
    # pipeline's inventory plan does, not stats re-derived from whatever
    # happens to be in the live rolling history at the moment.
    demand_stats = compute_demand_stats(train_df, group_col="Store")
    demand_stats_lookup = {
        int(row.Store): {
            "avg_weekly_demand": float(row.avg_weekly_demand),
            "std_weekly_demand": float(row.std_weekly_demand),
        }
        for row in demand_stats.itertuples()
    }

    bundle = {
        "model": model,
        "feature_columns": LGBM_FEATURES,
        "size_mean": size_mean,
        "size_std": size_std,
        "min_date": min_date,
        "stores": stores_lookup,
        "demand_stats": demand_stats_lookup,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "holdout_mape": holdout_metrics["MAPE"],
        "holdout_r2": holdout_metrics["R2"],
        "holdout_weeks": config.HOLDOUT_WEEKS,
    }

    ensure_dirs(config.MODEL_DIR)
    # Write to a temp file in the same directory, then atomically rename --
    # a kill or a full disk mid-write must never leave a corrupt .joblib at
    # the path load_bundle() checks for and trusts.
    tmp_path = config.MODEL_PATH.with_suffix(".joblib.tmp")
    joblib.dump(bundle, tmp_path)
    tmp_path.replace(config.MODEL_PATH)
    log.info("Persisted model bundle -> %s", config.MODEL_PATH)


if __name__ == "__main__":
    main()
