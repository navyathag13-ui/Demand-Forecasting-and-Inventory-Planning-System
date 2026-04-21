"""
main.py — End-to-end pipeline for the Demand Forecasting and Inventory Planning System.

Usage
-----
    python main.py

All outputs are written to outputs/ (charts, CSVs, metrics tables, reports).
"""

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Local imports
import config
from src.utils import setup_logging, ensure_dirs, save_csv, print_section
from src.data_processing import (
    load_raw_data,
    build_store_weekly,
    prepare_features,
    train_holdout_split,
)
from src.forecasting import (
    NaiveForecaster,
    MovingAverageForecaster,
    LinearForecaster,
    LGBMForecaster,
)
from src.evaluation import evaluate, compare_models
from src.inventory import (
    compute_demand_stats,
    build_inventory_plan,
    summarise_inventory,
)
from src.visualization import (
    plot_demand_trend,
    plot_actual_vs_predicted,
    plot_model_comparison,
    plot_feature_importance,
    plot_inventory_plan,
    plot_demand_by_store_type,
    plot_residuals,
)


def main() -> None:
    setup_logging()
    log = logging.getLogger(__name__)
    t0  = time.time()

    # Ensure output directories exist
    ensure_dirs(
        config.CHARTS_DIR,
        config.FORECASTS_DIR,
        config.METRICS_DIR,
        config.REPORTS_DIR,
    )

    # ==========================================================================
    # STEP 1 — Load and prepare data
    # ==========================================================================
    print_section("STEP 1 | DATA LOADING & PREPARATION")

    train_raw, stores = load_raw_data(config.DATA_DIR)
    df_weekly = build_store_weekly(train_raw, stores)

    print(f"  Dataset period : {df_weekly['Date'].min().date()} → "
          f"{df_weekly['Date'].max().date()}")
    print(f"  Stores         : {df_weekly['Store'].nunique()}")
    print(f"  Weeks          : {df_weekly['Date'].nunique()}")
    print(f"  Total rows     : {len(df_weekly):,}")

    # Feature engineering
    df_features = prepare_features(df_weekly)

    # Train / holdout split
    train_df, holdout_df = train_holdout_split(
        df_features, holdout_weeks=config.HOLDOUT_WEEKS
    )

    # ==========================================================================
    # STEP 2 — Train models
    # ==========================================================================
    print_section("STEP 2 | FORECASTING MODELS")

    TARGET = "Weekly_Sales"

    models = [
        NaiveForecaster(),
        MovingAverageForecaster(window=4),
        LinearForecaster(alpha=1.0),
        LGBMForecaster(),
    ]

    # Fit on training set
    for m in models:
        log.info("Fitting %s ...", m.name)
        m.fit(train_df, train_df[TARGET])

    # Predict on holdout
    predictions: dict[str, np.ndarray] = {}
    for m in models:
        predictions[m.name] = m.predict(holdout_df)

    # ==========================================================================
    # STEP 3 — Evaluation
    # ==========================================================================
    print_section("STEP 3 | MODEL EVALUATION")

    actual = holdout_df[TARGET].values
    results = [evaluate(actual, predictions[m.name], m.name) for m in models]
    metrics_table = compare_models(results)

    print("\n  Model Performance on Holdout Set (last 12 weeks):\n")
    print(metrics_table.to_string())
    save_csv(metrics_table, config.METRICS_DIR / "model_comparison.csv")

    # Best model
    best_name = metrics_table.index[0]
    best_preds = predictions[best_name]
    improvement = metrics_table.loc[best_name, "RMSE_vs_Naive (%)"]
    print(f"\n  Best model : {best_name}")
    print(f"  RMSE improvement vs Naive : {improvement:.1f}%")

    # Save all forecasts
    forecast_df = holdout_df[["Store", "Date", TARGET]].copy()
    forecast_df["actual"] = actual
    for m in models:
        forecast_df[f"pred_{m.name.replace(' ', '_').replace('(','').replace(')','').replace('/','_')}"] = predictions[m.name]
    forecast_df["predicted"] = best_preds  # best model column
    save_csv(forecast_df, config.FORECASTS_DIR / "holdout_forecasts.csv", index=False)

    # ==========================================================================
    # STEP 4 — Inventory planning
    # ==========================================================================
    print_section("STEP 4 | INVENTORY PLANNING")

    demand_stats = compute_demand_stats(train_df, group_col="Store")

    # Aggregate best-model forecasts by store for forward-looking demand
    store_forecasts = (
        forecast_df.groupby("Store")[["predicted"]]
        .mean()
        .reset_index()
    )

    inventory_plan = build_inventory_plan(
        demand_stats    = demand_stats,
        forecast_df     = store_forecasts,
        lead_time_weeks = config.LEAD_TIME_WEEKS,
        z_score         = config.Z_SCORE,
        ordering_cost   = config.ORDERING_COST_USD,
        unit_value      = config.UNIT_VALUE_USD,
        holding_cost_pct= config.UNIT_HOLDING_COST_PCT,
    )

    print(summarise_inventory(inventory_plan))

    # Alert details
    alerts = inventory_plan[inventory_plan["replenishment_alert"]]
    if len(alerts):
        print(f"\n  Stores requiring replenishment action:\n")
        print(alerts[["Store", "demand_used", "safety_stock",
                       "reorder_point", "recommended_order_qty"]]
              .to_string(index=False))

    save_csv(inventory_plan, config.REPORTS_DIR / "inventory_plan.csv", index=False)

    # ==========================================================================
    # STEP 5 — Charts
    # ==========================================================================
    print_section("STEP 5 | VISUALISATIONS")

    # 1. Demand trend
    plot_demand_trend(
        df_weekly,
        output_path=config.CHARTS_DIR / "01_demand_trend.png",
    )

    # 2. Actual vs Predicted (aggregate across stores per date)
    holdout_by_date = (
        holdout_df.groupby("Date")[TARGET].sum().reset_index()
    )
    pred_by_date: dict[str, np.ndarray] = {}
    for m in models:
        tmp = holdout_df.copy()
        tmp["pred"] = predictions[m.name]
        pred_by_date[m.name] = tmp.groupby("Date")["pred"].sum().values

    plot_actual_vs_predicted(
        results     = pred_by_date,
        actual      = holdout_by_date[TARGET].values,
        dates       = holdout_by_date["Date"].values,
        output_path = config.CHARTS_DIR / "02_actual_vs_predicted.png",
    )

    # 3. Model comparison
    plot_model_comparison(
        metrics_table,
        output_path=config.CHARTS_DIR / "03_model_comparison.png",
    )

    # 4. Feature importance (LightGBM)
    lgbm_model = next(m for m in models if isinstance(m, LGBMForecaster))
    plot_feature_importance(
        lgbm_model.feature_importances,
        output_path=config.CHARTS_DIR / "04_feature_importance.png",
    )

    # 5. Inventory planning scatter
    plot_inventory_plan(
        inventory_plan,
        output_path=config.CHARTS_DIR / "05_inventory_plan.png",
    )

    # 6. Demand by store type
    plot_demand_by_store_type(
        df_weekly,
        output_path=config.CHARTS_DIR / "06_demand_by_store_type.png",
    )

    # 7. Residuals for best model
    plot_residuals(
        actual      = actual,
        predicted   = best_preds,
        model_name  = best_name,
        output_path = config.CHARTS_DIR / "07_residuals.png",
    )

    # ==========================================================================
    # STEP 6 — Summary report
    # ==========================================================================
    print_section("STEP 6 | REPORT")

    report_lines = [
        "DEMAND FORECASTING & INVENTORY PLANNING — EXECUTIVE SUMMARY",
        "=" * 60,
        "",
        "DATASET",
        f"  Source        : Walmart Weekly Sales (train.csv, stores.csv)",
        f"  Period        : {df_weekly['Date'].min().date()} – {df_weekly['Date'].max().date()}",
        f"  Stores        : {df_weekly['Store'].nunique()} (Types A, B, C)",
        f"  Holdout weeks : {config.HOLDOUT_WEEKS}",
        "",
        "FORECASTING RESULTS (holdout set)",
    ]
    for _, row in metrics_table.reset_index().iterrows():
        report_lines.append(
            f"  {row['Model']:<30} MAE={row['MAE']:>10,.0f}  "
            f"RMSE={row['RMSE']:>10,.0f}  MAPE={row['MAPE']:>5.1f}%"
        )

    report_lines += [
        "",
        f"  Best model    : {best_name}",
        f"  RMSE vs naive : -{improvement:.1f}%",
        "",
        "INVENTORY PLANNING (2-week lead time, 95% service level)",
        f"  Avg safety stock     : ${inventory_plan['safety_stock'].mean():>10,.0f}",
        f"  Avg reorder point    : ${inventory_plan['reorder_point'].mean():>10,.0f}",
        f"  Avg EOQ              : ${inventory_plan['eoq'].mean():>10,.0f}",
        f"  Replenishment alerts : {inventory_plan['replenishment_alert'].sum()} stores",
        "",
        "OUTPUTS",
        f"  Charts    → outputs/charts/",
        f"  Forecasts → outputs/forecasts/",
        f"  Metrics   → outputs/metrics/",
        f"  Reports   → outputs/reports/",
        "",
        f"Pipeline completed in {time.time() - t0:.1f}s.",
    ]

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    report_path = config.REPORTS_DIR / "executive_summary.txt"
    report_path.write_text(report_text)
    log.info("Report saved → %s", report_path)


if __name__ == "__main__":
    main()
