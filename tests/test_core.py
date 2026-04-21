"""
tests/test_core.py — Unit tests for metric calculations, inventory formulas,
and preprocessing sanity checks.

Run with:  python -m pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation import mae, rmse, mape, r2, evaluate, compare_models
from src.inventory import (
    compute_safety_stock,
    compute_reorder_point,
    compute_eoq,
    compute_demand_stats,
    build_inventory_plan,
)
from src.data_processing import (
    add_time_features,
    add_store_features,
    add_lag_features,
    train_holdout_split,
)


# ===========================================================================
# Metric tests
# ===========================================================================

class TestMetrics:

    def test_mae_perfect(self):
        a = np.array([1.0, 2.0, 3.0])
        assert mae(a, a) == pytest.approx(0.0)

    def test_mae_known(self):
        a = np.array([10.0, 20.0])
        p = np.array([12.0, 18.0])
        assert mae(a, p) == pytest.approx(2.0)

    def test_rmse_perfect(self):
        a = np.array([5.0, 10.0, 15.0])
        assert rmse(a, a) == pytest.approx(0.0)

    def test_rmse_known(self):
        a = np.array([0.0, 4.0])
        p = np.array([3.0, 0.0])
        # errors = [3, 4] → MSE = 12.5 → RMSE = 3.536
        assert rmse(a, p) == pytest.approx(np.sqrt(12.5), rel=1e-4)

    def test_mape_zero_actuals_excluded(self):
        a = np.array([0.0, 100.0])
        p = np.array([10.0, 110.0])
        # Only the second pair counts: |100-110|/100 = 10%
        assert mape(a, p) == pytest.approx(10.0, rel=1e-4)

    def test_r2_perfect(self):
        a = np.arange(10, dtype=float)
        assert r2(a, a) == pytest.approx(1.0)

    def test_r2_mean_prediction(self):
        a = np.array([1.0, 2.0, 3.0])
        p = np.full(3, a.mean())
        assert r2(a, p) == pytest.approx(0.0)

    def test_evaluate_returns_all_keys(self):
        a = np.array([100.0, 200.0, 300.0])
        result = evaluate(a, a + 10, model_name="test_model")
        for key in ("Model", "MAE", "RMSE", "MAPE", "R2"):
            assert key in result

    def test_compare_models_sorted_by_rmse(self):
        r1 = {"Model": "A", "MAE": 10, "RMSE": 20, "MAPE": 5.0, "R2": 0.9}
        r2_ = {"Model": "B", "MAE": 5,  "RMSE": 10, "MAPE": 2.5, "R2": 0.95}
        df = compare_models([r1, r2_])
        assert df.index[0] == "B"   # lower RMSE first


# ===========================================================================
# Inventory formula tests
# ===========================================================================

class TestInventory:

    def test_safety_stock_zero_std(self):
        ss = compute_safety_stock(std_demand=0, lead_time_weeks=2, z_score=1.645)
        assert ss == pytest.approx(0.0)

    def test_safety_stock_known(self):
        # Z=1.645, σ=100, L=4 → SS = 1.645 * 100 * 2 = 329
        ss = compute_safety_stock(std_demand=100, lead_time_weeks=4, z_score=1.645)
        assert ss == pytest.approx(329.0, rel=1e-3)

    def test_reorder_point_equals_demand_plus_ss(self):
        avg, std, L, z = 200, 50, 2, 1.645
        ss   = compute_safety_stock(std, L, z)
        rop  = compute_reorder_point(avg, std, L, z)
        assert rop == pytest.approx(avg * L + ss, rel=1e-6)

    def test_eoq_positive(self):
        q = compute_eoq(avg_weekly_demand=1000, ordering_cost=150,
                        unit_value=25, holding_cost_pct=0.2)
        assert q > 0

    def test_eoq_zero_demand(self):
        q = compute_eoq(avg_weekly_demand=0, ordering_cost=150,
                        unit_value=25, holding_cost_pct=0.2)
        assert q == pytest.approx(0.0)

    def test_compute_demand_stats_shape(self):
        df = pd.DataFrame({
            "Store": [1, 1, 2, 2],
            "Weekly_Sales": [100.0, 200.0, 300.0, 400.0],
        })
        stats = compute_demand_stats(df)
        assert len(stats) == 2
        assert "avg_weekly_demand" in stats.columns
        assert "std_weekly_demand" in stats.columns

    def test_build_inventory_plan_columns(self):
        stats = pd.DataFrame({
            "Store": [1],
            "avg_weekly_demand": [1000.0],
            "std_weekly_demand": [200.0],
            "n_weeks": [52],
            "cv": [0.2],
        })
        plan = build_inventory_plan(
            demand_stats     = stats,
            forecast_df      = None,
            lead_time_weeks  = 2,
            z_score          = 1.645,
            ordering_cost    = 150,
            unit_value       = 25,
            holding_cost_pct = 0.2,
        )
        for col in ("safety_stock", "reorder_point", "eoq", "recommended_order_qty",
                    "replenishment_alert"):
            assert col in plan.columns


# ===========================================================================
# Data processing tests
# ===========================================================================

class TestDataProcessing:

    def _make_df(self) -> pd.DataFrame:
        dates = pd.date_range("2020-01-01", periods=20, freq="W")
        return pd.DataFrame({
            "Store": [1] * 20,
            "Date":  dates,
            "Type":  ["A"] * 20,
            "Size":  [100_000] * 20,
            "IsHoliday": [False] * 20,
            "Weekly_Sales": np.linspace(1000, 2000, 20),
        })

    def test_add_time_features_columns(self):
        df = self._make_df()
        out = add_time_features(df)
        for col in ("year", "month", "week_of_year", "trend",
                    "week_sin", "week_cos"):
            assert col in out.columns

    def test_add_store_features_type_code(self):
        df = self._make_df()
        out = add_store_features(df)
        assert "type_code" in out.columns
        assert out["type_code"].iloc[0] == 0  # "A" → 0

    def test_lag_features_no_future_leakage(self):
        df = self._make_df()
        out = add_lag_features(df, lags=[1])
        # lag_1 at row i should equal Weekly_Sales at row i-1
        assert out["sales_lag_1"].iloc[1] == pytest.approx(df["Weekly_Sales"].iloc[0])

    def test_train_holdout_split_sizes(self):
        df = self._make_df()
        # Minimal features to avoid dropna() eliminating all rows
        df["sales_lag_1"] = df["Weekly_Sales"].shift(1)
        train, holdout = train_holdout_split(df, holdout_weeks=4)
        assert len(holdout) > 0
        assert len(train) > 0
        assert train["Date"].max() < holdout["Date"].min()

    def test_holdout_no_future_dates_in_train(self):
        df = self._make_df()
        df["sales_lag_1"] = df["Weekly_Sales"].shift(1)
        train, holdout = train_holdout_split(df, holdout_weeks=4)
        assert train["Date"].max() < holdout["Date"].min()

    def test_forecast_output_shape(self):
        from src.forecasting import MovingAverageForecaster
        df = self._make_df()
        df = add_lag_features(df, lags=[1, 2, 4, 12])
        train, holdout = train_holdout_split(df, holdout_weeks=4)
        m = MovingAverageForecaster(window=4)
        m.fit(train, train["Weekly_Sales"])
        preds = m.predict(holdout)
        assert preds.shape[0] == len(holdout)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
