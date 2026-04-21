"""
inventory.py — Translate demand forecasts into actionable inventory decisions.

Inventory model used: Continuous-review (s, Q) policy
  - s  = reorder point   (trigger a replenishment when stock falls to s)
  - Q  = order quantity  (economic order quantity)

Formula reference:
  Safety Stock (SS)    = Z × σ_demand × √(lead_time)
  Reorder Point (ROP)  = μ_demand × lead_time + SS
  EOQ                  = √(2 × D × K / h)
    where D = annual demand, K = ordering cost, h = annual holding cost per unit

All quantities are in units of $1 weekly sales (i.e. demand is measured
in dollars, not physical units, since item counts are not in the dataset).
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_demand_stats(
    df: pd.DataFrame,
    group_col: str = "Store",
    sales_col: str = "Weekly_Sales",
) -> pd.DataFrame:
    """
    Compute per-group demand statistics from historical (training) data.

    Parameters
    ----------
    df : pd.DataFrame
        Historical store-week data.
    group_col : str
        Column to group by (e.g. 'Store').
    sales_col : str
        Column containing weekly demand values.

    Returns
    -------
    pd.DataFrame with columns: group_col, avg_weekly_demand, std_weekly_demand,
        cv (coefficient of variation), n_weeks.
    """
    stats = (
        df.groupby(group_col)[sales_col]
        .agg(
            avg_weekly_demand="mean",
            std_weekly_demand="std",
            n_weeks="count",
        )
        .reset_index()
    )
    stats["cv"] = (stats["std_weekly_demand"] / stats["avg_weekly_demand"]).round(3)
    return stats


def compute_safety_stock(
    std_demand: float,
    lead_time_weeks: float,
    z_score: float,
) -> float:
    """
    Safety stock to absorb demand variability during lead time.

    SS = Z × σ_demand × √(lead_time)
    """
    return z_score * std_demand * np.sqrt(lead_time_weeks)


def compute_reorder_point(
    avg_demand: float,
    std_demand: float,
    lead_time_weeks: float,
    z_score: float,
) -> float:
    """
    Reorder point: order when on-hand inventory falls to this level.

    ROP = avg_demand × lead_time + safety_stock
    """
    ss = compute_safety_stock(std_demand, lead_time_weeks, z_score)
    return avg_demand * lead_time_weeks + ss


def compute_eoq(
    avg_weekly_demand: float,
    ordering_cost: float,
    unit_value: float,
    holding_cost_pct: float,
) -> float:
    """
    Economic Order Quantity — minimises total holding + ordering cost.

    EOQ = √(2 × D_annual × K / h)
      D_annual = avg_weekly_demand × 52
      h        = holding_cost_pct × unit_value (per unit per year)
    """
    D = avg_weekly_demand * 52
    h = holding_cost_pct * unit_value
    if h <= 0 or D <= 0:
        return 0.0
    return float(np.sqrt(2 * D * ordering_cost / h))


def build_inventory_plan(
    demand_stats: pd.DataFrame,
    forecast_df: Optional[pd.DataFrame],
    lead_time_weeks: float,
    z_score: float,
    ordering_cost: float,
    unit_value: float,
    holding_cost_pct: float,
    group_col: str = "Store",
) -> pd.DataFrame:
    """
    Full inventory planning table for each store.

    Parameters
    ----------
    demand_stats : pd.DataFrame
        Output of compute_demand_stats().
    forecast_df : pd.DataFrame or None
        Forward-looking forecasts (avg used if provided).
    lead_time_weeks, z_score, ordering_cost, unit_value, holding_cost_pct
        Inventory model parameters (see config.py).

    Returns
    -------
    pd.DataFrame with one row per store containing:
        safety_stock, reorder_point, eoq, recommended_order_qty,
        replenishment_alert (bool)
    """
    plan = demand_stats.copy()

    # If we have model forecasts, use forecast mean as forward-looking demand
    if forecast_df is not None and "predicted" in forecast_df.columns:
        fwd = (
            forecast_df.groupby(group_col)["predicted"]
            .mean()
            .rename("forecast_avg_demand")
        )
        plan = plan.merge(fwd, on=group_col, how="left")
        plan["demand_used"] = plan["forecast_avg_demand"].fillna(plan["avg_weekly_demand"])
    else:
        plan["demand_used"] = plan["avg_weekly_demand"]

    plan["safety_stock"] = plan.apply(
        lambda r: compute_safety_stock(r["std_weekly_demand"], lead_time_weeks, z_score),
        axis=1,
    ).round(2)

    plan["reorder_point"] = plan.apply(
        lambda r: compute_reorder_point(
            r["demand_used"], r["std_weekly_demand"], lead_time_weeks, z_score
        ),
        axis=1,
    ).round(2)

    plan["eoq"] = plan.apply(
        lambda r: compute_eoq(
            r["demand_used"], ordering_cost, unit_value, holding_cost_pct
        ),
        axis=1,
    ).round(2)

    # Recommended order: cover review period + lead time at forecasted demand
    plan["recommended_order_qty"] = (
        plan["demand_used"] * (lead_time_weeks + 4) + plan["safety_stock"]
    ).round(2)

    # Flag stores where average forecast exceeds historical mean by > 10%
    if "forecast_avg_demand" in plan.columns:
        plan["replenishment_alert"] = (
            plan["forecast_avg_demand"] > plan["avg_weekly_demand"] * 1.10
        )
    else:
        plan["replenishment_alert"] = False

    logger.info(
        "Inventory plan computed for %d stores. Alerts: %d.",
        len(plan), plan["replenishment_alert"].sum(),
    )
    return plan


def summarise_inventory(plan: pd.DataFrame) -> str:
    """Return a human-readable plain-English inventory summary."""
    n_stores  = len(plan)
    n_alerts  = plan["replenishment_alert"].sum()
    avg_ss    = plan["safety_stock"].mean()
    avg_rop   = plan["reorder_point"].mean()
    avg_eoq   = plan["eoq"].mean()

    lines = [
        "=" * 60,
        "  INVENTORY PLANNING SUMMARY",
        "=" * 60,
        f"  Stores analysed       : {n_stores}",
        f"  Replenishment alerts  : {n_alerts} stores (demand ↑ >10%)",
        f"  Avg safety stock      : ${avg_ss:,.0f}",
        f"  Avg reorder point     : ${avg_rop:,.0f}",
        f"  Avg EOQ               : ${avg_eoq:,.0f}",
        "=" * 60,
    ]
    return "\n".join(lines)
