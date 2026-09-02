"""
serving/inventory.py — Live inventory plan for one store, computed against
the current live forecast instead of the static batch output in
outputs/reports/inventory_plan.csv.

Reuses src/inventory.py's exact formulas (compute_safety_stock,
compute_reorder_point, compute_eoq) -- no reimplementation, so a change to
the underlying inventory math only ever needs to happen in one place.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.inventory import compute_eoq, compute_reorder_point, compute_safety_stock
from .model_store import ModelBundle


@dataclass(frozen=True)
class LiveInventoryPlan:
    store_id: int
    demand_used: float          # the live forecast driving this plan
    avg_weekly_demand: float    # historical baseline, for comparison
    safety_stock: float
    reorder_point: float
    eoq: float
    recommended_order_qty: float
    replenishment_alert: bool


def build_live_inventory_plan(
    store_id: int,
    forecast: float,
    bundle: ModelBundle,
    *,
    lead_time_weeks: float,
    z_score: float,
    ordering_cost: float,
    unit_value: float,
    holding_cost_pct: float,
    review_period_weeks: float,
) -> LiveInventoryPlan:
    stats = bundle.demand_stats_for(store_id)

    safety_stock = compute_safety_stock(stats.std_weekly_demand, lead_time_weeks, z_score)
    reorder_point = compute_reorder_point(forecast, stats.std_weekly_demand, lead_time_weeks, z_score)
    eoq = compute_eoq(forecast, ordering_cost, unit_value, holding_cost_pct)
    # Same formula as build_inventory_plan(): cover review period + lead
    # time at forecasted demand.
    recommended_order_qty = forecast * (lead_time_weeks + review_period_weeks) + safety_stock
    # Same >10%-over-historical-average rule as the batch pipeline.
    replenishment_alert = forecast > stats.avg_weekly_demand * 1.10

    return LiveInventoryPlan(
        store_id=store_id,
        demand_used=round(forecast, 2),
        avg_weekly_demand=round(stats.avg_weekly_demand, 2),
        safety_stock=round(safety_stock, 2),
        reorder_point=round(reorder_point, 2),
        eoq=round(eoq, 2),
        recommended_order_qty=round(recommended_order_qty, 2),
        replenishment_alert=bool(replenishment_alert),
    )
