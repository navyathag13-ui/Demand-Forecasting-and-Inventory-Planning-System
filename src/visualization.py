"""
visualization.py — Publication-quality charts for demand forecasting results.

All functions save figures to disk and optionally return the Figure object.
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------

PALETTE   = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]
DPI       = 150
FONT_SIZE = 11

plt.rcParams.update({
    "font.family":   "DejaVu Sans",
    "font.size":     FONT_SIZE,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":    DPI,
})


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved chart → %s", path)


# ---------------------------------------------------------------------------
# 1. Demand trend
# ---------------------------------------------------------------------------

def plot_demand_trend(
    df: pd.DataFrame,
    store_ids: Optional[list[int]] = None,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Line chart of weekly sales over time for selected stores.

    Highlights the holiday flag with vertical bands.
    """
    if store_ids is None:
        # Pick 3 representative stores: one of each type
        store_ids = df.groupby("Type")["Store"].first().values.tolist()[:3]

    subset = df[df["Store"].isin(store_ids)].copy()

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, sid in enumerate(store_ids):
        s = subset[subset["Store"] == sid]
        label = f"Store {sid} (Type {s['Type'].iloc[0]})"
        ax.plot(s["Date"], s["Weekly_Sales"] / 1e6,
                color=PALETTE[i % len(PALETTE)], label=label, linewidth=1.6)

    # Holiday shading
    holidays = subset[subset["IsHoliday"]]["Date"].unique()
    for hdate in holidays:
        ax.axvspan(hdate, hdate + pd.Timedelta("7D"),
                   alpha=0.08, color="#FF5722", linewidth=0)

    ax.set_title("Weekly Demand Trend by Store", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Weekly Sales ($M)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.1fM"))
    ax.legend(frameon=False)
    ax.annotate("Holiday periods (shaded)", xy=(0.01, 0.94),
                xycoords="axes fraction", fontsize=9, color="#FF5722")
    fig.tight_layout()

    if output_path:
        _save(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# 2. Actual vs Predicted
# ---------------------------------------------------------------------------

def plot_actual_vs_predicted(
    results: dict[str, np.ndarray],
    actual: np.ndarray,
    dates: np.ndarray,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Overlay actual demand with predictions from multiple models.

    Parameters
    ----------
    results : {model_name: predictions array}
    actual  : array of true values
    dates   : date array aligned with actual/predictions
    """
    fig, ax = plt.subplots(figsize=(13, 5))

    ax.plot(dates, actual / 1e6, color="black",
            linewidth=2, label="Actual", zorder=5)

    for i, (name, preds) in enumerate(results.items()):
        ax.plot(dates, preds / 1e6, color=PALETTE[i % len(PALETTE)],
                linewidth=1.5, linestyle="--", alpha=0.9, label=name)

    ax.set_title("Actual vs Predicted Weekly Demand (Holdout Period)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Weekly Sales ($M)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.1fM"))
    ax.legend(frameon=False)
    fig.tight_layout()

    if output_path:
        _save(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# 3. Model comparison bar chart
# ---------------------------------------------------------------------------

def plot_model_comparison(
    metrics_df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Grouped bar chart comparing MAE, RMSE, MAPE across models.
    """
    models   = metrics_df.index.tolist()
    metrics  = ["MAE", "RMSE", "MAPE"]
    n_models = len(models)
    n_metrics = len(metrics)
    x        = np.arange(n_models)
    width    = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Model Performance Comparison (Holdout Set)",
                 fontsize=14, fontweight="bold")

    for j, (ax, metric) in enumerate(zip(axes, metrics)):
        vals = metrics_df[metric].values
        bars = ax.bar(x, vals, color=PALETTE[:n_models], edgecolor="white",
                      linewidth=0.8)
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=25, ha="right", fontsize=9)
        unit = "%" if metric == "MAPE" else "$"
        ax.set_ylabel(f"{unit}{metric}")
        # Label bars
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                    f"{val:,.0f}" if metric != "MAPE" else f"{val:.1f}%",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    if output_path:
        _save(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# 4. Feature importance
# ---------------------------------------------------------------------------

def plot_feature_importance(
    importances: pd.Series,
    top_n: int = 15,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Horizontal bar chart of LightGBM feature importances."""
    top = importances.head(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(top.index, top.values, color="#2196F3", edgecolor="white")
    ax.set_title(f"LightGBM — Top {top_n} Feature Importances",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance (gain)")
    for bar in bars:
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():,.0f}", va="center", fontsize=8)
    fig.tight_layout()
    if output_path:
        _save(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# 5. Inventory planning chart
# ---------------------------------------------------------------------------

def plot_inventory_plan(
    plan: pd.DataFrame,
    top_n: int = 20,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Scatter plot: average weekly demand vs reorder point, sized by safety stock.
    Alerts highlighted in orange.
    """
    subset = plan.head(top_n).copy()

    fig, ax = plt.subplots(figsize=(11, 6))

    colors = subset["replenishment_alert"].map({True: "#FF5722", False: "#2196F3"})
    sizes  = (subset["safety_stock"] / subset["safety_stock"].max() * 400).clip(lower=30)

    sc = ax.scatter(
        subset["avg_weekly_demand"] / 1e6,
        subset["reorder_point"] / 1e6,
        c=colors, s=sizes, alpha=0.8, edgecolors="white", linewidths=0.8,
    )

    for _, row in subset.iterrows():
        ax.annotate(
            f"S{int(row['Store'])}",
            (row["avg_weekly_demand"] / 1e6, row["reorder_point"] / 1e6),
            fontsize=7.5, ha="center", va="bottom",
        )

    # Legend
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#FF5722",
               markersize=10, label="Replenishment Alert"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2196F3",
               markersize=10, label="Normal"),
    ]
    ax.legend(handles=legend_elems, frameon=False)

    ax.set_title("Inventory Planning — Reorder Point vs Avg Weekly Demand\n"
                 "(bubble size = safety stock level)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Avg Weekly Demand ($M)")
    ax.set_ylabel("Reorder Point ($M)")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("$%.1fM"))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.1fM"))
    fig.tight_layout()

    if output_path:
        _save(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# 6. Demand distribution by store type
# ---------------------------------------------------------------------------

def plot_demand_by_store_type(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Box plot of weekly sales distribution per store type."""
    fig, ax = plt.subplots(figsize=(9, 5))
    types   = sorted(df["Type"].unique())
    data    = [df[df["Type"] == t]["Weekly_Sales"].values / 1e6 for t in types]
    bp = ax.boxplot(data, labels=types, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_title("Weekly Sales Distribution by Store Type",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Store Type")
    ax.set_ylabel("Weekly Sales ($M)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.1fM"))
    fig.tight_layout()
    if output_path:
        _save(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# 7. Residual analysis
# ---------------------------------------------------------------------------

def plot_residuals(
    actual: np.ndarray,
    predicted: np.ndarray,
    model_name: str = "LightGBM",
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Two-panel residual diagnostic: scatter + histogram."""
    residuals = actual - predicted

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Residual Analysis — {model_name}", fontsize=13, fontweight="bold")

    # Residuals vs predicted
    ax1.scatter(predicted / 1e6, residuals / 1e6, alpha=0.35,
                color="#2196F3", s=18, edgecolors="white", linewidths=0.3)
    ax1.axhline(0, color="red", linewidth=1.2, linestyle="--")
    ax1.set_xlabel("Predicted Weekly Sales ($M)")
    ax1.set_ylabel("Residual ($M)")
    ax1.set_title("Residuals vs Predicted")

    # Histogram
    ax2.hist(residuals / 1e6, bins=40, color="#2196F3", edgecolor="white", alpha=0.8)
    ax2.axvline(0, color="red", linewidth=1.2, linestyle="--")
    ax2.set_xlabel("Residual ($M)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Residual Distribution")

    fig.tight_layout()
    if output_path:
        _save(fig, output_path)
    return fig
