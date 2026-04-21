"""
evaluation.py — Forecasting accuracy metrics and model comparison utilities.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mape(actual: np.ndarray, predicted: np.ndarray, eps: float = 1e-6) -> float:
    """
    Mean Absolute Percentage Error (%).

    Rows where actual ≈ 0 are excluded to avoid division instability.
    """
    mask = np.abs(actual) > eps
    if mask.sum() == 0:
        return float("nan")
    return float(100 * np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])))


def r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Coefficient of determination (R²)."""
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


# ---------------------------------------------------------------------------
# Unified evaluation
# ---------------------------------------------------------------------------

def evaluate(
    actual: np.ndarray,
    predicted: np.ndarray,
    model_name: str = "model",
) -> dict[str, float]:
    """
    Compute all metrics for one model and return as a labelled dict.

    Parameters
    ----------
    actual, predicted : array-like of shape (n,)
    model_name : str
        Used as the row label in comparison tables.

    Returns
    -------
    dict with keys: model, MAE, RMSE, MAPE, R2
    """
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    results = {
        "Model": model_name,
        "MAE":   round(mae(a, p),  2),
        "RMSE":  round(rmse(a, p), 2),
        "MAPE":  round(mape(a, p), 2),
        "R2":    round(r2(a, p),   4),
    }
    logger.info("[%s] MAE=%.0f | RMSE=%.0f | MAPE=%.1f%% | R²=%.3f",
                model_name, results["MAE"], results["RMSE"],
                results["MAPE"], results["R2"])
    return results


def compare_models(results: list[dict]) -> pd.DataFrame:
    """
    Build a ranked comparison table from a list of evaluate() outputs.

    Sorted by RMSE ascending (lower is better).
    """
    df = pd.DataFrame(results).set_index("Model")
    df = df.sort_values("RMSE")
    df["Rank"] = range(1, len(df) + 1)

    # Improvement over naive baseline (last row after sort by RMSE descending)
    worst_rmse = df["RMSE"].max()
    df["RMSE_vs_Naive (%)"] = (
        (worst_rmse - df["RMSE"]) / worst_rmse * 100
    ).round(1)
    return df
