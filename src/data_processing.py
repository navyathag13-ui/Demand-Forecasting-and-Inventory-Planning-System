"""
data_processing.py — Load, clean, and engineer features for demand forecasting.

All transformations are deterministic and leakage-free (no future data used
when computing lag / rolling features).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_raw_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw training sales and store-metadata CSVs.

    Parameters
    ----------
    data_dir : Path
        Directory that contains train.csv and stores.csv.

    Returns
    -------
    train : pd.DataFrame
        Raw weekly sales records.
    stores : pd.DataFrame
        Store type and size metadata.
    """
    def _find(name: str) -> Path:
        candidates = [data_dir / name, data_dir.parent / name]
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(
            f"'{name}' not found in {data_dir} or {data_dir.parent}."
        )

    train  = pd.read_csv(_find("train.csv"),  parse_dates=["Date"])
    stores = pd.read_csv(_find("stores.csv"))
    logger.info("Loaded %d sales rows across %d stores.",
                len(train), stores["Store"].nunique())
    return train, stores


# ---------------------------------------------------------------------------
# Cleaning & aggregation
# ---------------------------------------------------------------------------

def build_store_weekly(
    train: pd.DataFrame,
    stores: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge, clean, and aggregate to one row per (Store, Date).

    Steps
    -----
    1. Drop negative sales (returns / reversals — not demand signals).
    2. Merge store metadata.
    3. Sum all department sales within each store-week.
    4. Sort chronologically within each store.

    Returns
    -------
    pd.DataFrame with columns:
        Store, Date, IsHoliday, Type, Size, Weekly_Sales
    """
    clean = train[train["Weekly_Sales"] >= 0].copy()
    n_dropped = len(train) - len(clean)
    if n_dropped:
        logger.info("Dropped %d negative-sales rows (%.1f%%).",
                    n_dropped, 100 * n_dropped / len(train))

    merged = clean.merge(stores, on="Store", how="left")

    agg = (
        merged
        .groupby(["Store", "Date", "IsHoliday", "Type", "Size"])
        .agg(Weekly_Sales=("Weekly_Sales", "sum"))
        .reset_index()
        .sort_values(["Store", "Date"])
        .reset_index(drop=True)
    )
    logger.info("Store-week dataset: %d rows, %d stores, %d unique weeks.",
                len(agg), agg["Store"].nunique(), agg["Date"].nunique())
    return agg


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append calendar and cyclical time features.

    Cyclical encoding (sin/cos) prevents the model from treating
    week 52 → week 1 as a large discontinuity.
    """
    df = df.copy()
    df["year"]         = df["Date"].dt.year
    df["month"]        = df["Date"].dt.month
    df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
    df["quarter"]      = df["Date"].dt.quarter
    df["is_month_end"] = df["Date"].dt.is_month_end.astype(int)

    # Cyclical encodings
    df["week_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Trend index (weeks since first observation)
    min_date = df["Date"].min()
    df["trend"] = ((df["Date"] - min_date).dt.days / 7).astype(int)
    return df


def add_store_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode store type and normalise store size."""
    df = df.copy()
    type_map = {"A": 0, "B": 1, "C": 2}
    df["type_code"]   = df["Type"].map(type_map).astype(int)
    size_mean         = df["Size"].mean()
    size_std          = df["Size"].std()
    df["size_norm"]   = (df["Size"] - size_mean) / size_std
    return df


def add_lag_features(df: pd.DataFrame, lags: list[int] = [1, 2, 4, 12]) -> pd.DataFrame:
    """
    Add lag and rolling-window features per store.

    All lags are computed within each store independently to prevent
    cross-store contamination. NaN rows introduced by lagging are
    retained here; callers should dropna() after train/holdout split.
    """
    df = df.copy().sort_values(["Store", "Date"])

    for lag in lags:
        df[f"sales_lag_{lag}"] = df.groupby("Store")["Weekly_Sales"].shift(lag)

    for window in [4, 12]:
        df[f"sales_roll_{window}"] = (
            df.groupby("Store")["Weekly_Sales"]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        )
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature-engineering pipeline applied to store-week data.

    Returns the enriched DataFrame. Callers should dropna() before fitting.
    """
    df = add_time_features(df)
    df = add_store_features(df)
    df = add_lag_features(df)
    return df


# ---------------------------------------------------------------------------
# Train / holdout split
# ---------------------------------------------------------------------------

def train_holdout_split(
    df: pd.DataFrame,
    holdout_weeks: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological split: the last `holdout_weeks` unique dates form the holdout.

    Ensures NO future information leaks into training features.
    """
    dates_sorted = sorted(df["Date"].unique())
    cutoff_date  = dates_sorted[-holdout_weeks]

    train_df   = df[df["Date"] < cutoff_date].dropna()
    holdout_df = df[df["Date"] >= cutoff_date].dropna()

    logger.info(
        "Split: %d train rows (up to %s) | %d holdout rows (%s onward).",
        len(train_df), dates_sorted[-holdout_weeks - 1].date(),
        len(holdout_df), cutoff_date.date(),
    )
    return train_df, holdout_df
