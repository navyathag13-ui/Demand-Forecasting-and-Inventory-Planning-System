"""
scripts/simulate_stream.py — Streams weekly sales events to POST /events/sales
on an interval, then checks GET /forecast (and /inventory) for a highlighted
store after each week. Makes the whole loop -- ingest, feature update, live
forecast changing -- demonstrable end to end without manual curl commands.

Data source: the real 12-week holdout set if data/train.csv + data/stores.csv
are present (these are weeks the model never trained on -- a genuine "new
data arriving" simulation). Falls back to synthetic future weeks, generated
from the persisted model's per-store demand stats, if the raw data files
aren't available.

Usage
-----
    python scripts/simulate_stream.py
    python scripts/simulate_stream.py --interval 1 --highlight-store 5
    python scripts/simulate_stream.py --weeks 4 --base-url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from pathlib import Path

import httpx
import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.data_processing import build_store_weekly, load_raw_data, prepare_features, train_holdout_split
from src.utils import setup_logging

logger = logging.getLogger(__name__)


def load_stream_weeks(weeks_cap: int | None = None) -> list[tuple[pd.Timestamp, pd.DataFrame]]:
    """
    Chronological list of (date, DataFrame[Store, Weekly_Sales, IsHoliday]).

    Uses the real 12-week holdout set when data/ is present. Falls back to
    synthetic weeks (derived from the persisted model's demand stats) when
    it isn't -- e.g. a fresh clone that hasn't downloaded the Kaggle CSVs.
    """
    try:
        train_raw, stores = load_raw_data(config.DATA_DIR)
    except FileNotFoundError:
        logger.info("data/train.csv or data/stores.csv not found -- generating synthetic weeks instead.")
        return _synthetic_weeks(weeks_cap or 6)

    df_weekly = build_store_weekly(train_raw, stores)
    df_features = prepare_features(df_weekly)
    _, holdout_df = train_holdout_split(df_features, holdout_weeks=config.HOLDOUT_WEEKS)

    weeks = sorted(
        ((date, group[["Store", "Weekly_Sales", "IsHoliday"]]) for date, group in holdout_df.groupby("Date")),
        key=lambda item: item[0],
    )
    return weeks[:weeks_cap] if weeks_cap else weeks


def _synthetic_weeks(n_weeks: int) -> list[tuple[pd.Timestamp, pd.DataFrame]]:
    if not config.MODEL_PATH.exists():
        raise SystemExit(
            "No data/ files and no persisted model to fall back on. "
            "Run `python scripts/train_and_persist.py` first, or add "
            "data/train.csv + data/stores.csv."
        )
    bundle = joblib.load(config.MODEL_PATH)
    rng = random.Random(7)  # deterministic synthetic stream, same reasoning as seed data elsewhere
    start = pd.Timestamp(bundle["min_date"]) + pd.Timedelta(weeks=200)

    weeks = []
    for i in range(n_weeks):
        date = start + pd.Timedelta(weeks=i)
        rows = [
            {
                "Store": store_id,
                "Weekly_Sales": max(0.0, stats["avg_weekly_demand"] * (1 + rng.uniform(-0.05, 0.05))),
                "IsHoliday": False,
            }
            for store_id, stats in bundle["demand_stats"].items()
        ]
        weeks.append((date, pd.DataFrame(rows)))
    return weeks


def stream(client: httpx.Client, base_url: str, weeks, interval: float, highlight_store: int | None):
    for date, rows in weeks:
        week_str = pd.Timestamp(date).date().isoformat()
        logger.info("Week %s: posting %d store events ...", week_str, len(rows))

        for row in rows.itertuples():
            payload = {
                "store_id": int(row.Store),
                "week_ending_date": week_str,
                "sales": float(row.Weekly_Sales),
                "is_holiday": bool(row.IsHoliday),
            }
            try:
                resp = client.post(f"{base_url}/events/sales", json=payload)
                if resp.status_code != 201:
                    logger.warning("  store %s -> %s: %s", row.Store, resp.status_code, resp.text)
            except httpx.HTTPError as exc:
                logger.warning("  store %s -> request failed: %s", row.Store, exc)

        if highlight_store is not None:
            try:
                forecast = client.get(f"{base_url}/forecast/{highlight_store}").json()
                inventory = client.get(f"{base_url}/inventory/{highlight_store}").json()
                logger.info(
                    "  -> store %s forecast for %s: $%s (replenishment_alert=%s)",
                    highlight_store,
                    forecast.get("target_week"),
                    f"{forecast.get('predicted_demand', 0):,.0f}",
                    inventory.get("replenishment_alert"),
                )
            except httpx.HTTPError as exc:
                logger.warning("  couldn't fetch forecast/inventory for store %s: %s", highlight_store, exc)

        time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--interval", type=float, default=2.0, help="Seconds between weeks.")
    parser.add_argument("--weeks", type=int, default=None, help="Cap the number of weeks streamed.")
    parser.add_argument(
        "--highlight-store", type=int, default=1,
        help="Store to print forecast/inventory for after each week. Pass 0 to disable.",
    )
    args = parser.parse_args()

    setup_logging()
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one INFO line per request is too noisy for a 45-store week
    weeks = load_stream_weeks(args.weeks)
    if not weeks:
        raise SystemExit("No weeks to stream.")

    logger.info("Streaming %d week(s) to %s, %.1fs apart.", len(weeks), args.base_url, args.interval)
    with httpx.Client(timeout=5.0) as client:
        stream(client, args.base_url, weeks, args.interval, args.highlight_store or None)
    logger.info("Done.")


if __name__ == "__main__":
    main()
