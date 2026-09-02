# Demand Forecasting and Inventory Planning System

**A production-style analytics pipeline that forecasts retail demand and translates forecasts into concrete inventory decisions — safety stock, reorder points, and replenishment alerts — using 3 years of Walmart weekly sales data.**

---

## Business Problem

Retailers lose an estimated $1.75 trillion globally each year to inventory distortion — excess stock tying up capital, and stockouts driving customers to competitors. Accurate demand forecasting is the foundation of every inventory management system: without it, safety stock calculations are guesses, reorder points are arbitrary, and replenishment decisions are reactive rather than proactive.

This project demonstrates a complete, end-to-end pipeline that connects demand forecasting directly to inventory planning, answering two practical business questions:

1. **What will demand look like over the next 12 weeks?**
2. **Given that forecast, what should our inventory position be right now?**

---

## Dataset

| Property | Detail |
|---|---|
| Source | [Walmart Store Sales Forecasting (Kaggle)](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting) |
| Period | February 2010 – October 2012 |
| Stores | 45 stores across three formats: A (large), B (medium), C (small) |
| Granularity | Weekly, aggregated at store level |
| Records | 421,570 raw rows → 6,435 store-week records after aggregation |
| Target | Total weekly sales revenue per store ($) |

Negative sales records (returns/adjustments) were excluded before modeling, as they do not represent demand.

---

## Methodology

```
Raw Data (train.csv, stores.csv)
         │
         ▼
    Data Cleaning
    ├── Remove negative sales (1,285 rows, 0.3%)
    ├── Merge store type & size metadata
    └── Aggregate all departments → store-week level
         │
         ▼
  Feature Engineering
    ├── Calendar: year, quarter, month, week-of-year
    ├── Cyclical encoding: sin/cos of week and month (avoids week 52→1 jump)
    ├── Lag features: t-1, t-2, t-4, t-12 week sales per store
    ├── Rolling means: 4-week and 12-week trailing averages
    ├── Store metadata: type encoding, normalised size
    └── Binary: IsHoliday, is_month_end
         │
         ▼
  Chronological Train/Holdout Split
  ├── Train: 2010-02-05 → 2012-08-03  (5,355 rows)
  └── Holdout: 2012-08-10 → 2012-10-26  (540 rows, 12 weeks)
         │
         ▼
  Four Forecasting Models
  ├── Naive Baseline (last observed value)
  ├── Moving Average (4-week trailing mean)
  ├── Ridge Regression (calendar + store features)
  └── LightGBM (all features + lag/rolling)
         │
         ▼
  Evaluation on Holdout Set
  (MAE, RMSE, MAPE, R²)
         │
         ▼
  Inventory Planning
  ├── Safety stock per store
  ├── Reorder point
  ├── Economic Order Quantity
  └── Replenishment alerts
         │
         ▼
  Outputs: forecasts, metrics, charts, executive report
```

---

## Forecasting Models

### 1. Naive Baseline
Predicts next week's sales as the most recent observed value. This is the floor — any useful model must outperform it.

### 2. Moving Average (4-week)
Predicts demand as the rolling mean of the last 4 weeks. Smooths noise but cannot capture trends or holiday spikes.

### 3. Ridge Regression
Ordinary least-squares with L2 regularisation on calendar and store features. Captures global trend and seasonality linearly; fast and interpretable.

### 4. LightGBM *(best performer)*
Gradient-boosted trees trained on the full feature set. Captures non-linear interactions between lag patterns, calendar cycles, and store characteristics. Uses a chronological 10% validation split for early stopping to prevent overfitting.

---

## Results

All metrics computed on the 12-week holdout set (unseen during training).

| Model | MAE | RMSE | MAPE | R² |
|---|---|---|---|---|
| **LightGBM** | **$40,164** | **$56,481** | **4.1%** | **0.989** |
| Moving Average (4w) | $49,290 | $69,849 | 4.8% | 0.982 |
| Naive (Last Value) | $59,240 | $85,343 | 6.0% | 0.974 |
| Ridge Regression | $229,001 | $294,137 | 24.9% | 0.688 |

**LightGBM reduces RMSE by 33.8% vs the Moving Average baseline and by 33.8% vs the Naive baseline.**

The Ridge Regression performs poorly because store-level weekly demand is highly non-linear (holiday spikes, seasonal patterns, store-size interactions) that a linear model cannot capture without extensive manual feature engineering.

### Key Drivers (LightGBM Feature Importance)

The top predictive features are lag-based (recent sales are the strongest predictor of next week's sales), followed by rolling averages and calendar features:

1. `sales_lag_1` — last week's sales
2. `sales_roll_4` — 4-week rolling mean
3. `sales_lag_4` — 4-week lag
4. `sales_roll_12` — 12-week rolling mean
5. `trend` — linear time index
6. `week_sin / week_cos` — cyclical week encoding

---

## Inventory Planning

Using the LightGBM forecasts as forward-looking demand signals, we compute industry-standard inventory policy parameters for all 45 stores.

**Model: Continuous-review (s, Q) policy**
- *s* (reorder point) = trigger replenishment when stock falls here
- *Q* (order quantity) = Economic Order Quantity

**Assumptions** (illustrative — configurable in `config.py`):
- Lead time: 2 weeks
- Service level: 95% → Z-score = 1.645
- Review cycle: 4 weeks
- Holding cost: 20% of unit value per year

**Formulas:**
```
Safety Stock    = Z × σ_demand × √(lead_time)
Reorder Point   = μ_demand × lead_time + Safety_Stock
EOQ             = √(2 × D_annual × K / h)
```

### System-wide Results (45 stores)

| Metric | Value |
|---|---|
| Avg weekly demand per store | $1.16M |
| Avg safety stock | $347,448 |
| Avg reorder point | $2.39M |
| Avg EOQ | $54,579 |
| Replenishment alerts (demand ↑ >10%) | **2 stores** |

Stores 38 and 44 are flagged: their forecasted demand exceeds historical average by more than 10%, signalling that current inventory levels may be insufficient for the upcoming period.

---

## Charts

All charts are generated automatically in `outputs/charts/`:

| File | Description |
|---|---|
| `01_demand_trend.png` | Weekly sales trend by store type over 3 years |
| `02_actual_vs_predicted.png` | Actual vs predicted demand (all models, holdout period) |
| `03_model_comparison.png` | Side-by-side MAE / RMSE / MAPE across models |
| `04_feature_importance.png` | Top 15 LightGBM predictors |
| `05_inventory_plan.png` | Reorder point vs demand scatter (alerts highlighted) |
| `06_demand_by_store_type.png` | Sales distribution box plot by store type |
| `07_residuals.png` | LightGBM residual scatter + distribution |

---

## Real-Time Serving Layer

The pipeline above answers "what's the plan" once, in a batch run. The serving layer answers the same two questions (what will demand look like, what should inventory be) *live*, as new weekly sales data arrives — without retraining anything.

**Near-real-time, not streaming, on purpose.** The data is weekly, store-level sales. A service that promised millisecond-latency updates on top of weekly data would be solving a problem this dataset doesn't have. "Live" here means: the moment a week's actual sales are ingested, every forecast and inventory number for that store reflects it on the very next request — not a nightly batch, not a polling delay, but also not pretending the underlying signal changes faster than once a week.

**The model is trained offline, once, and never retrains itself.** `scripts/train_and_persist.py` fits `LGBMForecaster` on the exact same `train_df`/holdout split `main.py` already validated (same seed, same `LGBM_PARAMS`) — it's the same model whose 4.1% MAPE is documented above, not a new one — and `joblib`-dumps it, along with the feature-engineering constants training used (`size_mean`, `size_std`, the `trend` reference date, per-store historical demand stats). The API loads that bundle once at startup and holds it in memory for the life of the process.

```
POST /events/sales  ──▶  per-store rolling history  ──▶  recompute lag/rolling/
  (new weekly record)      (in-memory, one per store)      calendar features
                                                                    │
                                                                    ▼
                                                          cached feature vector
                                                          (NOT a cached prediction)
                                                                    │
                        ┌───────────────────────────────────────────┤
                        ▼                                           ▼
              GET /forecast/{store}                       GET /inventory/{store}
              runs the persisted model                    same live forecast, fed through
              fresh, every request                        src/inventory.py's own formulas
```

That distinction — cache the *features*, never the *prediction* — is the one piece of this layer most likely to go subtly wrong if built carelessly. A cached forecast would go stale the instant a new event arrived and nobody would notice until the numbers stopped matching reality. Both `/forecast` and `/inventory` re-run inference against whatever the current feature snapshot is at request time, every time; only the (comparatively expensive to rebuild) feature vector itself is cached.

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/events/sales` | Ingest one store's weekly sales record. Rejects a duplicate or out-of-order week (`409`) or an unknown store (`404`) instead of silently overwriting. |
| GET | `/forecast/{store_id}` | Predicted demand for the week *after* the store's most recently ingested week, run fresh against the current feature vector. `404` if the store has no history yet. |
| GET | `/inventory/{store_id}` | Safety stock, reorder point, EOQ, and a replenishment alert (>10% over historical average), computed against the live forecast via the same formulas the batch pipeline uses. `404` under the same condition as `/forecast`. |
| GET | `/health` | Model-loaded flag, uptime, how many stores have received data yet, most recent event timestamp. |
| GET | `/metrics` | Total events ingested, total forecasts served, average inference latency — plain JSON, not a Prometheus setup. |

### What "next week's forecast" actually means

When week W's sales arrive, the model is asked to predict W+1, not W. Concretely: `sales_lag_1` becomes W's own value, `sales_lag_2`/`sales_lag_4`/`sales_lag_12` shift the same way, and the rolling means include W — but the *calendar* features (`trend`, `week_sin`/`cos`, `month_sin`/`cos`, `quarter`, `is_month_end`) are computed for W+1's date, since that's the week actually being forecasted. One thing this can't know: whether W+1 is a holiday. The sales feed only reports whether the week that just happened was a holiday, not whether the upcoming one will be — that's calendar knowledge, not a sales signal. `IsHoliday` for the forecasted week defaults to `False`; a production version of this would source it from a known holiday calendar instead. Documented here, not silently guessed at.

### Concurrency

FastAPI runs sync endpoints in a thread pool, so two events for the same store arriving close together is a real scenario, not a hypothetical. Each store's history is guarded by its own lock covering the full check-then-append-then-recompute sequence, and the cached feature snapshot is swapped in with a single reference assignment as the last step — a concurrent reader always sees either the fully-old or fully-new snapshot, never a partially-built one. `tests/test_serving.py` fires 20 threads at the same store (distinct weeks, and separately, ten threads posting the *same* week) to check this holds, not just reasons about it in a comment.

### What a deliberate review pass caught

Once this felt feature-complete, a dedicated review pass (concurrency correctness, feature-math correctness against `src/data_processing.py`, and general API/cleanup) found the design itself held up — no data races, no stale-cache paths, and the incrementally-recomputed features matched `src/`'s own training-time computation exactly, numerically verified against the real persisted model. It did catch real gaps in the edges:

- `/forecast` and `/inventory` let a model-inference failure surface as a raw 500 instead of a clean error — now caught and mapped to a `503`.
- `src/inventory.py`'s `build_inventory_plan` had `+ 4` hardcoded into the recommended-order-quantity formula instead of taking `review_period_weeks` as a parameter like every other formula input. Harmless today only because `config.REVIEW_PERIOD_WEEKS` also happens to be 4 — but it meant the batch pipeline and this serving layer would silently disagree the moment that config value ever changed. Now parameterized properly, sourced from config at both call sites.
- `scripts/train_and_persist.py` wrote the model bundle directly to its final path; a kill mid-write would've left a corrupt `.joblib` that `load_bundle()` would then fail on ambiguously. Now writes to a temp file and renames atomically.
- Every forecast/inventory test used a stand-in model that ignores its input, so nothing actually exercised the real `LGBMForecaster` through the serving path — a drift between the feature dict's keys and the model's expected columns would have degraded predictions silently rather than failing a test. Added one integration test that loads the real persisted model and runs a real feature snapshot through it end to end.

### Running it

```bash
# 1. Train and persist the model (writes models/lgbm_forecaster.joblib)
python scripts/train_and_persist.py

# 2. Start the API
uvicorn serving.app:app --reload

# 3. In another terminal, stream events against it
python scripts/simulate_stream.py --interval 1 --highlight-store 1
```

The simulator streams the real 12-week holdout set — weeks the model never trained on — if `data/` is present, or falls back to synthetic future weeks derived from each store's historical demand stats if it isn't. After each week it prints the highlighted store's updated forecast and replenishment alert, so the full ingest → feature update → forecast change loop is visible without manually curling anything.

---

## Project Structure

```
Demand-Forecasting-and-Inventory-Planning-System/
├── README.md                   # This file
├── requirements.txt            # Reproducible dependency spec
├── .gitignore
├── config.py                   # All tunable parameters in one place
├── main.py                     # End-to-end batch pipeline entry point
│
├── data/
│   ├── train.csv               # Raw weekly sales (421,570 records)
│   └── stores.csv              # Store metadata (45 stores)
│
├── src/                         # The batch pipeline -- proven, unchanged
│   ├── data_processing.py      # Load, clean, aggregate, feature engineering
│   ├── forecasting.py          # Naive, MA, Ridge, LightGBM models
│   ├── evaluation.py           # MAE, RMSE, MAPE, R², comparison table
│   ├── inventory.py            # Safety stock, ROP, EOQ, inventory plan
│   ├── visualization.py        # 7 publication-quality charts
│   └── utils.py                # Logging, file I/O helpers
│
├── serving/                     # The near-real-time serving layer (new)
│   ├── app.py                  # FastAPI app: events/forecast/inventory/health/metrics
│   ├── model_store.py          # Loads the persisted model bundle once at startup
│   ├── feature_state.py        # Per-store rolling history + feature recomputation
│   ├── inference.py            # Runs the persisted model against a cached snapshot
│   ├── inventory.py            # Live inventory plan (reuses src/inventory.py's formulas)
│   ├── metrics.py              # In-memory counters for GET /metrics
│   └── schemas.py              # Pydantic request/response models
│
├── scripts/
│   ├── train_and_persist.py    # Fits the model exactly as main.py does, joblib-dumps it
│   └── simulate_stream.py      # Streams weekly events to the running API on an interval
│
├── models/                      # Persisted model bundle (gitignored)
│   └── lgbm_forecaster.joblib
│
├── outputs/                    # All generated outputs (gitignored)
│   ├── charts/                 # 7 PNG charts
│   ├── forecasts/              # holdout_forecasts.csv
│   ├── metrics/                # model_comparison.csv
│   └── reports/                # executive_summary.txt
│
└── tests/
    ├── test_core.py            # 22 unit tests (metrics, inventory, data)
    └── test_serving.py         # 37 unit + HTTP tests (serving layer, including concurrency)
```

---

## How to Run

### 1. Clone and set up

```bash
git clone https://github.com/navyathag13-ui/Demand-Forecasting-and-Inventory-Planning-System.git
cd Demand-Forecasting-and-Inventory-Planning-System
pip install -r requirements.txt
```

### 2. Add data files

Download `train.csv` and `stores.csv` from [Kaggle](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting) and place them in the `data/` directory.

### 3. Run the pipeline

```bash
python main.py
```

All outputs are written to `outputs/` automatically. Full run completes in ~2 seconds.

### 4. Run tests

```bash
python -m pytest tests/ -v
```

Expected: **59 passed** in a few seconds (22 batch-pipeline tests + 37 serving-layer tests; one of those is skipped instead of passed if you haven't run `train_and_persist.py` yet, since it loads the real model).

### 5. Try the real-time serving layer

```bash
python scripts/train_and_persist.py       # writes models/lgbm_forecaster.joblib
uvicorn serving.app:app --reload          # starts the API on :8000
python scripts/simulate_stream.py         # in another terminal -- streams events in
```

See [Real-Time Serving Layer](#real-time-serving-layer) above for what's actually happening.

### 6. Adjust parameters

Edit `config.py` to change:
- `HOLDOUT_WEEKS` — validation window size
- `LEAD_TIME_WEEKS` — supply chain lead time assumption
- `SERVICE_LEVEL` — target fill rate (e.g. 0.95, 0.99)
- `LGBM_PARAMS` — model hyperparameters

---

## Business Insights

1. **Lag-1 is the dominant signal.** Last week's sales explains more variance than all calendar features combined. This means reactive replenishment (based on what just sold) can capture most of the signal — but lag-12 (same week last year) is important for seasonal correction.

2. **Holiday spikes are partially captured.** The `IsHoliday` flag and nearby lag features allow the model to anticipate holiday weeks, reducing surprise stockouts during peak periods.

3. **Store type matters more than size.** Type-A stores (largest format) have 3× the weekly demand variance of Type-C stores, but their relative MAPE is comparable — suggesting the model scales well across formats.

4. **2 stores need immediate attention.** Stores 38 and 44 are forecast to see >10% demand increases vs. historical averages. Without proactive replenishment, these stores risk stockouts during the forecast window.

5. **Safety stock investment is justified.** Average safety stock of $347K per store at 95% service level is a reasonable trade-off: preventing a stockout event that might cost multiples of that figure in lost revenue.

---

## Limitations

- **No unit-level data.** Sales are in dollars, not units. Inventory formulas are dollar-denominated and depend on assumed unit economics (configurable in `config.py`).
- **No exogenous variables.** The Kaggle dataset includes fuel prices, temperature, and CPI (`features.csv`). These are not included here but would likely improve accuracy.
- **Static parameters.** Lead time, ordering cost, and holding cost are assumed constants. A production system would pull these from an ERP.
- **Store-level aggregation.** This pipeline forecasts at store level (sum of all departments). Department-level forecasting is architecturally supported but increases complexity and run time.

---

## Future Improvements

- [ ] Add `features.csv` exogenous variables (temperature, CPI, fuel price)
- [ ] Department-level forecasting for SKU-level inventory decisions
- [ ] Hyperparameter tuning with Optuna (objective: MAPE on holdout)
- [ ] Prediction intervals for safety stock sizing under forecast uncertainty
- [ ] Store-specific lead time parameterisation
- [x] REST API wrapper for real-time forecast serving -- see [Real-Time Serving Layer](#real-time-serving-layer)

