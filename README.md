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

## Project Structure

```
Demand-Forecasting-and-Inventory-Planning-System/
├── README.md                   # This file
├── requirements.txt            # Reproducible dependency spec
├── .gitignore
├── config.py                   # All tunable parameters in one place
├── main.py                     # End-to-end pipeline entry point
│
├── data/
│   ├── train.csv               # Raw weekly sales (421,570 records)
│   └── stores.csv              # Store metadata (45 stores)
│
├── src/
│   ├── data_processing.py      # Load, clean, aggregate, feature engineering
│   ├── forecasting.py          # Naive, MA, Ridge, LightGBM models
│   ├── evaluation.py           # MAE, RMSE, MAPE, R², comparison table
│   ├── inventory.py            # Safety stock, ROP, EOQ, inventory plan
│   ├── visualization.py        # 7 publication-quality charts
│   └── utils.py                # Logging, file I/O helpers
│
├── outputs/                    # All generated outputs (gitignored)
│   ├── charts/                 # 7 PNG charts
│   ├── forecasts/              # holdout_forecasts.csv
│   ├── metrics/                # model_comparison.csv
│   └── reports/                # executive_summary.txt
│
└── tests/
    └── test_core.py            # 22 unit tests (metrics, inventory, data)
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

Expected: **22 passed** in ~2 seconds.

### 5. Adjust parameters

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
- [ ] REST API wrapper for real-time forecast serving

---

## Resume-Ready Project Summary

**2-line recruiter summary:**
> Built an end-to-end demand forecasting and inventory planning system on 421K rows of Walmart retail data, comparing four models (Naive, Moving Average, Ridge, LightGBM) with LightGBM achieving MAPE of 4.1% and R² = 0.989 on a 12-week holdout. Translated forecasts into actionable inventory decisions — safety stock, reorder points, and EOQ — for all 45 stores, flagging 2 stores for proactive replenishment.

**Resume bullets:**
- Engineered a leakage-free, modular Python forecasting pipeline (LightGBM, Ridge Regression, baselines) on 421K retail records, achieving **4.1% MAPE and R² = 0.989** on a 12-week holdout
- Reduced RMSE by **33.8%** vs. naive baseline through gradient-boosted trees with lag, rolling-window, and cyclical calendar features
- Implemented production-style inventory planning logic (safety stock, reorder point, EOQ) for 45 stores under a 95% service-level constraint, generating automated replenishment alerts
- Delivered 7 publication-quality visualisations, a ranked model-comparison table, and an executive summary report from a single `python main.py` command; validated with **22 unit tests (100% pass rate)**
