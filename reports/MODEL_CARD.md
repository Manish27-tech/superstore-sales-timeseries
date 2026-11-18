# Model card: weekly revenue forecaster

## What it is

- **Task:** forecast total weekly sales revenue 1-13 weeks ahead from a fixed origin.
- **Selected model:** ridge regression on log1p revenue, chosen by mean walk-forward RMSE.
- **Training data:** 194 weekly periods ending 2017-09-18.
- **Evaluation:** 13 held-out weeks, 2017-09-18 to 2017-12-18.
- **Features:** 13-week rolling level, same-week-last-year revenue, week-of-year and month harmonics, quarter, and horizon. All knowable at the origin.

## Performance

| Model | RMSE | MAE | MAPE | MASE | R² |
|---|---|---|---|---|---|
| seasonal naive | 11,719 | 9,577 | 53.5% | 1.00 | -0.81 |
| ridge regression | 10,409 | 9,089 | 46.4% | 0.95 | -0.43 |
| gradient boosting | 9,718 | 8,030 | 41.4% | 0.84 | -0.25 |

Improvement over seasonal naive: **11.2% RMSE**.

## Calibration and uncertainty

- Intervals are empirical 5-95% residual quantiles per horizon from the walk-forward folds.
- Observed coverage on the held-out weeks: 61.5% (nominal 90%). The interval is too narrow for a level shift of the size seen in Q4 2017.

## Where it fails

- Systematic under-prediction in the evaluation quarter: mean error -6,180 USD per week, 10 of 13 weeks low.
- Median absolute percentage error 46.8%; only 1 of 13 weeks land within 20%.
- Weekly revenue is lumpy: a single copier order can add $14k to a $12k week. No model on this sample fixes that.

## Intended use

Planning support and a teaching example of honest time-series evaluation. Not a replacement for a committed financial forecast; not suitable for inventory commitments without a human review of pipeline and promotions.

## Features that carry the model

| Feature | Permutation importance (RMSE, USD) |
|---|---|
| `target_week_of_year_cos` | 990 ± 769 |
| `target_quarter` | 213 ± 151 |
| `target_lag52` | 142 ± 297 |
| `target_month_sin` | 111 ± 259 |
| `horizon` | 110 ± 129 |
| `origin_roll13` | 0 ± 0 |

## Data provenance

Tableau/Kaggle *Sample - Superstore*, 9,994 order lines, 2014-01-03 to 2017-12-30, pinned by SHA-256 and by row, order, sales and profit totals in `src/superstore/config.py`.
