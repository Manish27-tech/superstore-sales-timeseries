# Methodology

This document explains *why* the pipeline is shaped the way it is, so a reader
who disagrees with a choice can find the alternative that was tried and rejected.

## 1. Data and verification

- The extract is the Tableau/Kaggle *Sample - Superstore* CSV, pinned by
  SHA-256 and by row, order, sales and profit totals in `src/superstore/config.py`.
  `data.verify` refuses to continue if either changes.
- Dates are parsed as `dd/mm/yy` (`%d/%m/%y`). The wrong parse produces shipping
  lags with a median of ~122 days; the correct one gives 0-7 days with a median
  of 4. `parse_dates` records that proof in `data.describe`.
- Two boundary weeks are incomplete and are dropped from every modelled series;
  calendar months are never trimmed.

## 2. Aggregation

- Weeks are Monday-starting. This keeps the weekly series aligned with how the
  business actually plans (Monday morning stand-ups, weekly replenishment).
- Months are calendar months with month-end timestamps so gaps are visible.
- Days are zero-filled: a day with no sale is a real zero, not a missing value.

## 3. Decomposition

- Multiplicative, because the seasonal amplitude grows with the level.
- Trend is a centred 12-month moving average; the first and last six months
  have no centred trend and are excluded from the seasonal index rather than
  silently distorted.
- Seasonal factors are month averages of the detrended series.
- Trend and seasonal *strength* follow the usual variance-ratio definition.

## 4. Anomalies

- Robust z-score of `revenue / same-week-last-year-revenue`, against the
  median and MAD of that ratio.
- Two business guards: a baseline of at least \$2,000 and an absolute gap of at
  least \$3,000. Without them a quiet week with a \$300 baseline produces a
  dramatic but meaningless ratio.
- Dips are deliberately not flagged: the guards remove low-baseline weeks where
  a dip is indistinguishable from noise.

## 5. Forecasting

- **Direct multi-step**, not recursive: one row per (origin, horizon) pair, with
  features known at the origin. A recursive lag-1 setup would leak the answer.
- Features are a compact eight: 13-week rolling level, same-week-last-year
  revenue, week-of-year and month harmonics, quarter, and horizon. A wider set
  let a linear model extrapolate a trend index far outside its training range.
- Model selection uses mean walk-forward RMSE *inside* the training period,
  never the hold-out. Selection and evaluation windows are disjoint.
- Predictions are clipped to 0.4-2.5x the same week last year, which is the
  documented guard against extrapolation blow-ups.
- Intervals are empirical 5/95% residual quantiles per horizon from the
  walk-forward folds, clipped so the point forecast always lies inside its band.

## 6. What was tried and rejected

- A trend index feature: improved in-sample fit, destroyed out-of-sample fit.
- Gradient boosting as the reported model: it scores better on this window but
  not in walk-forward selection; with four folds the gap is noise.
- Recursive one-step forecasting: leaked the target through lag-1 features.
- Monthly aggregation for the forecast: more stable, but the project's stated
  horizon is weekly; the weekly model is the one reported, with the limitation
  documented rather than hidden.
