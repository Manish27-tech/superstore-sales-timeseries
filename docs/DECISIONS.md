# Decisions

A short log of the choices a reviewer is most likely to question.

1. **Dataset.** Tableau/Kaggle Sample - Superstore, pinned by SHA-256 and totals.
   Rejected alternatives: Online Retail II and bike sharing (already used in
   earlier projects), and any dataset without a date column.

2. **Date format `%d/%m/%y`.** Proven by shipping lag. The alternative parse
   gives a median lag of 122 days, which is impossible for a retail operation.

3. **Weekly Monday-start series, two boundary weeks dropped.** Keeps every
   modelled week at seven days. Monthly series are never trimmed.

4. **Direct multi-step panel.** Recursive lag-1 forecasting leaks the target.
   The panel has one row per (origin, horizon) pair.

5. **Compact feature set (eight features).** A wider set let a linear model
   extrapolate a trend index far outside its training range. The eight that
   remain are stable across every backtest window.

6. **Ridge is the reported model.** Selected by mean walk-forward RMSE
   (8,040 vs 8,581 for gradient boosting). Gradient boosting scores better on
   the hold-out window, but with four folds that gap is selection noise, and it
   is stated as such rather than hidden.

7. **Predictions clipped to 0.4-2.5x the same week last year.** Documented
   guard against extrapolation blow-ups; without it the linear model produces
   negative forecasts in some windows.

8. **Intervals are empirical per-horizon quantiles, clipped to contain the
   point forecast.** Nominal 90%, observed 61.5% on the hold-out, which is the
   honest finding: the interval is too narrow for a level shift of this size.

9. **Dashboard reads only committed feeds.** Raw data is git-ignored and
   downloaded by the pipeline; Streamlit Community Cloud cannot run the
   download step reliably, so every byte the app needs is committed.

10. **No weekday-based recommendations.** Wednesday holds 3.7% of order lines
    in the published extract, which is a property of the data, not the business.
    Weekday profiles are shown for completeness and flagged as unreliable.
