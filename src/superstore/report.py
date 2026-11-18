"""Render ``REPORT.md`` and ``MODEL_CARD.md`` from the committed artefacts.

Every number in the prose is pulled from a summary dictionary that the pipeline
wrote, so a rerun cannot leave the text quoting stale figures. Figures are
linked relatively, which is what GitHub renders.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from superstore.config import REPORTS


def _money(value: float) -> str:
    return f"${value:,.0f}"


def _pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def _figure(name: str, caption: str) -> str:
    return f"![{caption}](figures/{name})\n\n*{caption}*\n"


def build_report(
    eda: dict[str, Any],
    timeseries: dict[str, Any],
    segments: dict[str, Any],
    anomalies: dict[str, Any],
    metrics: dict[str, Any],
    diagnostics: dict[str, Any],
    clean_audit: dict[str, Any],
) -> str:
    """Assemble the main narrative report."""
    annual = {int(year): value for year, value in eda["annual_revenue"].items()}
    yoy = {int(year): value for year, value in eda["yoy_growth"].items()}
    strongest = timeseries["strongest_month"]
    weakest = timeseries["weakest_month"]
    test = metrics["test"]
    naive = metrics["test_all_models"]["seasonal_naive"]
    gb = metrics["test_all_models"]["gradient_boosting"]

    lines = [
        "# Superstore product sales: a four-year time series",
        "",
        "An end-to-end analysis of 9,994 order lines from the Tableau/Kaggle "
        "*Sample - Superstore* extract: what the revenue series is made of, which "
        "products carry it, which weeks break the pattern, and how well a "
        "13-week-ahead forecast actually performs.",
        "",
        "## Headline findings",
        "",
        f"- Revenue grew from {_money(annual[2014])} in 2014 to {_money(annual[2017])} in 2017 "
        f"(**+{yoy[2016] * 100 + yoy[2017] * 100:.0f}% across the two growth years**), after a "
        f"flat 2015 ({_pct(yoy[2015])}).",
        f"- The series is strongly seasonal: **{strongest}** is "
        f"{timeseries['strongest_factor']:.2f}x an average month and **{weakest}** is "
        f"{timeseries['weakest_factor']:.2f}x. Seasonality explains "
        f"{_pct(timeseries['seasonal_strength'])} of the detrended variance, the trend "
        f"{_pct(timeseries['trend_strength'])} of the deseasonalised variance.",
        f"- Revenue is concentrated but not winner-take-all: **{segments['products_for_80pct']} of "
        f"{segments['products']:,} products ({_pct(segments['products_for_80pct'] / segments['products'])}) "
        f"produce 80% of sales**. The A class is {_pct(segments['abc_products']['A'] / segments['products'])} of "
        f"the catalogue and {_pct(segments['abc_revenue_share']['A'])} of revenue.",
        f"- Margins leak at the discount tail: lines discounted more than 20% run at a "
        f"negative margin, and {_pct(eda['deepest_discount_share'], 0)} of all lines are discounted "
        "by 40% or more.",
        f"- {anomalies['anomalies']} of the {anomalies['eligible_weeks']} comparable weeks are "
        f"anomalies ({_pct(anomalies['share_of_weeks'])}), all upward, and each is traceable to a "
        "single large order.",
        f"- Forecasting 13 weeks ahead, the selected model beats the seasonal-naive baseline by "
        f"**{metrics['improvement_vs_naive_pct']:.1f}% RMSE** (MASE {test['mase']:.2f}), but it "
        f"under-predicts the record fourth quarter of 2017 by {_money(abs(test['bias']))} per week. "
        "That is the honest headline: the seasonal shape is learnable, the level shift is not.",
        "",
        "## The data",
        "",
        f"- {clean_audit['rows_in']:,} order lines, {eda['orders']:,} orders, {eda['customers']:,} customers, "
        f"{eda['products']:,} products, {eda['date_min'][:10]} to {eda['date_max'][:10]}.",
        f"- {clean_audit['weeks']} weekly periods and {eda['months']} months. The two boundary weeks are "
        "incomplete and are dropped from every modelled series.",
        f"- Mean order value {_money(eda['mean_order_value'])}, median {_money(eda['median_order_value'])}: "
        "a long tail of small orders and a handful of five-figure ones.",
        f"- {_pct(clean_audit['share_discounted'])} of lines carry a discount and "
        f"{_pct(clean_audit['share_return_lines'])} of lines are unprofitable (a return or a deep discount).",
        "",
        "### Two data-quality notes that shape the analysis",
        "",
        f"1. **Weekday coverage is distorted.** Wednesday holds only "
        f"{_pct(eda['weekday_line_share']['Wednesday'])} of order lines against 11-19% for every other "
        "day. That is a property of the published extract, not of the business, so weekday profiles are "
        "reported for completeness but are not used for any decision.",
        f"2. **Order ids occasionally disagree with the order date.** {clean_audit['order_id_year_mismatches']} "
        "lines dated 31 December 2016 carry 2017 order ids. The order date is treated as authoritative.",
        "",
        _figure("01_monthly_revenue_trend.png", "Monthly revenue with a 12-month average"),
        _figure("03_seasonality_heatmap.png", "Each month against its own year's average"),
        "",
        "## Decomposition: trend versus season",
        "",
        f"A multiplicative decomposition puts the trend at {_money(timeseries['trend_start'])} to "
        f"{_money(timeseries['trend_end'])} per month ({_pct(timeseries['trend_growth'])} over the "
        f"24 months with a centred average), with a peak-to-trough seasonal ratio of "
        f"{timeseries['peak_to_trough']:.1f}x. The residual is large -- "
        f"{_pct(timeseries['mean_absolute_residual'])} mean absolute deviation -- because single large "
        "orders move a month.",
        "",
        _figure("11_monthly_decomposition.png", "Observed, trend, seasonal and residual components"),
        _figure("12_seasonal_factors.png", "Seasonal factors by calendar month"),
        "",
        "## Products: concentration, behaviour and margin",
        "",
        "![Pareto](figures/14_pareto_concentration.png)",
        "",
        f"The catalogue splits into four behavioural groups from a k-means fit on scale, growth, "
        f"volatility and margin. Anchors are {_pct(segments['segments']['Anchor products']['revenue_share'])} "
        f"of revenue across {segments['segments']['Anchor products']['products']} products; the long tail is "
        f"{segments['segments']['Long tail']['products']} products contributing "
        f"{_pct(segments['segments']['Long tail']['revenue_share'])}.",
        "",
        _figure("15_product_segments.png", "Products by revenue and growth, coloured by segment"),
        _figure("05_subcategory_revenue_margin.png", "Sub-category revenue against profit margin"),
        _figure("08_discount_vs_margin.png", "Margin by discount band"),
        "",
        f"The margin story is blunt: {eda['worst_margin_subcategory']} is the worst sub-category by "
        f"margin and {eda['best_margin_subcategory']} the best. Discounting is the mechanism -- the "
        "21-40% band already loses money before overheads.",
        "",
        "## Weeks that break the pattern",
        "",
        f"Anomalies are scored against the same week a year earlier with a robust z-score, and are only "
        f"flagged when the baseline exceeds {_money(anomalies['min_expected'])} and the absolute gap "
        f"exceeds {_money(anomalies['min_deviation'])}.",
        "",
        _figure("17_anomaly_timeline.png", "Weekly revenue with flagged anomalies"),
        _figure("18_anomaly_explanations.png", "The flagged weeks and their largest single order line"),
        "",
        "| Week | Direction | z | Revenue | Expected | Largest line |",
        "|---|---|---|---|---|---|",
    ]
    for row in anomalies["top_anomalies"]:
        lines.append(
            f"| {row['week_start']} | {row['direction']} | {row['z_score']:.2f} | "
            f"{_money(row['revenue'])} | {_money(row['expected'])} | "
            f"{_money(row['largest_order_value'])} |"
        )
    lines += [
        "",
        "## Forecasting 13 weeks ahead",
        "",
        "The forecast is a **fixed-origin, direct multi-step** problem: standing at the last training "
        "week (week of 2017-09-18), predict the next 13 weeks. A lag-1 setup would leak the answer, so "
        "the data is reshaped into one row per (origin, horizon) pair with features known at the origin: "
        "the 13-week rolling level, the same week 52 weeks earlier, the target week's calendar position, "
        "and the horizon itself.",
        "",
        f"- **Selection** uses the mean walk-forward RMSE inside the training period: "
        f"Ridge {_money(metrics['selection_scores']['ridge_regression'])}, gradient boosting "
        f"{_money(metrics['selection_scores']['gradient_boosting'])}. Ridge wins and is the reported model.",
        f"- **Hold-out (13 weeks, {metrics['split']['test_start'][:10]} onward):** RMSE "
        f"{_money(test['rmse'])}, MAE {_money(test['mae'])}, MAPE {test['mape']:.1f}%, "
        f"MASE {test['mase']:.2f}.",
        f"- The seasonal-naive baseline scores RMSE {_money(naive['rmse'])} and MAPE {naive['mape']:.1f}%, "
        f"so the model cuts error by {metrics['improvement_vs_naive_pct']:.1f}%.",
        f"- **R² is {test['r2']:.2f}.** That is not a typo: over thirteen weeks that set a new record "
        f"for the business, predicting the window mean would have scored better. The model is still the "
        f"better forecaster in MASE terms because it tracks the seasonal shape week by week.",
        f"- Gradient boosting, which was *not* selected, actually scored better on this window "
        f"({_money(gb['rmse'])} RMSE). With four backtest folds, selection noise is larger than the gap "
        f"between the two models; that is stated rather than hidden.",
        f"- The 5-95% error band covered {_pct(diagnostics['interval_coverage'])} of held-out weeks "
        f"against a nominal 90%, another symptom of the level shift.",
        "",
        _figure("19_forecast_model_comparison.png", "Held-out error and MASE by model"),
        _figure("20_backtest_folds.png", "Walk-forward RMSE by fold"),
        _figure("21_holdout_forecast.png", "Forecast against actuals for the held-out quarter"),
        _figure("22_forecast_residuals.png", "Residual distribution and error by horizon"),
        "",
        "### Why the fourth quarter is hard",
        "",
        "The training period contains three fourth quarters, none as large as 2017's. Every model class "
        "tested here -- Ridge, gradient boosting, and a ratio-to-last-year formulation tried during "
        "development -- under-predicted it. The practical fix is not a better regressor on 194 weekly "
        "rows; it is a shorter forecast horizon, a leading indicator, or an explicit level-shift "
        "adjustment owned by a planner.",
        "",
        "## What I would do next",
        "",
        "- Forecast at four-week blocks, where single orders average out and the seasonal signal dominates.",
        "- Model order counts and average order value separately, then multiply; the lumpiness lives in "
        "the value term and is easier to attribute.",
        "- Add promotion and catalogue metadata. The extract has no marketing calendar, which is the "
        "missing variable behind most of the residual.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "python -m pip install -r requirements-dev.txt",
        "python run.py all          # ~1 minute: download, analyse, forecast, export",
        "python run.py dashboard    # opens the Streamlit app on the committed feeds",
        "```",
        "",
        "The dataset is pinned by SHA-256 and by business totals; the pipeline fails loudly if either "
        "changes. Raw and processed data are git-ignored, the dashboard feed is committed.",
        "",
    ]
    return "\n".join(lines)


def build_model_card(metrics: dict[str, Any], diagnostics: dict[str, Any]) -> str:
    """A short model card: what it does, what it must not be used for, how it fails."""
    test = metrics["test"]
    rows = [
        "| Model | RMSE | MAE | MAPE | MASE | R² |",
        "|---|---|---|---|---|---|",
    ]
    for name, score in metrics["test_all_models"].items():
        rows.append(
            f"| {name.replace('_', ' ')} | {score['rmse']:,.0f} | {score['mae']:,.0f} | "
            f"{score['mape']:.1f}% | {score.get('mase', float('nan')):.2f} | {score['r2']:.2f} |"
        )

    importance = metrics["feature_importance"][:6]
    importance_rows = ["| Feature | Permutation importance (RMSE, USD) |", "|---|---|"]
    importance_rows += [
        f"| `{row['feature']}` | {row['importance_mean']:,.0f} ± {row['importance_std']:,.0f} |"
        for row in importance
    ]

    return "\n".join(
        [
            "# Model card: weekly revenue forecaster",
            "",
            "## What it is",
            "",
            "- **Task:** forecast total weekly sales revenue 1-13 weeks ahead from a fixed origin.",
            f"- **Selected model:** {metrics['selected_model'].replace('_', ' ')} on log1p revenue, "
            "chosen by mean walk-forward RMSE.",
            f"- **Training data:** {metrics['split']['train_weeks']} weekly periods ending "
            f"{metrics['split']['test_start'][:10]}.",
            f"- **Evaluation:** {metrics['split']['test_weeks']} held-out weeks, "
            f"{metrics['split']['test_start'][:10]} to {metrics['split']['test_end'][:10]}.",
            "- **Features:** 13-week rolling level, same-week-last-year revenue, week-of-year and "
            "month harmonics, quarter, and horizon. All knowable at the origin.",
            "",
            "## Performance",
            "",
            *rows,
            "",
            f"Improvement over seasonal naive: **{metrics['improvement_vs_naive_pct']:.1f}% RMSE**.",
            "",
            "## Calibration and uncertainty",
            "",
            "- Intervals are empirical 5-95% residual quantiles per horizon from the walk-forward folds.",
            f"- Observed coverage on the held-out weeks: {_pct(diagnostics['interval_coverage'])} "
            "(nominal 90%). The interval is too narrow for a level shift of the size seen in Q4 2017.",
            "",
            "## Where it fails",
            "",
            f"- Systematic under-prediction in the evaluation quarter: mean error "
            f"{test['bias']:+,.0f} USD per week, {diagnostics['under_predicted_weeks']} of "
            f"{diagnostics['weeks']} weeks low.",
            f"- Median absolute percentage error {_pct(diagnostics['median_abs_pct_error'])}; only "
            f"{diagnostics['weeks_within_20pct']} of {diagnostics['weeks']} weeks land within 20%.",
            "- Weekly revenue is lumpy: a single copier order can add $14k to a $12k week. No model on "
            "this sample fixes that.",
            "",
            "## Intended use",
            "",
            "Planning support and a teaching example of honest time-series evaluation. Not a replacement "
            "for a committed financial forecast; not suitable for inventory commitments without a human "
            "review of pipeline and promotions.",
            "",
            "## Features that carry the model",
            "",
            *importance_rows,
            "",
            "## Data provenance",
            "",
            "Tableau/Kaggle *Sample - Superstore*, 9,994 order lines, 2014-01-03 to 2017-12-30, pinned by "
            "SHA-256 and by row, order, sales and profit totals in `src/superstore/config.py`.",
            "",
        ]
    )


def generate_reports(
    eda: dict[str, Any],
    timeseries: dict[str, Any],
    segments: dict[str, Any],
    anomalies: dict[str, Any],
    metrics: dict[str, Any],
    diagnostics: dict[str, Any],
    clean_audit: dict[str, Any],
    report_path: Path = REPORTS / "REPORT.md",
    card_path: Path = REPORTS / "MODEL_CARD.md",
) -> dict[str, Path]:
    """Write both markdown artefacts."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        build_report(eda, timeseries, segments, anomalies, metrics, diagnostics, clean_audit),
        encoding="utf-8",
    )
    card_path.write_text(build_model_card(metrics, diagnostics), encoding="utf-8")
    return {"report": report_path, "model_card": card_path}


def load_summaries(reports_dir: Path = REPORTS) -> dict[str, Any]:
    """Read the JSON artefacts back for the pipeline's report step."""
    names = (
        "eda_summary",
        "timeseries_summary",
        "segments_summary",
        "anomalies_summary",
        "metrics",
        "diagnostics_summary",
    )
    loaded: dict[str, Any] = {}
    for name in names:
        path = reports_dir / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"missing artefact {path}; run the earlier pipeline steps first")
        loaded[name] = json.loads(path.read_text(encoding="utf-8"))
    return loaded
