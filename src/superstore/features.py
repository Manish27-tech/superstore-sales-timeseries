"""Aggregation and feature construction.

Two jobs live here:

1. **Resampling** the order lines into daily, weekly, monthly and product-level
   tables. Weeks are Monday-starting and the two boundary weeks are dropped so
   every modelled week covers seven calendar days.
2. **Building the direct multi-step panel** used by the forecaster. For an
   origin week ``O`` and horizon ``h`` the target is week ``O + h``; every
   feature is either known at ``O`` or is a calendar value for the target week,
   which is known in advance by definition.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from superstore.config import HORIZON_WEEKS, PROFIT, QUANTITY, TARGET

CALENDAR_FEATURES = (
    "horizon",
    "target_week_of_year_sin",
    "target_week_of_year_cos",
    "target_month_sin",
    "target_month_cos",
    "target_quarter",
    "target_is_q4",
    "target_weeks_since_start",
)
HISTORY_FEATURES = (
    "origin_level",
    "origin_roll4",
    "origin_roll13",
    "target_lag52",
    "target_lag104",
    "origin_yoy_growth",
)

# The forecaster deliberately uses a compact subset. In walk-forward tests the
# wider set let a linear model extrapolate a trend index far outside its
# training range; these eight features are stable across every backtest window.
FORECAST_FEATURES = (
    "origin_roll13",
    "target_lag52",
    "target_week_of_year_sin",
    "target_week_of_year_cos",
    "target_month_sin",
    "target_month_cos",
    "target_quarter",
    "horizon",
)
NUMERIC_FEATURES = HISTORY_FEATURES + CALENDAR_FEATURES


def weekly_revenue(frame: pd.DataFrame, drop_partial: bool = True) -> pd.DataFrame:
    """Aggregate order lines into Monday-starting weekly totals."""
    grouped = (
        frame.groupby("Week Start")
        .agg(
            revenue=(TARGET, "sum"),
            profit=(PROFIT, "sum"),
            units=(QUANTITY, "sum"),
            orders=("Order ID", "nunique"),
            lines=(TARGET, "size"),
        )
        .sort_index()
    )
    grouped["margin"] = grouped["profit"] / grouped["revenue"]
    grouped["aov"] = grouped["revenue"] / grouped["orders"]
    if drop_partial and len(grouped) > 2:
        grouped = grouped.iloc[1:-1]
    grouped.index.name = "week_start"
    return grouped.reset_index()


def monthly_revenue(frame: pd.DataFrame) -> pd.DataFrame:
    """Monthly totals, with month-end timestamps so gaps are visible."""
    grouped = (
        frame.groupby("Month Start")
        .agg(
            revenue=(TARGET, "sum"),
            profit=(PROFIT, "sum"),
            units=(QUANTITY, "sum"),
            orders=("Order ID", "nunique"),
        )
        .sort_index()
    )
    grouped["margin"] = grouped["profit"] / grouped["revenue"]
    grouped.index.name = "month_start"
    return grouped.reset_index()


def daily_revenue(frame: pd.DataFrame) -> pd.DataFrame:
    """Daily totals reindexed onto every calendar day, filling quiet days with 0."""
    grouped = (
        frame.groupby(frame["Order Date"].dt.normalize())
        .agg(revenue=(TARGET, "sum"), orders=("Order ID", "nunique"))
        .sort_index()
    )
    full_range = pd.date_range(grouped.index.min(), grouped.index.max(), freq="D")
    grouped = grouped.reindex(full_range, fill_value=0)
    grouped.index.name = "date"
    return grouped.reset_index()


def monthly_category_revenue(frame: pd.DataFrame) -> pd.DataFrame:
    """Revenue by month and category, for stacked-area views."""
    grouped = (
        frame.groupby(["Month Start", "Category"])[TARGET]
        .sum()
        .unstack(fill_value=0.0)
        .sort_index()
    )
    grouped.index.name = "month_start"
    return grouped.reset_index()


def product_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per product with level, trend and volatility statistics.

    Growth compares the mean monthly revenue of the final four active months
    with the mean of the first four, so a product that only launched in 2016 is
    still scored on its own life rather than against 2014 zeros.
    """
    monthly = (
        frame.groupby(["Product ID", "Month Start"])[TARGET].sum().reset_index()
    )
    rows: list[dict[str, object]] = []
    for product_id, group in monthly.groupby("Product ID"):
        series = group.set_index("Month Start")[TARGET].sort_index()
        lines = frame[frame["Product ID"] == product_id]
        first, last = series.index.min(), series.index.max()
        active_months = len(series)
        span_months = (last.to_period("M") - first.to_period("M")).n + 1
        # Missing months inside the product's life are genuine zero-sale months.
        full = series.reindex(
            pd.period_range(first, last, freq="M").to_timestamp(), fill_value=0.0
        )
        baseline = full.iloc[:4].mean()
        recent = full.iloc[-4:].mean()
        growth = (recent - baseline) / baseline if baseline > 0 else np.nan
        rows.append(
            {
                "product_id": product_id,
                "product_name": lines["Product Name"].iloc[0],
                "category": lines["Category"].iloc[0],
                "sub_category": lines["Sub-Category"].iloc[0],
                "revenue": float(lines[TARGET].sum()),
                "profit": float(lines[PROFIT].sum()),
                "units": int(lines[QUANTITY].sum()),
                "orders": int(lines["Order ID"].nunique()),
                "margin": float(lines[PROFIT].sum() / lines[TARGET].sum()),
                "active_months": active_months,
                "span_months": span_months,
                "first_sale": first,
                "last_sale": last,
                "mean_monthly": float(full.mean()),
                "growth": float(growth) if np.isfinite(growth) else 0.0,
                "volatility": float(full.std(ddof=0) / full.mean()) if full.mean() > 0 else 0.0,
                "share_of_total": float(lines[TARGET].sum() / frame[TARGET].sum()),
            }
        )
    summary = pd.DataFrame(rows).sort_values("revenue", ascending=False).reset_index(drop=True)
    summary["revenue_rank"] = np.arange(1, len(summary) + 1)
    summary["cumulative_share"] = summary["revenue"].cumsum() / summary["revenue"].sum()
    return summary


def abc_classes(product_frame: pd.DataFrame, thresholds=(0.8, 0.95)) -> pd.DataFrame:
    """Classic ABC split on cumulative revenue share."""
    ordered = product_frame.sort_values("revenue", ascending=False).copy()
    cumulative = ordered["revenue"].cumsum() / ordered["revenue"].sum()
    ordered["abc_class"] = np.where(
        cumulative <= thresholds[0],
        "A",
        np.where(cumulative <= thresholds[1], "B", "C"),
    )
    return ordered


def _origin_row(
    revenue: np.ndarray,
    origin_index: int,
    horizon: int,
    start: pd.Timestamp,
) -> dict[str, float] | None:
    """Feature row for forecasting week ``origin_index + horizon``."""
    target_index = origin_index + horizon
    if target_index >= len(revenue) or origin_index < 104:
        return None
    target_date = start + pd.Timedelta(weeks=target_index)
    level = revenue[origin_index]
    roll4 = revenue[origin_index - 3 : origin_index + 1].mean()
    roll13 = revenue[origin_index - 12 : origin_index + 1].mean()
    lag52 = revenue[target_index - 52]
    lag104 = revenue[target_index - 104]
    # Trailing four weeks against the same four weeks a year earlier.
    previous_year = revenue[origin_index - 55 : origin_index - 51].sum()
    current_year = revenue[origin_index - 3 : origin_index + 1].sum()
    return {
        "origin_level": float(np.log1p(level)),
        "origin_roll4": float(np.log1p(roll4)),
        "origin_roll13": float(np.log1p(roll13)),
        "target_lag52": float(np.log1p(lag52)),
        "target_lag104": float(np.log1p(lag104)),
        "origin_yoy_growth": float(
            (current_year - previous_year) / previous_year if previous_year > 0 else 0.0
        ),
        "horizon": float(horizon),
        "target_week_of_year_sin": float(
            np.sin(2 * np.pi * target_date.isocalendar().week / 52.18)
        ),
        "target_week_of_year_cos": float(
            np.cos(2 * np.pi * target_date.isocalendar().week / 52.18)
        ),
        "target_month_sin": float(np.sin(2 * np.pi * (target_date.month - 1) / 12)),
        "target_month_cos": float(np.cos(2 * np.pi * (target_date.month - 1) / 12)),
        "target_quarter": float(target_date.quarter),
        "target_is_q4": float(target_date.quarter == 4),
        "target_weeks_since_start": float(target_index),
    }


def build_origin_panel(
    weekly: pd.DataFrame, origins: list[int], horizon: int = HORIZON_WEEKS
) -> pd.DataFrame:
    """Direct multi-step panel: one row per (origin, horizon) pair.

    Origins are positional indices into ``weekly``. Rows are only kept when the
    target week exists and the origin has at least two years of history, which
    is what the lag-104 feature needs.
    """
    revenue = weekly["revenue"].to_numpy(dtype=float)
    start = pd.Timestamp(weekly["week_start"].iloc[0])
    rows: list[dict[str, float]] = []
    for origin_index in origins:
        for step in range(1, horizon + 1):
            row = _origin_row(revenue, origin_index, step, start)
            if row is None:
                continue
            row["origin_index"] = float(origin_index)
            row["target_index"] = float(origin_index + step)
            row["target_week"] = start + pd.Timedelta(weeks=origin_index + step)
            rows.append(row)
    panel = pd.DataFrame(rows)
    if panel.empty:
        raise ValueError("no panel rows were produced; check the origin indices")
    panel["target_week"] = pd.to_datetime(panel["target_week"])
    panel["y"] = panel["target_index"].astype(int).map(lambda i: revenue[i])
    return panel


def feature_matrix(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a panel into the design matrix and the target."""
    X = panel[list(FORECAST_FEATURES)].astype("float32")
    y = panel["y"].astype("float64")
    return X, y


def training_origins(
    weekly: pd.DataFrame, horizon: int = HORIZON_WEEKS, step: int = 4
) -> list[int]:
    """Origin indices spaced ``step`` weeks apart, all with a full horizon."""
    last_origin = len(weekly) - horizon - 1
    return list(range(104, last_origin + 1, step))


def seasonal_naive(weekly: pd.DataFrame, origin: int, horizon: int = HORIZON_WEEKS) -> np.ndarray:
    """The same week one year earlier, back on the revenue scale."""
    revenue = weekly["revenue"].to_numpy(dtype=float)
    values = []
    for step in range(1, horizon + 1):
        index = origin + step - 52
        values.append(revenue[index] if index >= 0 else revenue[origin])
    return np.asarray(values, dtype=float)
