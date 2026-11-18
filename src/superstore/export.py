"""Publish the compact tables the Streamlit dashboard reads.

The dashboard runs on Streamlit Community Cloud, where the raw extract is not
available (it is git-ignored and downloaded by the pipeline). Everything the app
needs is therefore written here as small, committed CSV/JSON files: a few
thousand rows and well under a megabyte in total.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from superstore.config import DATA_DASHBOARD
from superstore.features import (
    daily_revenue,
    monthly_category_revenue,
    monthly_revenue,
    weekly_revenue,
)

PRODUCT_COLUMNS = (
    "product_id",
    "product_name",
    "category",
    "sub_category",
    "revenue",
    "profit",
    "units",
    "orders",
    "margin",
    "active_months",
    "first_sale",
    "last_sale",
    "mean_monthly",
    "growth",
    "volatility",
    "abc_class",
    "segment",
    "revenue_rank",
    "cumulative_share",
)


def weekday_profile(orders: pd.DataFrame) -> pd.DataFrame:
    """Revenue and volume by weekday, in calendar order."""
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    profile = (
        orders.groupby("Order Weekday")
        .agg(revenue=("Sales", "sum"), orders=("Order ID", "nunique"), lines=("Sales", "size"))
        .reindex(order)
        .reset_index()
    )
    profile.columns = ["weekday", "revenue", "orders", "lines"]
    return profile


def export_tables(
    orders: pd.DataFrame,
    segmented: pd.DataFrame,
    forecast: pd.DataFrame,
    anomalies: pd.DataFrame,
    kpis: dict[str, Any],
    metrics: dict[str, Any] | None = None,
    output_dir: Path = DATA_DASHBOARD,
) -> dict[str, Path]:
    """Write every dashboard feed and return the paths that were written."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    def write_csv(name: str, frame: pd.DataFrame) -> None:
        path = output_dir / name
        frame.to_csv(path, index=False, date_format="%Y-%m-%d")
        paths[name] = path

    write_csv("daily.csv", daily_revenue(orders))
    write_csv("weekly.csv", weekly_revenue(orders))
    write_csv("monthly.csv", monthly_revenue(orders))
    write_csv("monthly_category.csv", monthly_category_revenue(orders))
    write_csv("weekday.csv", weekday_profile(orders))
    write_csv("products.csv", segmented[list(PRODUCT_COLUMNS)])
    write_csv("forecast.csv", forecast)
    write_csv("anomalies.csv", anomalies)
    if metrics is not None and "backtest" in metrics:
        write_csv("backtest.csv", pd.DataFrame(metrics["backtest"]))

    dimensions = {
        "categories": sorted(orders["Category"].unique().tolist()),
        "sub_categories": sorted(orders["Sub-Category"].unique().tolist()),
        "regions": sorted(orders["Region"].unique().tolist()),
        "segments": sorted(orders["Segment"].unique().tolist()),
        "ship_modes": sorted(orders["Ship Mode"].unique().tolist()),
        "years": sorted(int(year) for year in orders["Order Year"].unique()),
        "date_min": str(orders["Order Date"].min().date()),
        "date_max": str(orders["Order Date"].max().date()),
    }
    dimension_path = output_dir / "dimensions.json"
    dimension_path.write_text(json.dumps(dimensions, indent=2) + "\n", encoding="utf-8")
    paths["dimensions.json"] = dimension_path

    kpi_path = output_dir / "kpis.json"
    kpi_path.write_text(json.dumps(kpis, indent=2, default=str) + "\n", encoding="utf-8")
    paths["kpis.json"] = kpi_path
    return paths
