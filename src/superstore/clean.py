"""Cleaning and calendar derivation for the order-line extract.

The published extract is unusually tidy -- no missing values, no duplicate row
ids -- so most of the work here is making the time dimension explicit and
auditing the parts that are easy to get wrong: returns (negative profit), the
handful of order ids that carry the wrong year, and the 2017 file having no
2018 orders despite shipping into January.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from superstore.config import DATA_PROCESSED, PROFIT, QUANTITY, TARGET

ORDERS_PATH = DATA_PROCESSED / "orders.csv.gz"

DISCOUNT_BANDS = (-0.01, 0.0, 0.1, 0.2, 0.4, 1.0)
DISCOUNT_LABELS = ("No discount", "1-10%", "11-20%", "21-40%", "41%+")


def _discount_band(discount: pd.Series) -> pd.Series:
    return pd.cut(
        discount,
        bins=DISCOUNT_BANDS,
        labels=DISCOUNT_LABELS,
        include_lowest=True,
    ).astype(str)


def clean_orders(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return the analysis frame plus an audit record of what changed."""
    working = frame.copy()

    duplicate_rows = int(working["Row ID"].duplicated().sum())
    working = working.drop_duplicates(subset=["Row ID"])

    if (working[TARGET] <= 0).any():
        raise ValueError("non-positive sales values are not expected in this extract")
    if working[QUANTITY].lt(0).any():
        raise ValueError("negative quantity found")

    # Calendar fields, all derived from Order Date so they cannot disagree with
    # the timestamp the way a hand-maintained column can.
    order_date = working["Order Date"]
    working["Order Year"] = order_date.dt.year
    working["Order Month"] = order_date.dt.month
    working["Order Quarter"] = order_date.dt.quarter
    working["Order Weekday"] = order_date.dt.day_name()
    working["Weekday Index"] = order_date.dt.dayofweek
    working["Week Start"] = order_date - pd.to_timedelta(order_date.dt.dayofweek, unit="D")
    working["Month Start"] = order_date.dt.to_period("M").dt.to_timestamp()
    working["Year Month"] = order_date.dt.to_period("M").astype(str)
    working["Day of Year"] = order_date.dt.dayofyear

    # Money and logistics fields.
    working["Ship Lag Days"] = (working["Ship Date"] - order_date).dt.days
    working["Margin"] = np.where(
        working[TARGET] > 0, working[PROFIT] / working[TARGET], np.nan
    )
    working["Unit Price"] = working[TARGET] / working[QUANTITY]
    working["Is Return"] = (working[PROFIT] < 0).astype(int)
    working["Has Discount"] = (working["Discount"] > 0).astype(int)
    working["Discount Band"] = _discount_band(working["Discount"])

    # Order ids embed the year they were opened. A handful (7 lines) are dated
    # 31 December of the previous year, which is a source quirk, not a parse bug.
    id_year = working["Order ID"].str.extract(r"-(\d{4})-")[0].astype(int)
    working["Id Year Mismatch"] = (id_year != working["Order Year"]).astype(int)

    # The weekly series starts on the first Monday and the boundary weeks are
    # dropped so every modelled week covers seven calendar days.
    first_week = working["Week Start"].min()
    last_week = working["Week Start"].max()
    partial_weeks = int(first_week < order_date.min()) + int(
        last_week + pd.Timedelta(days=6) > order_date.max()
    )

    report = {
        "rows_in": int(len(frame)),
        "rows_out": int(len(working)),
        "duplicate_rows_removed": duplicate_rows,
        "date_min": str(order_date.min()),
        "date_max": str(order_date.max()),
        "weeks": int(working["Week Start"].nunique()),
        "months": int(working["Month Start"].nunique()),
        "partial_weeks": partial_weeks,
        "median_ship_lag_days": float(working["Ship Lag Days"].median()),
        "share_discounted": round(float(working["Has Discount"].mean()), 4),
        "share_return_lines": round(float(working["Is Return"].mean()), 4),
        "share_return_value": round(
            float(working.loc[working["Is Return"] == 1, TARGET].sum() / working[TARGET].sum()),
            4,
        ),
        "order_id_year_mismatches": int(working["Id Year Mismatch"].sum()),
        "median_margin": round(float(working["Margin"].median()), 4),
        "total_sales": round(float(working[TARGET].sum()), 2),
        "total_profit": round(float(working[PROFIT].sum()), 2),
    }
    return working.reset_index(drop=True), report


def save_orders(frame: pd.DataFrame, path: Path = ORDERS_PATH) -> Path:
    """Persist the cleaned frame as a compressed CSV for later stages."""
    frame.to_csv(path, index=False, date_format="%Y-%m-%d")
    return path


def load_orders(path=ORDERS_PATH) -> pd.DataFrame:
    """Read the cleaned frame back, keeping the date columns as datetimes."""
    frame = pd.read_csv(path, parse_dates=["Order Date", "Ship Date", "Week Start", "Month Start"])
    return frame
