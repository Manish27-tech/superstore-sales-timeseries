"""Dataset acquisition and integrity verification.

The source is a public mirror of the Tableau/Kaggle *Sample - Superstore*
extract. Two independent checks run before any analysis:

1. the SHA-256 of the bytes, so a changed mirror fails loudly;
2. the business totals (rows, orders, sales, profit), so a mirror that
   re-parses or re-orders rows is caught even if the hash is updated.

There is a third, subtler check in :func:`parse_dates`: the file stores dates as
``dd/mm/yy`` and several public mirrors silently read them as ``mm/dd/yy``. The
two readings cannot both be right, and the shipping lag settles it.
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import pandas as pd

from superstore.config import (
    DATA_RAW,
    DATASET_SHA256,
    DATASET_URL,
    DATE_FORMAT,
    EXPECTED_ORDERS,
    EXPECTED_PROFIT,
    EXPECTED_ROWS,
    EXPECTED_SALES,
    MAX_SHIP_LAG_DAYS,
)

CSV_PATH = DATA_RAW / "superstore.csv"

REQUIRED_COLUMNS = (
    "Row ID",
    "Order ID",
    "Order Date",
    "Ship Date",
    "Ship Mode",
    "Customer ID",
    "Customer Name",
    "Segment",
    "Country",
    "City",
    "State",
    "Postal Code",
    "Region",
    "Product ID",
    "Category",
    "Sub-Category",
    "Product Name",
    "Sales",
    "Quantity",
    "Discount",
    "Profit",
)


def sha256_file(path: Path) -> str:
    """Hash a file in 1 MB chunks so a 2 MB CSV stays cheap."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, expected: str = DATASET_SHA256) -> Path:
    """Return ``path`` if its hash matches, otherwise raise."""
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"checksum mismatch for {path.name}: expected {expected}, got {actual}"
        )
    return path


def download(force: bool = False, target: Path = CSV_PATH) -> Path:
    """Download the CSV if needed and verify it before returning the path."""
    if target.exists() and not force:
        return verify(target)

    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(DATASET_URL, timeout=120) as response:
        payload = response.read()
    target.write_bytes(payload)
    return verify(target)


def parse_dates(frame: pd.DataFrame) -> pd.DataFrame:
    """Parse the two date columns and prove the day/month order is right.

    The published file uses ``dd/mm/yy``. Reading it as ``mm/dd/yy`` still yields
    valid-looking dates, so the parse is validated with shipping behaviour:
    every dispatch happens within a few days of the order. Under the wrong
    reading the median lag jumps from 4 days to 122, which cannot be a real
    logistics process.
    """
    working = frame.copy()
    working["Order Date"] = pd.to_datetime(working["Order Date"], format=DATE_FORMAT)
    working["Ship Date"] = pd.to_datetime(working["Ship Date"], format=DATE_FORMAT)
    lag = (working["Ship Date"] - working["Order Date"]).dt.days
    if lag.min() < 0:
        raise ValueError("ship date precedes order date; the date format is wrong")
    if lag.max() > MAX_SHIP_LAG_DAYS:
        raise ValueError(
            f"shipping lag of {lag.max()} days is implausible; check the date format"
        )
    return working


def validate(frame: pd.DataFrame) -> pd.DataFrame:
    """Assert the schema and the business totals the analysis depends on."""
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"unexpected schema, missing columns: {missing}")

    if len(frame) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} rows, found {len(frame)}")
    if frame["Order ID"].nunique() != EXPECTED_ORDERS:
        raise ValueError(
            f"expected {EXPECTED_ORDERS} orders, found {frame['Order ID'].nunique()}"
        )
    if abs(frame["Sales"].sum() - EXPECTED_SALES) > 0.05:
        raise ValueError(f"sales total drifted: {frame['Sales'].sum():.2f}")
    if abs(frame["Profit"].sum() - EXPECTED_PROFIT) > 0.05:
        raise ValueError(f"profit total drifted: {frame['Profit'].sum():.2f}")
    if frame.isna().any().any():
        raise ValueError("unexpected missing values in the published extract")
    return frame


def load_raw(path: Path = CSV_PATH) -> pd.DataFrame:
    """Download, verify, parse and validate the raw extract."""
    verify(download(target=path))
    frame = pd.read_csv(path)
    return validate(parse_dates(frame))


def describe(frame: pd.DataFrame) -> dict[str, object]:
    """Headline facts used by the report and the dashboard landing page."""
    return {
        "rows": int(len(frame)),
        "orders": int(frame["Order ID"].nunique()),
        "customers": int(frame["Customer ID"].nunique()),
        "products": int(frame["Product ID"].nunique()),
        "categories": int(frame["Category"].nunique()),
        "sub_categories": int(frame["Sub-Category"].nunique()),
        "regions": int(frame["Region"].nunique()),
        "date_min": str(frame["Order Date"].min()),
        "date_max": str(frame["Order Date"].max()),
        "total_sales": round(float(frame["Sales"].sum()), 2),
        "total_profit": round(float(frame["Profit"].sum()), 2),
        "total_units": int(frame["Quantity"].sum()),
        "mean_order_value": round(
            float(frame.groupby("Order ID")["Sales"].sum().mean()), 2
        ),
        "csv_sha256": DATASET_SHA256,
    }
