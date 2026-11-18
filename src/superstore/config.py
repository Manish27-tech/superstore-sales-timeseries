"""Single source of truth for paths, dataset pins and modelling constants."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_DASHBOARD = ROOT / "data" / "dashboard"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

for _path in (DATA_RAW, DATA_PROCESSED, DATA_DASHBOARD, ARTIFACTS, FIGURES):
    _path.mkdir(parents=True, exist_ok=True)

# The source is the Tableau/Kaggle "Sample - Superstore" extract, mirrored as a
# gist and pinned twice: the byte-level SHA-256 and a set of business totals
# that would catch a silently reparsed date column.
DATASET_URL = (
    "https://gist.githubusercontent.com/SharmaAshwini/"
    "8bd642f6c46792a9c40c8ccad60391e9/raw/Sample_Superstore.csv"
)
DATASET_SHA256 = "a798cff935a7ffa8032b793d6116eb87f6cb9e371b0f286267d9f10abb0ba00a"
EXPECTED_ROWS = 9_994
EXPECTED_ORDERS = 5_009
EXPECTED_SALES = 2_297_200.86
EXPECTED_PROFIT = 286_397.02

# The source stores dates as dd/mm/yy. This is verified in `data.py` against the
# shipping lag: the wrong (mm/dd/yy) reading produces a median lag of 122 days.
DATE_FORMAT = "%d/%m/%y"
MAX_SHIP_LAG_DAYS = 10

TARGET = "Sales"
PROFIT = "Profit"
QUANTITY = "Quantity"

# Weekly series: Monday-starting weeks. The last 13 complete weeks are held out,
# starting with the week of 2017-09-25 (which contains the Black Friday run-up).
TEST_START = "2017-09-25"
HORIZON_WEEKS = 13
BACKTEST_FOLDS = 6

RANDOM_STATE = 42

PALETTE = {
    "actual": "#2B5D8C",
    "predicted": "#E4572E",
    "interval": "#E4572E",
    "neutral": "#9AA7B8",
    "accent": "#C44536",
}

CATEGORY_COLORS = {
    "Furniture": "#4C72B0",
    "Office Supplies": "#55A868",
    "Technology": "#C44E52",
}

FIGURE_DPI = 140
