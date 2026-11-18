"""Shared styling, formatting and data loading for the dashboard.

The app is deliberately split into pure helpers here and view modules in
``dashboard/views`` so the logic can be tested without a Streamlit runtime.
Everything is read from the committed ``data/dashboard`` feed; the raw extract is
never needed at runtime, which is what makes the Streamlit Cloud deployment a
one-file install.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "dashboard"

NAVY = "#2B5D8C"
ORANGE = "#E4572E"
GREEN = "#55A868"
GOLD = "#DDB75D"
GREY = "#9AA7B8"
PURPLE = "#7B4B94"
RED = "#A4243B"

CATEGORY_COLORS = {
    "Furniture": "#4C72B0",
    "Office Supplies": "#55A868",
    "Technology": "#C44E52",
}

SEGMENT_COLORS = {
    "Anchor products": NAVY,
    "Rising products": GREEN,
    "Steady mid-tier": GOLD,
    "Long tail": GREY,
}

PLOTLY_LAYOUT = {
    "template": "plotly_white",
    "margin": {"l": 10, "r": 10, "t": 50, "b": 10},
    "font": {"family": "Inter, Segoe UI, sans-serif", "size": 13, "color": "#22303F"},
    "hoverlabel": {"bgcolor": "white", "font_size": 12},
    "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
}

CUSTOM_CSS = """
<style>
    .block-container {padding-top: 1.6rem; padding-bottom: 2.5rem; max-width: 1250px;}
    h1, h2, h3 {letter-spacing: -0.01em;}
    div[data-testid="stMetricValue"] {font-size: 1.55rem; color: #1B2A3A;}
    div[data-testid="stMetricLabel"] {color: #5A6B7C;}
    .kpi-note {color: #5A6B7C; font-size: 0.82rem;}
    .callout {
        background: #F2F6FA; border-left: 4px solid #2B5D8C;
        padding: 0.85rem 1rem; border-radius: 6px; font-size: 0.93rem;
    }
    .warn {
        background: #FFF4EC; border-left: 4px solid #E4572E;
        padding: 0.85rem 1rem; border-radius: 6px; font-size: 0.93rem;
    }
    footer {visibility: hidden;}
</style>
"""


def fmt_money(value: float, digits: int = 0) -> str:
    return f"${value:,.{digits}f}"


def fmt_compact(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:.0f}k"
    return f"${value:,.0f}"


def fmt_pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


@st.cache_data(show_spinner=False)
def load_table(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run `python run.py all` to regenerate the dashboard feed."
        )
    frame = pd.read_csv(path)
    for column in ("date", "week_start", "month_start", "first_sale", "last_sale"):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column])
    return frame


@st.cache_data(show_spinner=False)
def load_json(name: str) -> dict:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run `python run.py all` to regenerate the dashboard feed."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_feed() -> dict:
    """Load every table once; Streamlit caches each file individually."""
    return {
        "daily": load_table("daily.csv"),
        "weekly": load_table("weekly.csv"),
        "monthly": load_table("monthly.csv"),
        "monthly_category": load_table("monthly_category.csv"),
        "products": load_table("products.csv"),
        "weekday": load_table("weekday.csv"),
        "backtest": load_table("backtest.csv"),
        "forecast": load_table("forecast.csv"),
        "anomalies": load_table("anomalies.csv"),
        "kpis": load_json("kpis.json"),
        "dimensions": load_json("dimensions.json"),
    }


def apply_layout(figure, height: int = 380, title: str | None = None):
    """Apply the shared plotly layout to a figure and set its height."""
    figure.update_layout(**PLOTLY_LAYOUT, height=height)
    if title:
        figure.update_layout(title={"text": title, "x": 0.01, "xanchor": "left", "font": {"size": 16}})
    figure.update_layout(showlegend=figure.layout.showlegend)
    return figure
