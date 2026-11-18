"""Streamlit smoke test and helper formatting."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
from streamlit.testing.v1 import AppTest

from dashboard.theme import fmt_compact, fmt_money, fmt_pct, kpi_cards


def test_helpers_format_values():
    assert fmt_money(1234.5) == "$1,234"
    assert fmt_compact(2_297_200.86) == "$2.30M"
    assert fmt_pct(0.1234, 1) == "12.3%"


def test_app_runs_without_exceptions():
    ROOT = Path(__file__).resolve().parents[1]
    at = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=180)
    at.run()
    assert len(at.exception) == 0, [str(exc) for exc in at.exception]


def test_app_renders_all_sections():
    ROOT = Path(__file__).resolve().parents[1]
    at = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=180)
    at.run()
    # The sidebar radio exposes every section.
    assert len(at.radio) >= 1
    labels = [option for option in at.radio[0].options]
    assert set(labels) == {
        "Overview",
        "Seasonality",
        "Products",
        "Forecast",
        "Anomalies",
        "Explore",
        "About",
    }


def test_feed_tables_are_loadable():
    from dashboard.theme import load_feed

    feed = load_feed()
    assert set(feed) >= {
        "daily",
        "weekly",
        "monthly",
        "monthly_category",
        "weekday",
        "products",
        "forecast",
        "backtest",
        "anomalies",
        "kpis",
        "dimensions",
    }
    assert len(feed["products"]) == 1_862
    assert feed["forecast"].shape[0] == 13
    assert feed["backtest"].shape[0] == 4
    assert feed["dimensions"]["categories"] == ["Furniture", "Office Supplies", "Technology"]
