"""Shared fixtures.

The cleaned frame is loaded once per session (a few hundred milliseconds) and
every test reads from it. Tests that would train the forecast reuse a cached
``metrics`` payload from the committed artefacts, so the suite stays under a
minute on a laptop.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from superstore.clean import clean_orders  # noqa: E402
from superstore.data import load_raw  # noqa: E402


@pytest.fixture(scope="session")
def raw():
    return load_raw()


@pytest.fixture(scope="session")
def cleaned(raw):
    orders, audit = clean_orders(raw)
    return orders, audit


@pytest.fixture(scope="session")
def orders(cleaned):
    return cleaned[0]


@pytest.fixture(scope="session")
def audit(cleaned):
    return cleaned[1]


@pytest.fixture(scope="session")
def weekly(orders):
    from superstore.features import weekly_revenue

    return weekly_revenue(orders)


@pytest.fixture(scope="session")
def monthly(orders):
    from superstore.features import monthly_revenue

    return monthly_revenue(orders)


@pytest.fixture(scope="session")
def products(orders):
    from superstore.features import product_summary

    return product_summary(orders)


@pytest.fixture(scope="session")
def metrics():
    path = ROOT / "reports" / "metrics.json"
    if not path.exists():
        pytest.skip("run `python run.py all` first: reports/metrics.json is missing")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def diagnostics_summary():
    path = ROOT / "reports" / "diagnostics_summary.json"
    if not path.exists():
        pytest.skip("run `python run.py all` first")
    return json.loads(path.read_text(encoding="utf-8"))
