"""Generate the exploratory notebook from the package's own functions.

The notebook is a teaching artefact: every figure it shows is produced by the
same code the pipeline runs, so a reader can rerun a single cell and get the
same number the report quotes.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "01_superstore_timeseries.ipynb"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def main() -> None:
    cells = [
        md(
            "# Superstore product sales: a four-year time series\n\n"
            "Companion notebook to the `superstore-sales-timeseries` project. "
            "It walks through the same stages the pipeline runs, using the "
            "functions in `src/superstore`, so every number here matches "
            "`reports/REPORT.md`.\n"
        ),
        code(
            "import sys\n"
            "from pathlib import Path\n"
            "sys.path.insert(0, str(Path('..') / 'src'))\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "from superstore.data import load_raw\n"
            "from superstore.clean import clean_orders\n"
            "from superstore.features import weekly_revenue, monthly_revenue, product_summary\n"
        ),
        md("## 1. Load and verify the pinned dataset"),
        code(
            "raw = load_raw()\n"
            "orders, audit = clean_orders(raw)\n"
            "print(audit)\n"
            "print('rows', len(orders), 'weeks', orders['Week Start'].nunique())\n"
        ),
        md("## 2. Weekly revenue series"),
        code(
            "weekly = weekly_revenue(orders)\n"
            "weekly.head()\n"
            "weekly['revenue'].plot(figsize=(11, 3.5), title='Weekly revenue')\n"
            "plt.show()\n"
        ),
        md("## 3. Decomposition"),
        code(
            "from superstore.timeseries import decompose, seasonal_factors, strength\n"
            "monthly = monthly_revenue(orders)\n"
            "decomposed = decompose(monthly)\n"
            "print(seasonal_factors(decomposed).round(3).to_dict())\n"
            "print(strength(decomposed))\n"
        ),
        md("## 4. Product concentration and segments"),
        code(
            "from superstore.segments import build_segments, summarise\n"
            "products = product_summary(orders)\n"
            "segmented, profile = build_segments(products)\n"
            "summary = summarise(segmented, profile)\n"
            "print('ABC', summary['abc_products'], summary['abc_revenue_share'])\n"
            "print('segments', summary['segments'])\n"
        ),
        md("## 5. Weekly anomalies"),
        code(
            "from superstore.anomalies import detect_weekly_anomalies, explain_anomalies\n"
            "scored = detect_weekly_anomalies(weekly)\n"
            "flagged = scored[scored['is_anomaly']]\n"
            "explained = explain_anomalies(orders, flagged)\n"
            "explained[['week_start', 'revenue', 'expected', 'z_score', 'largest_order_value']]\n"
        ),
        md("## 6. 13-week forecast"),
        code(
            "from superstore.forecast import evaluate_holdout\n"
            "frame, result = evaluate_holdout(weekly)\n"
            "payload = result['payload']\n"
            "print(payload['selected_model'], payload['test'])\n"
            "frame[['week_start', 'actual', 'predicted', 'lower', 'upper']].round(0)\n"
        ),
        md(
            "## 7. Diagnostics\n\n"
            "See `reports/figures/19-22` for the model comparison, backtest folds, "
            "hold-out forecast and residual diagnostics."
        ),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
