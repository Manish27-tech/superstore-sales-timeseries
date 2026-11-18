"""Verify that every committed artefact the dashboard and docs rely on is present.

Run with ``python scripts/check_artifacts.py``. Exit code 0 means the repo is
self-consistent; exit code 1 lists what is missing or stale. This is the check
Streamlit Community Cloud and CI run before building the app.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "reports/metrics.json",
    "reports/eda_summary.json",
    "reports/timeseries_summary.json",
    "reports/segments_summary.json",
    "reports/anomalies_summary.json",
    "reports/diagnostics_summary.json",
    "reports/REPORT.md",
    "reports/MODEL_CARD.md",
    "data/dashboard/kpis.json",
    "data/dashboard/dimensions.json",
    "data/dashboard/daily.csv",
    "data/dashboard/weekly.csv",
    "data/dashboard/monthly.csv",
    "data/dashboard/monthly_category.csv",
    "data/dashboard/weekday.csv",
    "data/dashboard/products.csv",
    "data/dashboard/forecast.csv",
    "data/dashboard/backtest.csv",
    "data/dashboard/anomalies.csv",
]

REQUIRED_FIGURES = 22


def check() -> tuple[bool, list[str]]:
    problems: list[str] = []
    for relative in REQUIRED_FILES:
        path = ROOT / relative
        if not path.exists() or path.stat().st_size == 0:
            problems.append(f"missing or empty: {relative}")

    figure_dir = ROOT / "reports" / "figures"
    pngs = sorted(figure_dir.glob("*.png")) if figure_dir.exists() else []
    if len(pngs) < REQUIRED_FIGURES:
        problems.append(
            f"only {len(pngs)} of {REQUIRED_FIGURES} figures in reports/figures"
        )

    metrics_path = ROOT / "reports" / "metrics.json"
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            for key in ("selected_model", "test", "test_all_models", "split", "feature_importance"):
                if key not in metrics:
                    problems.append(f"metrics.json missing key: {key}")
            if metrics.get("test", {}).get("n") != 13:
                problems.append("metrics.json hold-out is not 13 weeks")
        except json.JSONDecodeError:
            problems.append("metrics.json is not valid JSON")

    kpis_path = ROOT / "data" / "dashboard" / "kpis.json"
    if kpis_path.exists():
        try:
            kpis = json.loads(kpis_path.read_text(encoding="utf-8"))
            if "sales" not in kpis or "forecast" not in kpis:
                problems.append("kpis.json is missing the sales or forecast block")
        except json.JSONDecodeError:
            problems.append("kpis.json is not valid JSON")

    return (not problems), problems


def main() -> int:
    ok, problems = check()
    if ok:
        print(json.dumps({"status": "ok", "files": len(REQUIRED_FILES), "figures": REQUIRED_FIGURES}, indent=2))
        return 0
    print(json.dumps({"status": "failed", "problems": problems}, indent=2))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
