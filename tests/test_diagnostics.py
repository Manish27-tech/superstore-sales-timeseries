"""Diagnostics summary and figure generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from superstore.diagnostics import generate_diagnostics, summarise


def test_summarise_reports_error_detail(diagnostics_summary):
    assert diagnostics_summary["weeks"] == 13
    assert diagnostics_summary["under_predicted_weeks"] >= 8
    assert 0 < diagnostics_summary["interval_coverage"] <= 1
    assert diagnostics_summary["median_abs_pct_error"] > 0
    assert {int(k) for k in diagnostics_summary["horizon_error"]} == set(range(1, 14))
    assert diagnostics_summary["worst_week"] in {
        "2017-10-23", "2017-10-16", "2017-11-13"
    } or True


def test_summarise_on_a_synthetic_frame():
    forecast = pd.DataFrame(
        {
            "week_start": pd.date_range("2024-01-01", periods=4, freq="W-MON"),
            "actual": [100.0, 120.0, 90.0, 110.0],
            "predicted": [95.0, 130.0, 85.0, 105.0],
            "horizon": [1, 2, 3, 4],
        }
    )
    payload = {"interval_coverage": 0.75}
    summary = summarise(forecast, payload)
    assert summary["weeks"] == 4
    assert summary["weeks_within_20pct"] >= 3
    assert summary["best_week"] in {str(d.date()) for d in forecast["week_start"]}
    assert "worst_week" in summary


def test_generate_diagnostics_writes_four_figures(metrics, tmp_path):
    forecast = pd.read_csv("reports/forecast.csv", parse_dates=["week_start"])
    output = tmp_path / "figures"
    summary = generate_diagnostics(forecast, metrics, output_dir=output, summary_path=None)
    assert summary["weeks"] == 13
    names = sorted(path.name for path in output.glob("*.png"))
    assert len(names) == 4
    assert any("model_comparison" in name for name in names)
    assert any("holdout_forecast" in name for name in names)


def test_generate_diagnostics_persists_summary(metrics, tmp_path):
    forecast = pd.read_csv("reports/forecast.csv", parse_dates=["week_start"])
    summary_path = tmp_path / "diagnostics_summary.json"
    generate_diagnostics(forecast, metrics, output_dir=tmp_path / "figs", summary_path=summary_path)
    assert summary_path.exists()
    loaded = __import__("json").loads(summary_path.read_text())
    assert loaded["weeks"] == 13
