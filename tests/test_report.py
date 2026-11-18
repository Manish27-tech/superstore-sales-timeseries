"""Report and model card generation from the committed artefacts."""

from __future__ import annotations

import json
from pathlib import Path

from superstore.report import build_model_card, build_report, load_summaries


def test_load_summaries_returns_all_payloads():
    summaries = load_summaries()
    assert set(summaries) == {
        "eda_summary",
        "timeseries_summary",
        "segments_summary",
        "anomalies_summary",
        "metrics",
        "diagnostics_summary",
    }


def test_build_report_contains_the_headline_and_figures():
    summaries = load_summaries()
    report = build_report(
        summaries["eda_summary"],
        summaries["timeseries_summary"],
        summaries["segments_summary"],
        summaries["anomalies_summary"],
        summaries["metrics"],
        summaries["diagnostics_summary"],
        {
            "rows_in": summaries["eda_summary"]["rows"],
            "rows_out": summaries["eda_summary"]["rows"],
            "duplicate_rows_removed": 0,
            "weeks": summaries["eda_summary"]["weeks"],
            "order_id_year_mismatches": 7,
            "share_discounted": summaries["eda_summary"]["discount_share"],
            "share_return_lines": summaries["eda_summary"]["return_line_share"],
            "median_margin": 0.27,
            "median_ship_lag_days": 4.0,
            "partial_weeks": 2,
            "date_min": summaries["eda_summary"]["date_min"],
            "date_max": summaries["eda_summary"]["date_max"],
            "total_sales": summaries["eda_summary"]["total_sales"],
            "total_profit": summaries["eda_summary"]["total_profit"],
        },
    )
    assert report.startswith("# Superstore product sales")
    assert "## Headline findings" in report
    assert "## Forecasting 13 weeks ahead" in report
    assert "figures/21_holdout_forecast.png" in report
    assert "figures/19_forecast_model_comparison.png" in report
    assert "nan" not in report.lower()
    assert "414 of" in report
    assert "R²" in report


def test_model_card_is_honest_about_failure():
    summaries = load_summaries()
    card = build_model_card(summaries["metrics"], summaries["diagnostics_summary"])
    assert "# Model card" in card
    assert "Where it fails" in card
    assert "Intended use" in card
    assert "Systematic under-prediction" in card or "under-predicted" in card
    assert "ridge regression" in card.lower() or "ridge" in card.lower()


def test_generate_reports_writes_both_files(tmp_path):
    from superstore.report import generate_reports

    summaries = load_summaries()
    paths = generate_reports(
        summaries["eda_summary"],
        summaries["timeseries_summary"],
        summaries["segments_summary"],
        summaries["anomalies_summary"],
        summaries["metrics"],
        summaries["diagnostics_summary"],
        {
            "rows_in": summaries["eda_summary"]["rows"],
            "rows_out": summaries["eda_summary"]["rows"],
            "duplicate_rows_removed": 0,
            "weeks": summaries["eda_summary"]["weeks"],
            "order_id_year_mismatches": 7,
            "share_discounted": summaries["eda_summary"]["discount_share"],
            "share_return_lines": summaries["eda_summary"]["return_line_share"],
            "median_margin": 0.27,
            "median_ship_lag_days": 4.0,
            "partial_weeks": 2,
            "date_min": summaries["eda_summary"]["date_min"],
            "date_max": summaries["eda_summary"]["date_max"],
            "total_sales": summaries["eda_summary"]["total_sales"],
            "total_profit": summaries["eda_summary"]["total_profit"],
        },
        report_path=tmp_path / "REPORT.md",
        card_path=tmp_path / "MODEL_CARD.md",
    )
    assert paths["report"].exists() and paths["model_card"].exists()
    assert "Model card" in paths["model_card"].read_text() or "model card" in paths["model_card"].read_text().lower()
