"""Smoke checks over the committed summary artefacts (no retraining)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "reports" / name).read_text(encoding="utf-8"))


def test_eda_summary_has_the_expected_shape():
    summary = _load("eda_summary.json")
    assert summary["rows"] == 9_994
    assert summary["orders"] == 5_009
    assert summary["products"] == 1_862
    assert summary["products_for_80pct"] == 414
    assert "weekday_artifact" in summary


def test_timeseries_summary_is_internally_consistent():
    summary = _load("timeseries_summary.json")
    assert 0 < summary["trend_strength"] < 1
    assert 0 < summary["seasonal_strength"] < 1
    assert summary["strongest_month"] in {"November", "December"}
    assert summary["peak_to_trough"] > 3


def test_segments_summary_matches_the_catalogue():
    summary = _load("segments_summary.json")
    assert summary["products"] == 1_862
    assert sum(summary["abc_products"].values()) == 1_862
    assert summary["anchor_share"] > 0.6


def test_anomalies_summary_lists_four_spikes():
    summary = _load("anomalies_summary.json")
    assert summary["anomalies"] == 4
    assert summary["spikes"] == 4
    assert summary["dips"] == 0


def test_metrics_summary_beats_the_naive_baseline():
    summary = _load("metrics.json")
    assert summary["selected_model"] == "ridge_regression"
    assert summary["improvement_vs_naive_pct"] > 0
    assert summary["test"]["mase"] < 1.0
    assert summary["test"]["n"] == 13


def test_diagnostics_summary_covers_every_horizon():
    summary = _load("diagnostics_summary.json")
    assert summary["weeks"] == 13
    assert {int(key) for key in summary["horizon_error"]} == set(range(1, 14))
    assert 0 < summary["interval_coverage"] <= 1
