"""Anomaly detection: guards, signs, and explanations that point at real orders."""

from __future__ import annotations

import numpy as np
import pandas as pd

from superstore.anomalies import (
    detect_weekly_anomalies,
    explain_anomalies,
    summarise,
)


def test_scores_only_start_once_a_year_of_history_exists(weekly):
    scored = detect_weekly_anomalies(weekly)
    assert scored["z_score"].isna().sum() == 52
    assert scored["expected"].notna().sum() == len(weekly) - 52
    assert scored["z_score"].notna().sum() == 155


def test_guards_remove_tiny_baselines(weekly):
    scored = detect_weekly_anomalies(weekly)
    assert (scored.loc[scored["eligible"], "expected"] >= 2_000).all()
    assert (scored.loc[scored["eligible"], "deviation"].abs() >= 3_000).all()
    assert not scored.loc[~scored["eligible"], "is_anomaly"].any()


def test_four_anomalies_and_all_are_spikes(weekly):
    scored = detect_weekly_anomalies(weekly)
    flagged = scored[scored["is_anomaly"]]
    assert len(flagged) == 4
    assert (flagged["direction"] == "spike").all()
    assert (flagged["z_score"] >= 3.5).all()


def test_top_z_score_belongs_to_a_copier_order(weekly):
    scored = detect_weekly_anomalies(weekly)
    flagged = scored[scored["is_anomaly"]]
    top = flagged.nlargest(1, "z_score").iloc[0]
    assert str(top["week_start"].date()) == "2016-03-07"
    assert top["z_score"] > 7


def test_2017_copier_week_is_among_the_anomalies(weekly):
    scored = detect_weekly_anomalies(weekly)
    flagged = scored[scored["is_anomaly"]]
    weeks = {str(value.date()) for value in flagged["week_start"]}
    assert "2017-03-20" in weeks


def test_explanations_attach_the_largest_order_line(orders, weekly):
    scored = detect_weekly_anomalies(weekly)
    flagged = scored[scored["is_anomaly"]]
    explained = explain_anomalies(orders, flagged)
    assert len(explained) == len(flagged)
    for row in explained.itertuples():
        week = row.week_start
        lines = orders[orders["Week Start"] == week]
        assert abs(row.largest_order_value - lines["Sales"].max()) < 0.01
        assert 0 < row.largest_order_share <= 1
        assert row.largest_order_value <= row.revenue


def test_summary_is_internally_consistent(orders, weekly):
    scored = detect_weekly_anomalies(weekly)
    flagged = scored[scored["is_anomaly"]]
    explained = explain_anomalies(orders, flagged)
    summary = summarise(scored, explained)
    assert summary["weeks_scored"] == 155
    assert summary["eligible_weeks"] == int(scored["eligible"].sum())
    assert summary["anomalies"] == summary["spikes"] + summary["dips"] == 4
    assert summary["spikes"] == 4 and summary["dips"] == 0
    assert 0 < summary["share_of_weeks"] < 0.05
    assert len(summary["top_anomalies"]) == 4


def test_threshold_is_respected_on_synthetic_data():
    weeks = pd.date_range("2015-01-05", periods=120, freq="W-MON")
    revenue = np.full(120, 10_000.0)
    revenue[110:] = 200_000.0
    frame = pd.DataFrame({"week_start": weeks, "revenue": revenue})
    scored = detect_weekly_anomalies(frame, threshold=1.0, min_expected=1_000.0, min_deviation=1_000.0)
    assert scored["is_anomaly"].sum() >= 1
