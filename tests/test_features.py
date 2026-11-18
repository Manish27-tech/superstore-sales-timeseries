"""Aggregation, the direct multi-step panel, and the no-leakage guarantee."""

from __future__ import annotations

import numpy as np
import pandas as pd

from superstore.features import (
    FORECAST_FEATURES,
    abc_classes,
    build_origin_panel,
    daily_revenue,
    feature_matrix,
    monthly_category_revenue,
    monthly_revenue,
    seasonal_naive,
    training_origins,
    weekly_revenue,
)


def test_weekly_series_drops_both_partial_weeks(weekly):
    assert len(weekly) == 207
    assert weekly["week_start"].iloc[0] == pd.Timestamp("2014-01-06")
    assert weekly["week_start"].iloc[-1] == pd.Timestamp("2017-12-18")
    assert weekly["week_start"].dt.weekday.eq(0).all()
    assert weekly["week_start"].diff().dropna().dt.days.eq(7).all()


def test_weekly_totals_reconcile_with_the_raw_sales(raw, weekly):
    total = raw["Sales"].sum()
    assert abs(weekly["revenue"].sum() / total - 1) < 0.01
    assert (weekly["revenue"] > 0).all()
    assert (weekly["margin"] - weekly["profit"] / weekly["revenue"]).abs().max() < 1e-9


def test_monthly_and_daily_aggregates(orders, monthly):
    assert len(monthly) == 48
    assert monthly["month_start"].dt.day.eq(1).all()
    assert (monthly["aov"] > 0).all()

    daily = daily_revenue(orders)
    assert daily["date"].is_monotonic_increasing
    assert daily["date"].diff().dropna().dt.days.eq(1).all()
    assert (daily["revenue"] >= 0).all()
    assert (daily["revenue"] == 0).any()


def test_monthly_category_totals_match_monthly(orders, monthly):
    wide = monthly_category_revenue(orders)
    assert set(wide.columns) == {"month_start", "Furniture", "Office Supplies", "Technology"}
    assert np.allclose(wide.drop(columns="month_start").sum(axis=1), monthly["revenue"])


def test_product_summary_is_complete_and_consistent(products):
    assert len(products) == 1_862
    assert products["product_id"].is_unique
    assert (products["revenue"] > 0).all()
    assert products["revenue"].sum() > 2_000_000
    assert (products["margin"] - products["profit"] / products["revenue"]).abs().max() < 1e-9
    assert products["growth"].notna().sum() > 1_700


def test_abc_classes_are_ordered_by_revenue(products):
    labelled = abc_classes(products.sort_values("revenue", ascending=False).reset_index(drop=True))
    assert set(labelled["abc_class"].unique()) == {"A", "B", "C"}
    ordered = labelled["abc_class"].tolist()
    assert ordered == sorted(ordered, key={"A": 0, "B": 1, "C": 2}.get)


def test_origin_panel_targets_future_weeks_only(weekly):
    origin = 150
    panel = build_origin_panel(weekly, [origin], horizon=13)
    assert len(panel) == 13
    assert panel["horizon"].tolist() == list(range(1, 14))
    origin_week = weekly["week_start"].iloc[origin]
    assert (panel["target_week"] > origin_week).all()
    expected = [
        weekly["week_start"].iloc[origin + step] for step in range(1, 14)
    ]
    assert panel["target_week"].tolist() == expected
    assert np.allclose(panel["y"].to_numpy(), weekly["revenue"].iloc[origin + 1 : origin + 14])


def test_lag52_feature_is_last_years_value(weekly):
    origin = 160
    panel = build_origin_panel(weekly, [origin], horizon=3)
    revenue = weekly["revenue"].to_numpy()
    for row in panel.itertuples():
        target_index = int(row.target_index)
        assert np.isclose(row.target_lag52, np.log1p(revenue[target_index - 52]))


def test_feature_matrix_uses_only_the_declared_features(weekly):
    panel = build_origin_panel(weekly, [140, 144], horizon=13)
    X, y = feature_matrix(panel)
    assert list(X.columns) == list(FORECAST_FEATURES)
    assert len(X) == len(y) == 26
    assert np.isfinite(X.to_numpy()).all()
    assert (y > 0).all()


def test_training_origins_leave_room_for_the_horizon(weekly):
    origins = training_origins(weekly)
    assert origins[0] == 104
    assert origins[-1] <= len(weekly) - 13 - 1
    assert all(b - a == 4 for a, b in zip(origins, origins[1:]))


def test_seasonal_naive_is_literally_last_year(weekly):
    origin = 120
    forecast = seasonal_naive(weekly, origin, horizon=5)
    manual = weekly["revenue"].to_numpy()[origin + 1 - 52 : origin + 6 - 52]
    assert np.allclose(forecast, manual)
    assert (forecast > 0).all()
