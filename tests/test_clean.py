"""Cleaning invariants: no rows lost, calendar fields agree, audit numbers hold."""

from __future__ import annotations

import pandas as pd


def test_no_rows_are_dropped(orders, audit):
    assert audit["rows_in"] == audit["rows_out"] == len(orders) == 9_994
    assert audit["duplicate_rows_removed"] == 0


def test_calendar_fields_agree_with_the_order_date(orders):
    assert (orders["Week Start"].dt.weekday == 0).all()
    assert (orders["Week Start"] <= orders["Order Date"]).all()
    assert (orders["Order Date"] - orders["Week Start"]).dt.days.between(0, 6).all()
    assert (orders["Month Start"].dt.day == 1).all()
    assert (orders["Month Start"].dt.to_period("M") == orders["Order Date"].dt.to_period("M")).all()
    assert (orders["Order Year"] == orders["Order Date"].dt.year).all()


def test_weekday_index_is_monday_zero(orders):
    monday = orders[orders["Order Weekday"] == "Monday"]
    assert (monday["Weekday Index"] == 0).all()


def test_derived_financial_fields(orders):
    unit_price = orders["Sales"] / orders["Quantity"]
    assert (unit_price - orders["Unit Price"]).abs().max() < 1e-6
    assert (orders["Margin"] - orders["Profit"] / orders["Sales"]).abs().max() < 1e-9
    returns = orders["Profit"] < 0
    assert (orders["Is Return"] == returns).all()


def test_discount_bands_cover_every_row(orders):
    assert orders["Discount Band"].notna().all()
    assert set(orders["Discount Band"].unique()) == {
        "No discount",
        "1-10%",
        "11-20%",
        "21-40%",
        "41%+",
    }


def test_shipping_lag_and_discount_shares(orders, audit):
    assert orders["Ship Lag Days"].between(0, 8).all()
    assert 0.48 < audit["share_discounted"] < 0.56
    assert 0.15 < audit["share_return_lines"] < 0.22
    assert audit["median_ship_lag_days"] == 4.0
    assert 0.2 < audit["median_margin"] < 0.35


def test_order_id_year_mismatch_is_flagged_not_dropped(orders, audit):
    flagged = orders[orders["Id Year Mismatch"] == 1]
    assert len(flagged) == audit["order_id_year_mismatches"] == 7
    assert (flagged["Order Date"] == pd.Timestamp("2016-12-31")).all()
    assert (flagged["Ship Date"] >= pd.Timestamp("2017-01-01")).all()
    assert flagged["Order ID"].str.contains("-2017-").all()


def test_week_and_month_counts(orders, audit):
    assert audit["weeks"] == 209
    assert audit["months"] == 48
    assert orders["Week Start"].nunique() == 209
    assert orders["Month Start"].nunique() == 48
