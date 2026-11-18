"""Decomposition: a plausible trend, a stable seasonal index, honest strengths."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from superstore.timeseries import decompose, seasonal_factors, strength, trailing_growth


def test_decomposition_columns_and_edge_handling(monthly):
    decomposed = decompose(monthly)
    assert {"month_start", "revenue", "trend", "seasonal", "residual"} <= set(decomposed.columns)
    # A centred 12-month average cannot exist at the boundaries of the series.
    assert decomposed["trend"].isna().sum() == 11
    assert decomposed["trend"].notna().sum() == 37
    assert (decomposed["trend"].dropna() > 0).all()


def test_seasonal_index_is_a_ratio_series(monthly):
    factors = seasonal_factors(decompose(monthly))
    assert len(factors) == 12
    assert factors.notna().all()
    # Multiplicative factors should average one and spread widely.
    assert np.isclose(float(factors.mean()), 1.0, atol=1e-9)
    assert factors.min() < 0.6
    assert factors.max() > 1.4


def test_peak_months_are_november_and_december(monthly):
    factors = seasonal_factors(decompose(monthly))
    top_two = set(factors.nlargest(2).index)
    assert top_two == {"Nov", "Dec"}
    assert factors.idxmin() == "Feb"
    assert factors["Sep"] > 1.5


def test_strength_measures_are_bounded(monthly):
    scores = strength(decompose(monthly))
    assert {"trend_strength", "seasonal_strength", "residual_share"} <= set(scores)
    for name, value in scores.items():
        assert 0.0 <= value <= 1.0, name
    assert scores["seasonal_strength"] > scores["trend_strength"]


def test_decomposition_requires_two_cycles(monthly):
    with pytest.raises(ValueError):
        decompose(monthly.head(20))


def test_trailing_growth_is_a_rolling_sum(monthly):
    growth = trailing_growth(monthly)
    assert len(growth) == len(monthly)
    assert {"trailing_12m", "trailing_yoy"} <= set(growth.columns)
    assert growth["trailing_12m"].iloc[:11].isna().all()
    assert np.isclose(growth["trailing_12m"].iloc[-1], monthly["revenue"].iloc[-12:].sum())


def test_decomposition_uses_the_calendar_month_index(monthly):
    decomposed = decompose(monthly)
    assert decomposed["month_start"].is_monotonic_increasing
    assert decomposed["month_start"].dt.day.eq(1).all()
    assert isinstance(decomposed["seasonal"], pd.Series)
