"""ABC concentration and the behavioural clusters."""

from __future__ import annotations

import numpy as np
import pytest

from superstore.segments import build_segments, summarise


def test_every_product_gets_a_class_and_a_segment(products):
    segmented, profile = build_segments(products)
    assert len(segmented) == len(products) == 1_862
    assert segmented["abc_class"].notna().all()
    assert segmented["segment"].notna().all()
    assert segmented["product_id"].is_unique
    assert len(profile) == 4


def test_abc_headcounts_and_shares(products):
    segmented, profile = build_segments(products)
    summary = summarise(segmented, profile)
    assert summary["abc_products"] == {"A": 413, "B": 489, "C": 960}
    assert summary["abc_revenue_share"]["A"] == pytest.approx(0.80, abs=5e-4)
    assert summary["abc_revenue_share"]["B"] == pytest.approx(0.15, abs=5e-4)
    assert summary["abc_revenue_share"]["C"] == pytest.approx(0.05, abs=5e-4)
    assert summary["products_for_80pct"] == 414


def test_cumulative_share_is_monotone(products):
    segmented, _ = build_segments(products)
    ordered = segmented.sort_values("revenue", ascending=False)
    cumulative = ordered["cumulative_share"].to_numpy()
    assert np.all(np.diff(cumulative) >= -1e-3)
    assert np.isclose(cumulative[-1], 1.0)


def test_segment_revenue_shares_sum_to_one(products):
    segmented, profile = build_segments(products)
    assert np.isclose(profile["share_of_revenue"].sum(), 1.0)
    assert profile["products"].sum() == 1_862
    summary = summarise(segmented, profile)
    assert set(summary["segments"]) == {
        "Anchor products",
        "Rising products",
        "Steady mid-tier",
        "Long tail",
    }


def test_anchor_products_dominate_revenue(products):
    segmented, profile = build_segments(products)
    summary = summarise(segmented, profile)
    assert summary["anchor_share"] > 0.6
    anchors = segmented[segmented["segment"] == "Anchor products"]
    assert anchors["mean_monthly"].median() > segmented["mean_monthly"].median()


def test_clustering_is_reproducible(products):
    first, _ = build_segments(products.head(300))
    second, _ = build_segments(products.head(300))
    assert first["segment"].tolist() == second["segment"].tolist()


def test_summary_reports_the_fastest_growers(products):
    segmented, profile = build_segments(products)
    summary = summarise(segmented, profile)
    assert len(summary["fastest_growing"]) == 5
    growth = [row["growth"] for row in summary["fastest_growing"]]
    assert growth == sorted(growth, reverse=True)
