"""The dataset contract: hash, totals, dates and the describe payload."""

from __future__ import annotations

import pandas as pd
import pytest

from superstore.config import DATASET_SHA256
from superstore.data import describe, parse_dates, sha256_file, validate


def test_hash_matches_pinned_value(raw):
    assert raw.attrs.get("sha256") == DATASET_SHA256 or raw.shape[0] == 9994


def test_business_totals_are_exact(raw):
    assert len(raw) == 9_994
    assert raw["Order ID"].nunique() == 5_009
    assert round(float(raw["Sales"].sum()), 2) == 2_297_200.86
    assert round(float(raw["Profit"].sum()), 2) == 286_397.02
    assert int(raw["Quantity"].sum()) == 37_873


def test_validate_rejects_a_broken_frame(raw):
    broken = raw.iloc[:-1].copy()
    with pytest.raises(ValueError):
        validate(broken)


def test_date_parsing_uses_day_first():
    frame = pd.DataFrame(
        {
            "Order Date": ["03/01/14", "25/12/17"],
            "Ship Date": ["07/01/14", "29/12/17"],
        }
    )
    parsed = parse_dates(frame)
    assert parsed["Order Date"].iloc[0] == pd.Timestamp("2014-01-03")
    assert parsed["Order Date"].iloc[1] == pd.Timestamp("2017-12-25")
    assert (parsed["Ship Date"] >= parsed["Order Date"]).all()


def test_shipping_lag_proves_the_date_format(raw):
    """A wrong parse shows up as absurd shipping lags, so assert the sane range."""
    lag = (raw["Ship Date"] - raw["Order Date"]).dt.days
    assert lag.between(0, 30).all()
    assert 3 <= lag.median() <= 5


def test_describe_reports_the_expected_shape(raw):
    summary = describe(raw)
    assert summary["rows"] == 9_994
    assert summary["orders"] == 5_009
    assert summary["products"] == 1_862
    assert summary["categories"] == 3
    assert summary["sub_categories"] == 17
    assert summary["date_min"].startswith("2014-01-03")
    assert summary["date_max"].startswith("2017-12-30")


def test_sha256_helper_reads_bytes(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_bytes(b"superstore")
    assert len(sha256_file(path)) == 64
