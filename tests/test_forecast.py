"""Forecast design, scoring and the clipping guard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from superstore.features import build_origin_panel, weekly_revenue
from superstore.forecast import (
    _clip_predictions,
    build_gradient_boosting,
    build_ridge,
    evaluate_holdout,
    forecast_forward,
    mase,
    regression_metrics,
    train_and_evaluate,
)


def test_regression_metrics_are_exact_on_perfect_predictions():
    y = np.array([10.0, 20.0, 30.0])
    scores = regression_metrics(y, y)
    assert scores["rmse"] == 0.0
    assert scores["mae"] == 0.0
    assert scores["mape"] == 0.0
    assert scores["r2"] == 1.0
    assert scores["bias"] == 0.0


def test_mase_uses_the_naive_baseline():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([10.0, 20.0, 30.0])
    naive_errors = np.array([5.0, 5.0, 5.0])
    assert mase(y_true, y_pred, naive_errors) == 0.0
    assert mase(y_true, y_true + 10, naive_errors) == pytest.approx(2.0)


def test_clipping_keeps_predictions_inside_the_band(weekly):
    origin = 160
    panel = build_origin_panel(weekly, [origin], horizon=13)
    extreme = np.full(13, 1_000_000.0)
    clipped = _clip_predictions(extreme, panel)
    lag52 = panel["target_lag52"].to_numpy()
    upper = np.expm1(lag52) * 2.5
    assert (clipped <= upper + 1e-6).all()
    assert (clipped >= np.expm1(lag52) * 0.4 - 1e-6).all()


def test_holdout_has_thirteen_weeks_and_ordered_horizons(weekly):
    frame, result = evaluate_holdout(weekly)
    payload = result["payload"]
    assert len(frame) == 13
    assert frame["horizon"].tolist() == list(range(1, 14))
    assert frame["week_start"].is_monotonic_increasing
    assert set(frame.columns) >= {"week_start", "actual", "predicted", "horizon", "lower", "upper"}
    assert (frame["lower"] <= frame["predicted"]).all()
    assert (frame["predicted"] <= frame["upper"]).all()
    assert frame["model"].nunique() == 1


def test_holdout_metrics_match_the_committed_artefacts(weekly, metrics):
    frame, result = evaluate_holdout(weekly)
    payload = result["payload"]
    assert payload["selected_model"] == metrics["selected_model"] == "ridge_regression"
    assert payload["test"]["rmse"] == pytest.approx(metrics["test"]["rmse"], rel=1e-6)
    assert payload["test"]["mase"] == pytest.approx(metrics["test"]["mase"], rel=1e-6)
    assert payload["test"]["bias"] == pytest.approx(metrics["test"]["bias"], rel=1e-6)
    assert payload["improvement_vs_naive_pct"] == pytest.approx(
        metrics["improvement_vs_naive_pct"], rel=1e-6
    )
    assert payload["test_all_models"]["seasonal_naive"]["mase"] == 1.0
    assert payload["interval_coverage"] == pytest.approx(metrics["interval_coverage"], rel=1e-6)


def test_model_selection_uses_walk_forward_not_holdout(metrics):
    assert metrics["selection_metric"] == "mean walk-forward RMSE inside the training period"
    assert metrics["selection_scores"]["ridge_regression"] < metrics["selection_scores"]["gradient_boosting"]
    assert 0 < metrics["improvement_vs_naive_pct"] < 30
    assert metrics["test"]["r2"] < 0


def test_feature_importance_is_reported(metrics):
    importance = metrics["feature_importance"]
    assert len(importance) == len(metrics["features"]) == 8
    assert any(row["importance_mean"] > 0 for row in importance)
    assert {row["feature"] for row in importance} == set(metrics["features"])


def test_forecast_forward_returns_a_future_series(weekly):
    frame = forecast_forward(weekly, horizon=13)
    assert len(frame) == 13
    assert (frame["week_start"] > weekly["week_start"].max()).all()
    assert (frame["predicted"] > 0).all()
    assert set(frame.columns) >= {"week_start", "predicted", "horizon"}


def test_models_are_deterministic():
    first = build_ridge()
    second = build_ridge()
    assert str(first) == str(second)
    assert build_gradient_boosting() is not None
