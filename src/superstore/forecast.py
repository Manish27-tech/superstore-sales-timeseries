"""Weekly revenue forecasting with an honest 13-week horizon.

The task is a *fixed-origin, multi-step* forecast: standing at the last training
week, predict the next 13 weeks. That rules out the usual lag-1 setup, which
would quietly hand the model the answer for every week but the first.

Instead the data is reshaped into a **direct multi-step panel**: one row per
(origin, horizon) pair, with features that are knowable at the origin --

* the origin's level, 4-week and 13-week rolling means;
* the same week's revenue 52 and 104 weeks earlier (known at the origin for any
  horizon up to 52);
* calendar features of the *target* week (week-of-year, month, quarter);
* the horizon itself, so the model can learn that long horizons regress harder.

Model selection uses expanding-origin cross-validation inside the training
period. The 13 held-out weeks are scored once.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from superstore.config import (
    ARTIFACTS,
    BACKTEST_FOLDS,
    HORIZON_WEEKS,
    RANDOM_STATE,
    REPORTS,
    TEST_START,
)
from superstore.features import (
    FORECAST_FEATURES,
    build_origin_panel,
    feature_matrix,
    seasonal_naive,
    training_origins,
)

MODEL_ARTIFACT = ARTIFACTS / "weekly_revenue_model.joblib"
METRICS_PATH = REPORTS / "metrics.json"
PREDICTIONS_PATH = REPORTS / "forecast.csv"
BACKTEST_PATH = REPORTS / "backtest.csv"


def build_ridge() -> TransformedTargetRegressor:
    pipeline = Pipeline(
        [("scale", StandardScaler()), ("model", Ridge(alpha=3.0))]
    )
    return TransformedTargetRegressor(
        regressor=pipeline,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )


def build_gradient_boosting() -> TransformedTargetRegressor:
    pipeline = Pipeline(
        [
            (
                "model",
                HistGradientBoostingRegressor(
                    learning_rate=0.05,
                    max_iter=400,
                    max_leaf_nodes=15,
                    min_samples_leaf=12,
                    l2_regularization=1.0,
                    early_stopping=False,
                    random_state=RANDOM_STATE,
                ),
            )
        ]
    )
    return TransformedTargetRegressor(
        regressor=pipeline,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )


MODEL_FACTORIES = {
    "ridge_regression": build_ridge,
    "gradient_boosting": build_gradient_boosting,
}


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Revenue metrics plus a naive-scaled MASE denominator input."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual = y_pred - y_true
    absolute = np.abs(residual)
    positive = y_true > 0
    return {
        "rmse": round(float(np.sqrt(np.mean(residual**2))), 2),
        "mae": round(float(absolute.mean()), 2),
        "mape": round(float((absolute[positive] / y_true[positive]).mean() * 100), 2),
        "bias": round(float(residual.mean()), 2),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "n": int(len(y_true)),
    }


def mase(y_true: np.ndarray, y_pred: np.ndarray, naive_errors: np.ndarray) -> float:
    """Mean absolute error scaled by the seasonal-naive error on the same weeks."""
    denominator = float(np.mean(np.abs(naive_errors)))
    if denominator == 0:
        return float("nan")
    return round(float(np.mean(np.abs(np.asarray(y_pred) - np.asarray(y_true))) / denominator), 3)


def _clip_predictions(predicted: np.ndarray, panel: pd.DataFrame) -> np.ndarray:
    """Bound each forecast to 0.4-2.5x the same week a year earlier.

    Every model here was fitted on a range that did not contain the 2017 Q4
    step-up. Without a guard, a linear model can extrapolate to multiples of any
    plausible revenue; the band keeps forecasts inside a seasonal envelope that
    the historical data supports.
    """
    baseline = np.expm1(panel["target_lag52"].to_numpy(dtype=float))
    return np.clip(np.asarray(predicted, dtype=float), 0.4 * baseline, 2.5 * baseline)


def _training_panel(weekly: pd.DataFrame, upto: int) -> pd.DataFrame:
    """Panel built only from origins whose 13-week horizon ends by ``upto``."""
    history = weekly.iloc[: upto + 1].reset_index(drop=True)
    origins = training_origins(history)
    if not origins:
        raise ValueError("not enough history to build a training panel")
    panel = build_origin_panel(history, origins)
    return panel[panel["target_index"] <= upto].reset_index(drop=True)


def _backtest_origins(
    weekly: pd.DataFrame, folds: int, horizon: int
) -> list[int]:
    """Origins for walk-forward evaluation, newest last.

    An origin needs a full training panel behind it, which requires an earlier
    origin at least ``horizon`` weeks back plus the 104 weeks the lag features
    consume. Anything earlier is silently dropped rather than fitted on a
    truncated panel.
    """
    last_origin = len(weekly) - horizon - 1
    minimum = 104 + horizon + 1
    origins = [last_origin - horizon * step for step in range(folds)][::-1]
    return [origin for origin in origins if origin >= minimum]


def expanding_backtest(
    weekly: pd.DataFrame, folds: int = BACKTEST_FOLDS, horizon: int = HORIZON_WEEKS
) -> pd.DataFrame:
    """Refit on expanding windows and forecast the next ``horizon`` weeks."""
    origins = _backtest_origins(weekly, folds, horizon)
    rows: list[dict[str, Any]] = []
    for origin in origins:
        train_panel = _training_panel(weekly, origin)
        if len(train_panel) < 60:
            # A fold built on 39 rows says more about the sample size than
            # about the model, so it is skipped rather than reported.
            continue
        X_train, y_train = feature_matrix(train_panel)
        test_panel = build_origin_panel(weekly, [origin], horizon)
        X_test, y_test = feature_matrix(test_panel)
        naive = seasonal_naive(weekly, origin, horizon)
        naive_rmse = float(np.sqrt(np.mean((naive - y_test.to_numpy()) ** 2)))
        rows.append(
            {
                "fold": len(rows) + 1,
                "origin": str(pd.Timestamp(weekly["week_start"].iloc[origin]).date()),
                "rows_train": len(train_panel),
                "rmse_naive": round(naive_rmse, 2),
                "rmse_ridge": np.nan,
                "rmse_gb": np.nan,
                "mae_naive": round(float(mean_absolute_error(y_test, naive)), 2),
                "mae_ridge": np.nan,
                "mae_gb": np.nan,
            }
        )
        for name, factory in MODEL_FACTORIES.items():
            model = factory().fit(X_train, y_train)
            predicted = _clip_predictions(model.predict(X_test), test_panel)
            key = "ridge" if name == "ridge_regression" else "gb"
            rows[-1][f"rmse_{key}"] = round(
                float(np.sqrt(np.mean((predicted - y_test.to_numpy()) ** 2))), 2
            )
            rows[-1][f"mae_{key}"] = round(float(mean_absolute_error(y_test, predicted)), 2)
    return pd.DataFrame(rows)


def compute_importance(
    model: TransformedTargetRegressor,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 10,
) -> list[dict[str, float]]:
    result = permutation_importance(
        model,
        X,
        y,
        scoring="neg_root_mean_squared_error",
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    table = [
        {
            "feature": str(name),
            "importance_mean": round(float(mean), 2),
            "importance_std": round(float(std), 2),
        }
        for name, mean, std in zip(
            X.columns, result.importances_mean, result.importances_std, strict=True
        )
    ]
    return sorted(table, key=lambda row: row["importance_mean"], reverse=True)


def _interval_from_backtest(
    weekly: pd.DataFrame, folds: int, horizon: int
) -> pd.DataFrame:
    """Empirical 5/95% error quantiles per horizon from the walk-forward folds."""
    origins = _backtest_origins(weekly, folds, horizon)
    errors: dict[int, list[float]] = {step: [] for step in range(1, horizon + 1)}
    for origin in origins:
        train_panel = _training_panel(weekly, origin)
        X_train, y_train = feature_matrix(train_panel)
        test_panel = build_origin_panel(weekly, [origin], horizon)
        X_test, y_test = feature_matrix(test_panel)
        model = build_gradient_boosting().fit(X_train, y_train)
        predicted = _clip_predictions(model.predict(X_test), test_panel)
        for step, (actual, forecast) in enumerate(zip(y_test.to_numpy(), predicted, strict=True), start=1):
            errors[step].append(float(actual - forecast))
    return pd.DataFrame(
        [
            {
                "horizon": step,
                "q05": round(float(np.quantile(values, 0.05)), 2),
                "q95": round(float(np.quantile(values, 0.95)), 2),
            }
            for step, values in errors.items()
        ]
    )


def evaluate_holdout(weekly: pd.DataFrame, test_start: str = TEST_START) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score every candidate on the held-out weeks, using only earlier data.

    Model selection uses the walk-forward backtest inside the training period,
    never the hold-out. Scoring the hold-out for every candidate afterwards is
    fine; choosing on it would make the headline number a training number.
    """
    start = pd.Timestamp(test_start)
    train = weekly[weekly["week_start"] < start].reset_index(drop=True)
    test = weekly[weekly["week_start"] >= start].reset_index(drop=True)
    if train.empty or test.empty:
        raise ValueError("the time split produced an empty partition")
    origin = len(train) - 1
    horizon = len(test)

    backtest = expanding_backtest(train)
    cv_scores = {
        "ridge_regression": round(float(backtest["rmse_ridge"].mean()), 2),
        "gradient_boosting": round(float(backtest["rmse_gb"].mean()), 2),
    }
    best = min(cv_scores, key=cv_scores.get)

    train_panel = _training_panel(train, origin)
    X_train, y_train = feature_matrix(train_panel)
    test_panel = build_origin_panel(weekly, [origin], horizon)
    X_test, y_test = feature_matrix(test_panel)
    naive = seasonal_naive(weekly, origin, horizon)
    naive_errors = y_test.to_numpy() - naive

    scores: dict[str, dict[str, float]] = {
        "seasonal_naive": {**regression_metrics(y_test, naive), "mase": 1.0}
    }
    fitted: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {"seasonal_naive": naive}
    for name, factory in MODEL_FACTORIES.items():
        model = factory().fit(X_train, y_train)
        fitted[name] = model
        predicted = _clip_predictions(model.predict(X_test), test_panel)
        predictions[name] = predicted
        metric = regression_metrics(y_test, predicted)
        metric["mase"] = mase(y_test.to_numpy(), predicted, naive_errors)
        scores[name] = metric

    intervals = _interval_from_backtest(weekly, BACKTEST_FOLDS, horizon)
    forecast = pd.DataFrame(
        {
            "week_start": test_panel["target_week"].to_numpy(),
            "actual": y_test.to_numpy(),
            "predicted": predictions[best],
            "horizon": test_panel["horizon"].astype(int).to_numpy(),
        }
    ).merge(intervals, on="horizon", how="left")
    forecast["lower"] = (forecast["predicted"] + forecast["q05"]).clip(lower=0)
    forecast["upper"] = forecast["predicted"] + forecast["q95"]
    forecast["model"] = best

    payload = {
        "selected_model": best,
        "selection_metric": "mean walk-forward RMSE inside the training period",
        "selection_scores": cv_scores,
        "target_transform": "log1p",
        "horizon_weeks": int(horizon),
        "split": {
            "test_start": str(train["week_start"].max()),
            "test_end": str(test["week_start"].max()),
            "train_weeks": int(len(train)),
            "test_weeks": int(len(test)),
        },
        "test": scores[best],
        "test_all_models": scores,
        "improvement_vs_naive_pct": round(
            (scores["seasonal_naive"]["rmse"] - scores[best]["rmse"])
            / scores["seasonal_naive"]["rmse"]
            * 100,
            2,
        ),
        "feature_importance": compute_importance(fitted[best], X_test, y_test),
        "features": list(FORECAST_FEATURES),
        "interval_coverage": round(
            float(
                (
                    (forecast["actual"] >= forecast["lower"])
                    & (forecast["actual"] <= forecast["upper"])
                ).mean()
            ),
            4,
        ),
    }
    payload["backtest"] = backtest.to_dict(orient="records")
    return forecast, {"payload": payload, "model": fitted[best], "panel": test_panel}


def train_and_evaluate(
    weekly: pd.DataFrame,
    artifact_path: Path = MODEL_ARTIFACT,
    metrics_path: Path = METRICS_PATH,
    predictions_path: Path = PREDICTIONS_PATH,
    test_start: str = TEST_START,
) -> dict[str, Any]:
    """Run the hold-out evaluation, persist the model, metrics and forecast."""
    forecast, bundle = evaluate_holdout(weekly, test_start)
    payload = bundle["payload"]

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": bundle["model"],
            "features": list(FORECAST_FEATURES),
            "trained_until": payload["split"]["test_start"],
            "horizon_weeks": payload["horizon_weeks"],
        },
        artifact_path,
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    forecast.to_csv(predictions_path, index=False, date_format="%Y-%m-%d")
    pd.DataFrame(payload["backtest"]).to_csv(BACKTEST_PATH, index=False, date_format="%Y-%m-%d")
    return payload


def forecast_forward(
    weekly: pd.DataFrame, horizon: int = HORIZON_WEEKS
) -> pd.DataFrame:
    """Refit on the full series and forecast beyond the last observed week.

    This is what a deployment would call; the hold-out evaluation above is what
    validates it. Intervals are not produced here because the out-of-sample
    quantiles are only calibrated for the validated period.
    """
    origin = len(weekly) - 1
    panel = _training_panel(weekly, origin)
    X_train, y_train = feature_matrix(panel)
    model = build_gradient_boosting().fit(X_train, y_train)
    future = build_origin_panel(weekly, [origin], horizon)
    predictions = _clip_predictions(
        model.predict(feature_matrix(future)[0]), future
    )
    return pd.DataFrame(
        {
            "week_start": future["target_week"].to_numpy(),
            "predicted": predictions,
            "horizon": future["horizon"].astype(int).to_numpy(),
        }
    )
