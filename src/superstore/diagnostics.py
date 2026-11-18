"""Forecast diagnostics: how the model fails, not just how often.

Four figures that a reader needs before trusting a 13-week revenue forecast:

* the model comparison on the held-out weeks,
* the walk-forward folds, which show whether one window was luck,
* the forecast against what actually happened, with its interval,
* where the error lives -- by horizon, and in the residual distribution.

The diagnostics deliberately show the under-prediction of the 2017 fourth
quarter rather than smoothing it away: a model that misses a record quarter by
6% of its weekly revenue is worth knowing about.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from superstore.config import FIGURE_DPI, FIGURES, PALETTE, REPORTS  # noqa: E402

MODEL_LABELS = {
    "seasonal_naive": "Seasonal naive",
    "ridge_regression": "Ridge",
    "gradient_boosting": "Gradient boosting",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": FIGURE_DPI,
            "savefig.dpi": FIGURE_DPI,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "font.size": 9.5,
        }
    )


def _save(fig: plt.Figure, name: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_model_comparison(payload: dict[str, Any], output_dir: Path = FIGURES) -> Path:
    """Held-out RMSE, MAE and MASE for every candidate."""
    _style()
    scores = payload["test_all_models"]
    models = list(scores)
    labels = [MODEL_LABELS.get(name, name) for name in models]
    colors = [PALETTE["neutral"], PALETTE["actual"], PALETTE["predicted"]][: len(models)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))
    positions = np.arange(len(models))
    width = 0.38
    axes[0].bar(
        positions - width / 2,
        [scores[name]["rmse"] for name in models],
        width,
        color=colors,
        label="RMSE",
    )
    axes[0].bar(
        positions + width / 2,
        [scores[name]["mae"] for name in models],
        width,
        color=colors,
        alpha=0.55,
        hatch="//",
        label="MAE",
    )
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("USD per week")
    axes[0].yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda value, _: f"${value/1000:.0f}k")
    )
    axes[0].set_title("Held-out error, 13 weeks")
    axes[0].legend(fontsize=8, frameon=False)

    axes[1].bar(positions, [scores[name]["mase"] for name in models], color=colors)
    axes[1].axhline(1.0, color="black", ls="--", lw=1.1)
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("MASE vs seasonal naive")
    axes[1].set_title("Below 1.0 beats the naive baseline")
    for position, name in zip(positions, models, strict=True):
        value = scores[name]["mase"]
        axes[1].text(
            position,
            value + 0.02,
            f"{value:.2f}",
            ha="center",
            fontsize=9,
        )
    selected = payload["selected_model"]
    fig.suptitle(
        f"Selected on walk-forward error: {MODEL_LABELS.get(selected, selected)}",
        fontweight="bold",
    )
    return _save(fig, "19_forecast_model_comparison.png", output_dir)


def plot_backtest_folds(payload: dict[str, Any], output_dir: Path = FIGURES) -> Path:
    """RMSE per walk-forward fold, so one bad window is visible as one bad window."""
    _style()
    folds = pd.DataFrame(payload["backtest"])
    if folds.empty:
        raise ValueError("no backtest folds to plot")
    positions = np.arange(len(folds))
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.bar(
        positions - 0.26,
        folds["rmse_naive"],
        0.26,
        color=PALETTE["neutral"],
        label="Seasonal naive",
    )
    ax.bar(
        positions - 0.0,
        folds["rmse_ridge"],
        0.26,
        color=PALETTE["actual"],
        label="Ridge",
    )
    ax.bar(
        positions + 0.26,
        folds["rmse_gb"],
        0.26,
        color=PALETTE["predicted"],
        label="Gradient boosting",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels([f"Fold {int(fold)}\n{origin}" for fold, origin in zip(folds["fold"], folds["origin"], strict=True)])
    ax.set_ylabel("RMSE (USD per week)")
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda value, _: f"${value/1000:.0f}k")
    )
    ax.set_title("Walk-forward error by fold: a hard autumn, a strong spring")
    ax.legend(fontsize=8, frameon=False)
    return _save(fig, "20_backtest_folds.png", output_dir)


def plot_holdout_forecast(
    forecast: pd.DataFrame, payload: dict[str, Any], output_dir: Path = FIGURES
) -> Path:
    """Actual vs predicted for the held-out weeks, with the empirical interval."""
    _style()
    frame = forecast.sort_values("week_start")
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.fill_between(
        frame["week_start"],
        frame["lower"],
        frame["upper"],
        color=PALETTE["interval"],
        alpha=0.16,
        label="5-95% error band",
    )
    ax.plot(
        frame["week_start"],
        frame["actual"],
        color=PALETTE["actual"],
        lw=2.4,
        marker="o",
        ms=4,
        label="Actual",
    )
    ax.plot(
        frame["week_start"],
        frame["predicted"],
        color=PALETTE["predicted"],
        lw=2.0,
        ls="--",
        marker="s",
        ms=4,
        label=f"Forecast ({MODEL_LABELS.get(payload['selected_model'], payload['selected_model'])})",
    )
    ax.set_ylabel("Weekly revenue (USD)")
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda value, _: f"${value/1000:.0f}k")
    )
    shortfall = payload["test"]["bias"]
    ax.set_title(
        "The learned seasonal shape holds, but the record Q4 is under-predicted"
    )
    ax.text(
        0.01,
        0.04,
        f"Mean error {shortfall:+,.0f} USD per week",
        transform=ax.transAxes,
        fontsize=9,
        color=PALETTE["predicted"],
    )
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    return _save(fig, "21_holdout_forecast.png", output_dir)


def plot_residuals(
    forecast: pd.DataFrame, payload: dict[str, Any], output_dir: Path = FIGURES
) -> Path:
    """Residual distribution and the error grow-by-horizon pattern."""
    _style()
    frame = forecast.sort_values("horizon").copy()
    frame["residual"] = frame["predicted"] - frame["actual"]
    frame["pct_error"] = frame["residual"] / frame["actual"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    axes[0].hist(frame["residual"], bins=7, color=PALETTE["actual"], edgecolor="white")
    axes[0].axvline(0, color="black", lw=0.9)
    axes[0].axvline(
        frame["residual"].mean(),
        color=PALETTE["predicted"],
        ls="--",
        lw=1.3,
        label=f"mean {frame['residual'].mean():+,.0f}",
    )
    axes[0].set_xlabel("Forecast minus actual (USD)")
    axes[0].set_ylabel("Weeks")
    axes[0].set_title("Residuals are biased low, not centred on zero")
    axes[0].legend(fontsize=8, frameon=False)

    axes[1].plot(frame["horizon"], frame["pct_error"], marker="o", color=PALETTE["predicted"])
    axes[1].axhline(0, color="black", lw=0.9)
    axes[1].set_xlabel("Weeks ahead")
    axes[1].set_ylabel("Percentage error")
    axes[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    axes[1].set_title("Error does not grow with horizon -- the level shift hits every week")
    axes[1].set_xticks(frame["horizon"])
    return _save(fig, "22_forecast_residuals.png", output_dir)


def summarise(forecast: pd.DataFrame, payload: dict[str, Any]) -> dict[str, Any]:
    """Error detail for the report and the dashboard."""
    frame = forecast.sort_values("horizon").copy()
    frame["residual"] = frame["predicted"] - frame["actual"]
    frame["pct_error"] = frame["residual"] / frame["actual"]
    worst = frame.loc[frame["pct_error"].abs().idxmax()]
    best = frame.loc[frame["pct_error"].abs().idxmin()]
    return {
        "weeks": int(len(frame)),
        "worst_week": str(worst["week_start"].date()),
        "worst_pct_error": round(float(worst["pct_error"]), 4),
        "best_week": str(best["week_start"].date()),
        "best_pct_error": round(float(best["pct_error"]), 4),
        "mean_pct_error": round(float(frame["pct_error"].mean()), 4),
        "median_abs_pct_error": round(float(frame["pct_error"].abs().median()), 4),
        "weeks_within_20pct": int((frame["pct_error"].abs() <= 0.2).sum()),
        "interval_coverage": payload["interval_coverage"],
        "interval_nominal": 0.90,
        "under_predicted_weeks": int((frame["residual"] < 0).sum()),
        "horizon_error": {
            int(row["horizon"]): round(float(row["pct_error"]), 4)
            for _, row in frame.iterrows()
        },
    }


def generate_diagnostics(
    forecast: pd.DataFrame,
    payload: dict[str, Any],
    output_dir: Path = FIGURES,
    summary_path: Path | None = REPORTS / "diagnostics_summary.json",
) -> dict[str, Any]:
    """Render the four diagnostic figures and persist the error summary."""
    plot_model_comparison(payload, output_dir)
    plot_backtest_folds(payload, output_dir)
    plot_holdout_forecast(forecast, payload, output_dir)
    plot_residuals(forecast, payload, output_dir)
    summary = summarise(forecast, payload)
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
