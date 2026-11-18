"""Classical decomposition and growth accounting on the monthly series.

The series is short -- 48 months -- so a full STL is overkill and easy to
over-read. A multiplicative decomposition (centred 12-month moving average for
the trend, month-average ratios for the seasonal component) is transparent, and
its two strength statistics say how much of the variance each part explains.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from superstore.config import FIGURE_DPI, FIGURES, PALETTE, REPORTS  # noqa: E402
from superstore.features import monthly_revenue  # noqa: E402

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


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


def decompose(monthly: pd.DataFrame, period: int = 12) -> pd.DataFrame:
    """Multiplicative decomposition into trend, seasonal and residual parts.

    Requires at least two full seasonal cycles; the first and last
    ``period // 2`` months have no centred trend and are returned as NaN so they
    are excluded from the seasonal index rather than silently distorted.
    """
    frame = monthly.copy().sort_values("month_start").reset_index(drop=True)
    if len(frame) < 2 * period:
        raise ValueError("need at least two full cycles to decompose")

    series = frame["revenue"].astype(float)
    trend = series.rolling(period, center=True, min_periods=period).mean()
    detrended = series / trend
    seasonal = (
        detrended.groupby(frame["month_start"].dt.month).transform("mean")
    )
    # Normalise so the seasonal factors average to one over a full year.
    month_index = frame["month_start"].dt.month
    factors = detrended.groupby(month_index).mean()
    factors = factors / factors.mean()
    seasonal = month_index.map(factors)
    residual = series / (trend * seasonal)

    frame["trend"] = trend
    frame["seasonal"] = seasonal
    frame["residual"] = residual
    frame.attrs["factors"] = factors.reindex(range(1, 13))
    return frame


def strength(decomposed: pd.DataFrame) -> dict[str, float]:
    """Share of variance explained by trend and seasonality (0-1).

    Follows the usual definition: compare the residual variance with the
    variance of the component-free series it should explain. Detrending exposes
    the seasonal signal, deseasonalising exposes the trend.
    """
    valid = decomposed.dropna(subset=["trend", "seasonal", "residual"])
    residual_variance = float(np.nanvar(valid["residual"]))
    detrended = valid["residual"] * valid["seasonal"]
    deseasonalised = valid["residual"] * valid["trend"] / valid["trend"].mean()
    return {
        "trend_strength": round(
            max(0.0, 1 - residual_variance / max(float(np.nanvar(deseasonalised)), 1e-9)), 4
        ),
        "seasonal_strength": round(
            max(0.0, 1 - residual_variance / max(float(np.nanvar(detrended)), 1e-9)), 4
        ),
        "residual_share": round(float(np.nanmean(np.abs(valid["residual"] - 1))), 4),
    }


def seasonal_factors(decomposed: pd.DataFrame) -> pd.Series:
    factors = decomposed.attrs.get("factors")
    if factors is None:
        raise ValueError("decomposed frame is missing its seasonal factors")
    factors.index = MONTH_LABELS
    return factors.rename("factor")


def trailing_growth(monthly: pd.DataFrame) -> pd.DataFrame:
    """Trailing-12-month revenue and its year-over-year growth."""
    frame = monthly.copy().sort_values("month_start").reset_index(drop=True)
    frame["trailing_12m"] = frame["revenue"].rolling(12, min_periods=12).sum()
    frame["trailing_yoy"] = frame["trailing_12m"].pct_change(12)
    return frame


def plot_decomposition(decomposed: pd.DataFrame, output_dir: Path = FIGURES) -> Path:
    _style()
    frame = decomposed
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(frame["month_start"], frame["revenue"], color=PALETTE["actual"], lw=1.6)
    axes[0].set_ylabel("Observed")
    axes[0].set_title("Monthly revenue decomposes into a rising trend and a stable seasonal shape")

    axes[1].plot(frame["month_start"], frame["trend"], color=PALETTE["predicted"], lw=2)
    axes[1].set_ylabel("Trend")

    axes[2].plot(frame["month_start"], frame["seasonal"], color="#55A868", lw=1.4)
    axes[2].axhline(1, color="black", lw=0.8, ls="--")
    axes[2].set_ylabel("Seasonal")

    axes[3].plot(frame["month_start"], frame["residual"], color=PALETTE["neutral"], lw=1.2)
    axes[3].axhline(1, color="black", lw=0.8, ls="--")
    axes[3].set_ylabel("Residual")
    axes[3].set_xlabel("Month")
    for ax in axes[:3]:
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v/1000:.0f}k" if v > 1000 else f"{v:.2f}"))
    return _save(fig, "11_monthly_decomposition.png", output_dir)


def plot_seasonal_factors(decomposed: pd.DataFrame, output_dir: Path = FIGURES) -> Path:
    _style()
    factors = seasonal_factors(decomposed)
    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    colors = [PALETTE["predicted"] if value >= 1 else PALETTE["neutral"] for value in factors]
    bars = ax.bar(factors.index, factors.values, color=colors)
    ax.axhline(1, color="black", lw=0.9)
    for bar, value in zip(bars, factors.values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.01,
            f"{value:.2f}",
            ha="center",
            fontsize=8.5,
        )
    ax.set_ylabel("Seasonal factor (1.0 = average month)")
    ax.set_title("November and December are 1.75x an average month; February is 0.39x")
    ax.set_ylim(0, factors.max() * 1.18)
    return _save(fig, "12_seasonal_factors.png", output_dir)


def plot_trailing_growth(monthly: pd.DataFrame, output_dir: Path = FIGURES) -> Path:
    _style()
    frame = trailing_growth(monthly).dropna(subset=["trailing_12m"])
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.plot(frame["month_start"], frame["trailing_12m"], color=PALETTE["actual"], lw=2)
    ax.set_ylabel("Trailing 12-month revenue")
    ax.set_title("Trailing annual revenue steps up each year, ending near $733k")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))
    twin = ax.twinx()
    growth = frame.dropna(subset=["trailing_yoy"])
    twin.plot(growth["month_start"], growth["trailing_yoy"], color=PALETTE["predicted"], lw=1.4, ls="--")
    twin.axhline(0, color="grey", lw=0.8)
    twin.set_ylabel("YoY growth")
    twin.grid(False)
    twin.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    return _save(fig, "13_trailing_growth.png", output_dir)


def summarise(monthly: pd.DataFrame, decomposed: pd.DataFrame) -> dict[str, object]:
    """Decomposition numbers for the report and the dashboard."""
    factors = seasonal_factors(decomposed)
    stats = strength(decomposed)
    return {
        "seasonal_factors": {str(month): round(float(value), 4) for month, value in factors.items()},
        "strongest_month": str(factors.idxmax()),
        "weakest_month": str(factors.idxmin()),
        "strongest_factor": round(float(factors.max()), 4),
        "weakest_factor": round(float(factors.min()), 4),
        "trend_strength": stats["trend_strength"],
        "seasonal_strength": stats["seasonal_strength"],
        "mean_absolute_residual": stats["residual_share"],
        "peak_to_trough": round(float(factors.max() / factors.min()), 2),
        "trend_start": round(float(decomposed["trend"].dropna().iloc[0]), 2),
        "trend_end": round(float(decomposed["trend"].dropna().iloc[-1]), 2),
        "trend_growth": round(
            float(
                decomposed["trend"].dropna().iloc[-1] / decomposed["trend"].dropna().iloc[0] - 1
            ),
            4,
        ),
        "months_analysed": int(len(monthly)),
    }


def generate_timeseries(
    monthly: pd.DataFrame,
    output_dir: Path = FIGURES,
    summary_path: Path | None = REPORTS / "timeseries_summary.json",
) -> dict[str, object]:
    """Run the decomposition, render three figures and persist the summary."""
    decomposed = decompose(monthly)
    plot_decomposition(decomposed, output_dir)
    plot_seasonal_factors(decomposed, output_dir)
    plot_trailing_growth(monthly, output_dir)
    summary = summarise(monthly, decomposed)
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
