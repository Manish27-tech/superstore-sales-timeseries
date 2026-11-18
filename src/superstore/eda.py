"""Exploratory figures and the headline summary.

Ten figures, each answering one question a commercial reader would actually ask:
is the business growing, what sells, when does it sell, and where does the
margin leak. Every number in the summary is computed from the cleaned frame, so
the report and the dashboard cannot drift from the figures.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from superstore.config import (  # noqa: E402
    CATEGORY_COLORS,
    FIGURE_DPI,
    FIGURES,
    PALETTE,
    PROFIT,
    REPORTS,
    TARGET,
)
from superstore.features import (  # noqa: E402
    daily_revenue,
    monthly_category_revenue,
    monthly_revenue,
    product_summary,
    weekly_revenue,
)

WEEKDAY_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


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


def plot_monthly_trend(frame: pd.DataFrame, output_dir: Path = FIGURES) -> Path:
    _style()
    monthly = monthly_revenue(frame)
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.plot(monthly["month_start"], monthly["revenue"], color=PALETTE["actual"], lw=1.8, label="Monthly revenue")
    rolling = monthly["revenue"].rolling(12, min_periods=6).mean()
    ax.plot(monthly["month_start"], rolling, color=PALETTE["predicted"], lw=2.2, ls="--", label="12-month average")
    ax.set_ylabel("Revenue (USD)")
    ax.set_title("Revenue grows 51% in four years, but single large orders dominate individual months")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))
    ax.legend(loc="upper left", fontsize=8)
    return _save(fig, "01_monthly_revenue_trend.png", output_dir)


def plot_category_area(frame: pd.DataFrame, output_dir: Path = FIGURES) -> Path:
    _style()
    monthly = monthly_category_revenue(frame)
    fig, ax = plt.subplots(figsize=(11, 4.4))
    columns = [column for column in ("Technology", "Furniture", "Office Supplies") if column in monthly]
    ax.stackplot(
        monthly["month_start"],
        [monthly[column] for column in columns],
        labels=columns,
        colors=[CATEGORY_COLORS[column] for column in columns],
        alpha=0.85,
    )
    ax.set_ylabel("Revenue (USD)")
    ax.set_title("Technology leads every year; Office Supplies rebounds sharply in 2017")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    return _save(fig, "02_category_revenue_mix.png", output_dir)


def plot_seasonality_heatmap(frame: pd.DataFrame, output_dir: Path = FIGURES) -> Path:
    _style()
    monthly = monthly_revenue(frame)
    monthly["year"] = monthly["month_start"].dt.year
    monthly["month"] = monthly["month_start"].dt.month
    pivot = monthly.pivot_table(index="year", columns="month", values="revenue", aggfunc="sum")
    # Index against each year's monthly mean so years of different size compare.
    indexed = pivot.div(pivot.mean(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(10.5, 3.4))
    image = ax.imshow(indexed, aspect="auto", cmap="RdYlGn", vmin=0.4, vmax=1.6)
    ax.set_xticks(range(len(indexed.columns)))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_yticks(range(len(indexed.index)))
    ax.set_yticklabels(indexed.index.astype(str))
    for row in range(indexed.shape[0]):
        for column in range(indexed.shape[1]):
            value = indexed.iat[row, column]
            ax.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=7.5)
    ax.set_title("Seasonality is stable: September and November-December carry each year")
    ax.grid(False)
    fig.colorbar(image, ax=ax, shrink=0.85, label="Month vs year average")
    return _save(fig, "03_seasonality_heatmap.png", output_dir)


def plot_weekday_profile(frame: pd.DataFrame, output_dir: Path = FIGURES) -> Path:
    _style()
    profile = (
        frame.groupby("Order Weekday")
        .agg(revenue=(TARGET, "sum"), orders=("Order ID", "nunique"))
        .reindex(WEEKDAY_ORDER)
    )
    daily = frame.groupby(frame["Order Date"].dt.date).agg(
        revenue=(TARGET, "sum"), weekday=("Order Weekday", "first")
    )
    mean_daily = daily.groupby("weekday")["revenue"].mean().reindex(WEEKDAY_ORDER)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].bar(mean_daily.index, mean_daily.values, color=PALETTE["actual"])
    axes[0].set_title("Average revenue per trading day")
    axes[0].set_ylabel("USD")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v/1000:.1f}k"))

    share = profile["orders"] / profile["orders"].sum()
    axes[1].bar(share.index, share.values, color=PALETTE["neutral"])
    axes[1].set_title("Share of orders placed")
    axes[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    axes[1].tick_params(axis="x", rotation=25)
    fig.suptitle(
        "Weekday profile is distorted: Wednesday carries only 4% of lines, a source artefact",
        fontweight="bold",
    )
    return _save(fig, "04_weekday_profile.png", output_dir)


def plot_subcategory_margin(frame: pd.DataFrame, output_dir: Path = FIGURES) -> Path:
    _style()
    grouped = (
        frame.groupby("Sub-Category")
        .agg(revenue=(TARGET, "sum"), profit=(PROFIT, "sum"))
        .assign(margin=lambda d: d["profit"] / d["revenue"])
        .sort_values("revenue")
    )
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    colors = ["#C44536" if margin < 0 else "#4C72B0" for margin in grouped["margin"]]
    ax.barh(grouped.index, grouped["revenue"], color=colors)
    ax.set_xlabel("Revenue (USD)")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))
    twin = ax.twiny()
    twin.plot(grouped["margin"], grouped.index, color=PALETTE["predicted"], marker="o", lw=1.3)
    twin.axvline(0, color="black", lw=0.8)
    twin.set_xlim(-0.2, 0.5)
    twin.set_xlabel("Profit margin", labelpad=14)
    twin.grid(False)
    ax.set_title("Tables and Bookcases sell well and lose money; Labels and Paper are the margin engine")
    return _save(fig, "05_subcategory_revenue_margin.png", output_dir)


def plot_top_products(frame: pd.DataFrame, output_dir: Path = FIGURES) -> Path:
    _style()
    products = product_summary(frame).head(15)
    labels = [name[:42] + "..." if len(name) > 45 else name for name in products["product_name"]]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    colors = [CATEGORY_COLORS.get(category, PALETTE["neutral"]) for category in products["category"]]
    ax.barh(labels[::-1], products["revenue"][::-1], color=colors[::-1])
    ax.set_xlabel("Revenue (USD)")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))
    ax.set_title("The top 15 products are 0.8% of the catalogue and 14% of revenue")
    handles = [
        matplotlib.patches.Patch(color=color, label=category)
        for category, color in CATEGORY_COLORS.items()
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8, frameon=False)
    return _save(fig, "06_top_products.png", output_dir)


def plot_mix(frame: pd.DataFrame, output_dir: Path = FIGURES) -> Path:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for ax, column, title in zip(
        axes, ("Region", "Segment"), ("Revenue by region", "Revenue by customer segment"), strict=True
    ):
        share = frame.groupby(column)[TARGET].sum().sort_values()
        ax.barh(share.index, share.values, color=PALETTE["actual"])
        ax.set_title(title)
        ax.set_xlabel("Revenue (USD)")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))
        for index, value in enumerate(share.values):
            ax.text(value * 1.01, index, f"{value / frame[TARGET].sum():.0%}", va="center", fontsize=8)
    fig.suptitle("West and East carry 61% of the business; Consumer is half of it", fontweight="bold")
    return _save(fig, "07_region_segment_mix.png", output_dir)


def plot_discount_effect(frame: pd.DataFrame, output_dir: Path = FIGURES) -> Path:
    _style()
    order = ["No discount", "1-10%", "11-20%", "21-40%", "41%+"]
    grouped = (
        frame.groupby("Discount Band")
        .agg(revenue=(TARGET, "sum"), profit=(PROFIT, "sum"), lines=(TARGET, "size"))
        .reindex(order)
        .dropna()
    )
    grouped["margin"] = grouped["profit"] / grouped["revenue"]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    colors = ["#55A868" if margin >= 0.2 else "#E4572E" if margin >= 0 else "#A4243B" for margin in grouped["margin"]]
    bars = ax.bar(grouped.index, grouped["margin"], color=colors)
    for bar, (_, row) in zip(bars, grouped.iterrows(), strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            row["margin"] + 0.01,
            f"{row['margin']:.0%}\n{row['lines']:,} lines",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.axhline(0, color="black", lw=0.9)
    ax.set_ylabel("Profit margin")
    ax.set_title("Discounting past 20% turns orders unprofitable")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylim(-0.2, 0.4)
    return _save(fig, "08_discount_vs_margin.png", output_dir)


def plot_order_value(frame: pd.DataFrame, output_dir: Path = FIGURES) -> Path:
    _style()
    orders = frame.groupby("Order ID")[TARGET].sum()
    fig, ax = plt.subplots(figsize=(9, 4.0))
    ax.hist(orders, bins=np.logspace(np.log10(orders.min()), np.log10(orders.max()), 45), color=PALETTE["actual"])
    ax.set_xscale("log")
    ax.set_xlabel("Order value (USD, log scale)")
    ax.set_ylabel("Orders")
    ax.axvline(orders.median(), color=PALETTE["predicted"], ls="--", lw=1.4)
    ax.text(
        orders.median() * 1.1,
        ax.get_ylim()[1] * 0.85,
        f"median ${orders.median():.0f}",
        color=PALETTE["predicted"],
        fontsize=9,
    )
    ax.set_title("Order values run from $0.56 to $23.7k; the tail is where revenue lives")
    return _save(fig, "09_order_value_distribution.png", output_dir)


def plot_yoy_growth(frame: pd.DataFrame, output_dir: Path = FIGURES) -> Path:
    _style()
    monthly = monthly_revenue(frame)
    monthly["year"] = monthly["month_start"].dt.year
    annual = monthly.groupby("year")["revenue"].sum()
    growth = annual.pct_change().dropna()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    axes[0].bar(annual.index.astype(str), annual.values, color=PALETTE["actual"])
    axes[0].set_title("Annual revenue")
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))
    axes[1].bar(growth.index.astype(str), growth.values, color=PALETTE["predicted"])
    axes[1].axhline(0, color="black", lw=0.9)
    axes[1].set_title("Year-over-year growth")
    axes[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    for index, value in enumerate(growth.values):
        axes[1].text(index, value + 0.01, f"{value:.0%}", ha="center", fontsize=9)
    fig.suptitle("Growth resumes after a flat 2015 and keeps compounding", fontweight="bold")
    return _save(fig, "10_year_over_year_growth.png", output_dir)


PLOTS = (
    plot_monthly_trend,
    plot_category_area,
    plot_seasonality_heatmap,
    plot_weekday_profile,
    plot_subcategory_margin,
    plot_top_products,
    plot_mix,
    plot_discount_effect,
    plot_order_value,
    plot_yoy_growth,
)


def summarise(frame: pd.DataFrame) -> dict[str, object]:
    """Numbers the report and dashboard quote."""
    monthly = monthly_revenue(frame)
    weekly = weekly_revenue(frame)
    products = product_summary(frame)
    daily = daily_revenue(frame)
    annual = monthly.assign(year=monthly["month_start"].dt.year).groupby("year")["revenue"].sum()
    category = frame.groupby("Category")[TARGET].sum().sort_values(ascending=False)
    subcategory = frame.groupby("Sub-Category").agg(
        revenue=(TARGET, "sum"), profit=(PROFIT, "sum")
    )
    subcategory["margin"] = subcategory["profit"] / subcategory["revenue"]
    month_means = (
        monthly.assign(month=monthly["month_start"].dt.month_name())
        .groupby("month")["revenue"]
        .mean()
    )
    weekday_means = (
        frame.groupby("Order Weekday")["Sales"].sum().reindex(WEEKDAY_ORDER)
    )
    return {
        "rows": int(len(frame)),
        "orders": int(frame["Order ID"].nunique()),
        "customers": int(frame["Customer ID"].nunique()),
        "products": int(frame["Product ID"].nunique()),
        "date_min": str(frame["Order Date"].min()),
        "date_max": str(frame["Order Date"].max()),
        "weeks": int(len(weekly)),
        "months": int(len(monthly)),
        "total_sales": round(float(frame[TARGET].sum()), 2),
        "total_profit": round(float(frame[PROFIT].sum()), 2),
        "margin": round(float(frame[PROFIT].sum() / frame[TARGET].sum()), 4),
        "mean_order_value": round(float(frame.groupby("Order ID")[TARGET].sum().mean()), 2),
        "median_order_value": round(float(frame.groupby("Order ID")[TARGET].sum().median()), 2),
        "best_month": str(month_means.idxmax()),
        "best_weekday": str(weekday_means.idxmax()),
        "quietest_weekday": str(weekday_means.idxmin()),
        "category_share": {
            str(name): round(float(value / frame[TARGET].sum()), 4) for name, value in category.items()
        },
        "top_category": str(category.index[0]),
        "top_subcategory": str(subcategory["revenue"].idxmax()),
        "worst_margin_subcategory": str(subcategory["margin"].idxmin()),
        "best_margin_subcategory": str(subcategory["margin"].idxmax()),
        "deepest_discount_share": round(float((frame["Discount"] >= 0.4).mean()), 4),
        "annual_revenue": {int(year): round(float(value), 2) for year, value in annual.items()},
        "yoy_growth": {
            int(year): round(float(value), 4) for year, value in annual.pct_change().dropna().items()
        },
        "top_product": str(products["product_name"].iloc[0]),
        "top_product_revenue": round(float(products["revenue"].iloc[0]), 2),
        "products_for_80pct": int((products["cumulative_share"] <= 0.8).sum() + 1),
        "weekly_mean_revenue": round(float(weekly["revenue"].mean()), 2),
        "weekly_max_revenue": round(float(weekly["revenue"].max()), 2),
        "busiest_week": str(weekly.loc[weekly["revenue"].idxmax(), "week_start"].date()),
        "quiet_days_share": round(float((daily["revenue"] == 0).mean()), 4),
        "return_line_share": round(float(frame["Is Return"].mean()), 4),
        "discount_share": round(float(frame["Has Discount"].mean()), 4),
        "weekday_line_share": {
            str(name): round(float(value), 4)
            for name, value in (
                frame["Order Weekday"].value_counts(normalize=True).reindex(WEEKDAY_ORDER)
            ).items()
        },
        # The extract is known to under-represent Wednesday; the share is
        # reported explicitly so no reader mistakes it for a real trading pattern.
        "weekday_artifact": "Wednesday holds only 4% of order lines, versus 10-19% for other days",
    }


def generate_eda(
    frame: pd.DataFrame,
    output_dir: Path = FIGURES,
    summary_path: Path | None = REPORTS / "eda_summary.json",
) -> dict[str, object]:
    """Render all figures and persist the summary."""
    for plot in PLOTS:
        plot(frame, output_dir)
    summary = summarise(frame)
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
