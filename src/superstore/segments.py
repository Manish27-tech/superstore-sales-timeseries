"""Product segmentation: ABC concentration plus behavioural clusters.

Two complementary views, because they answer different questions:

* **ABC** answers "where does the revenue come from" -- a Pareto view that is
  easy to act on for assortment and stock decisions.
* **Clustering** answers "what kind of product is this" -- combining scale,
  growth, volatility and margin, which is what separates a stable anchor from a
  spiky one-off seller.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from superstore.config import (  # noqa: E402
    CATEGORY_COLORS,
    FIGURE_DPI,
    FIGURES,
    PALETTE,
    RANDOM_STATE,
    REPORTS,
)
from superstore.features import abc_classes, product_summary  # noqa: E402

CLUSTER_FEATURES = ("log_revenue", "growth", "volatility", "margin")
N_CLUSTERS = 4


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


def _name_clusters(profile: pd.DataFrame) -> dict[int, str]:
    """Give each cluster a plain-English name from its own statistics."""
    names: dict[int, str] = {}
    remaining = list(profile.index)
    anchor = profile["mean_revenue"].idxmax()
    names[anchor] = "Anchor products"
    remaining.remove(anchor)
    if remaining:
        growth = profile.loc[remaining, "mean_growth"].idxmax()
        names[growth] = "Rising products"
        remaining.remove(growth)
    if remaining:
        small = profile.loc[remaining, "mean_revenue"].idxmin()
        names[small] = "Long tail"
        remaining.remove(small)
    for cluster in remaining:
        names[cluster] = "Steady mid-tier"
    return names


def build_segments(product_frame: pd.DataFrame, n_clusters: int = N_CLUSTERS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach ABC classes and behavioural cluster labels to every product."""
    frame = abc_classes(product_frame).copy()
    frame["log_revenue"] = np.log1p(frame["revenue"])
    frame["growth_clipped"] = frame["growth"].clip(-0.9, 2.0)

    matrix = frame[["log_revenue", "growth_clipped", "volatility", "margin"]].to_numpy()
    scaled = StandardScaler().fit_transform(matrix)
    model = KMeans(n_clusters=n_clusters, n_init=20, random_state=RANDOM_STATE)
    frame["cluster"] = model.fit_predict(scaled)

    profile = (
        frame.groupby("cluster")
        .agg(
            products=("product_id", "size"),
            mean_revenue=("revenue", "mean"),
            total_revenue=("revenue", "sum"),
            mean_growth=("growth_clipped", "mean"),
            mean_volatility=("volatility", "mean"),
            mean_margin=("margin", "mean"),
            share_of_revenue=("revenue", lambda values: values.sum() / frame["revenue"].sum()),
        )
        .sort_values("total_revenue", ascending=False)
    )
    names = _name_clusters(profile)
    profile["segment"] = [names[cluster] for cluster in profile.index]
    frame["segment"] = frame["cluster"].map(names)
    profile = profile.reset_index()
    return frame, profile


def plot_pareto(segmented: pd.DataFrame, output_dir: Path = FIGURES) -> Path:
    _style()
    frame = segmented.sort_values("revenue", ascending=False).reset_index(drop=True)
    cumulative = frame["revenue"].cumsum() / frame["revenue"].sum()
    positions = np.arange(1, len(frame) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(positions, cumulative, color=PALETTE["actual"], lw=2)
    axes[0].axhline(0.8, color=PALETTE["predicted"], ls="--", lw=1.2)
    count_80 = int((cumulative <= 0.8).sum() + 1)
    axes[0].axvline(count_80, color=PALETTE["predicted"], ls="--", lw=1.2)
    axes[0].annotate(
        f"{count_80} products = 80% of revenue",
        xy=(count_80, 0.8),
        xytext=(count_80 + 60, 0.55),
        arrowprops={"arrowstyle": "->", "color": PALETTE["predicted"]},
        fontsize=9,
        color=PALETTE["predicted"],
    )
    axes[0].set_xlabel("Products, ranked by revenue")
    axes[0].set_ylabel("Cumulative revenue share")
    axes[0].set_title("Revenue is concentrated, but not winner-take-all")
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))

    class_summary = frame.groupby("abc_class").agg(
        products=("product_id", "size"), revenue=("revenue", "sum")
    )
    class_summary["revenue_share"] = class_summary["revenue"] / class_summary["revenue"].sum()
    class_summary["product_share"] = class_summary["products"] / class_summary["products"].sum()
    axes[1].bar(
        class_summary.index,
        class_summary["revenue_share"],
        color=[PALETTE["actual"], "#7FA6C9", PALETTE["neutral"]],
        label="Share of revenue",
    )
    axes[1].bar(
        class_summary.index,
        class_summary["product_share"],
        width=0.35,
        color=PALETTE["predicted"],
        alpha=0.85,
        label="Share of catalogue",
    )
    axes[1].set_title("A-class: 20% of the catalogue, 80% of revenue")
    axes[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    axes[1].legend(fontsize=8, frameon=False)
    return _save(fig, "14_pareto_concentration.png", output_dir)


def plot_segments(segmented: pd.DataFrame, output_dir: Path = FIGURES) -> Path:
    _style()
    frame = segmented.copy()
    upper = 2.5
    hidden = int((frame["growth"] > upper).sum())
    fig, ax = plt.subplots(figsize=(10, 5.6))
    palette = {
        "Anchor products": PALETTE["actual"],
        "Rising products": "#55A868",
        "Steady mid-tier": "#DDB75D",
        "Long tail": PALETTE["neutral"],
    }
    for segment, group in frame.groupby("segment"):
        ax.scatter(
            group["revenue"],
            group["growth"],
            s=18,
            alpha=0.65,
            label=f"{segment} (n={len(group)})",
            color=palette.get(segment, PALETTE["neutral"]),
            edgecolor="none",
        )
    ax.set_xscale("log")
    ax.set_ylim(-1.05, upper + 0.1)
    ax.axhline(0, color="black", lw=0.9)
    ax.set_xlabel("Product revenue (USD, log scale)")
    ax.set_ylabel("Growth: last 4 active months vs first 4")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_title("Four product behaviours: anchors, risers, mid-tier and the long tail")
    ax.legend(fontsize=8, frameon=False, loc="lower right", framealpha=0.9)
    if hidden:
        ax.text(
            0.01,
            0.97,
            f"{hidden} products above +250% are off-scale",
            transform=ax.transAxes,
            fontsize=8,
            color=PALETTE["neutral"],
            va="top",
        )
    return _save(fig, "15_product_segments.png", output_dir)


def plot_segment_profile(profile: pd.DataFrame, output_dir: Path = FIGURES) -> Path:
    _style()
    frame = profile.sort_values("share_of_revenue")
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9))
    axes[0].barh(frame["segment"], frame["share_of_revenue"], color=PALETTE["actual"])
    axes[0].set_title("Share of revenue")
    axes[0].xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))

    axes[1].barh(frame["segment"], frame["mean_growth"], color="#55A868")
    axes[1].axvline(0, color="black", lw=0.9)
    axes[1].set_title("Mean growth")
    axes[1].xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))

    axes[2].barh(frame["segment"], frame["mean_margin"], color="#DDB75D")
    axes[2].axvline(0, color="black", lw=0.9)
    axes[2].set_title("Mean margin")
    axes[2].xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))

    fig.suptitle("Each segment behaves differently, which is the point of splitting them", fontweight="bold")
    return _save(fig, "16_segment_profiles.png", output_dir)


def summarise(segmented: pd.DataFrame, profile: pd.DataFrame) -> dict[str, object]:
    """Segment sizes, revenue shares and the extremes worth quoting."""
    abc = segmented.groupby("abc_class").agg(
        products=("product_id", "size"), revenue=("revenue", "sum")
    )
    abc["revenue_share"] = abc["revenue"] / abc["revenue"].sum()
    risers = segmented[segmented["revenue"] >= 2_500].nlargest(5, "growth")
    return {
        "products": int(len(segmented)),
        "abc_products": {str(name): int(row["products"]) for name, row in abc.iterrows()},
        "abc_revenue_share": {
            str(name): round(float(row["revenue_share"]), 4) for name, row in abc.iterrows()
        },
        "products_for_80pct": int((segmented["cumulative_share"] <= 0.8).sum() + 1),
        "segments": {
            str(row["segment"]): {
                "products": int(row["products"]),
                "revenue_share": round(float(row["share_of_revenue"]), 4),
                "mean_growth": round(float(row["mean_growth"]), 4),
                "mean_margin": round(float(row["mean_margin"]), 4),
                "mean_volatility": round(float(row["mean_volatility"]), 4),
            }
            for _, row in profile.iterrows()
        },
        "fastest_growing": [
            {
                "product": str(row["product_name"]),
                "growth": round(float(row["growth"]), 3),
                "revenue": round(float(row["revenue"]), 2),
            }
            for _, row in risers.iterrows()
        ],
        "anchor_share": round(
            float(
                segmented.loc[segmented["segment"] == "Anchor products", "revenue"].sum()
                / segmented["revenue"].sum()
            ),
            4,
        ),
    }


def generate_segments(
    product_frame: pd.DataFrame,
    output_dir: Path = FIGURES,
    summary_path: Path | None = REPORTS / "segments_summary.json",
    table_path: Path | None = None,
) -> dict[str, object]:
    """Build segments, render three figures and persist the product table."""
    segmented, profile = build_segments(product_frame)
    plot_pareto(segmented, output_dir)
    plot_segments(segmented, output_dir)
    plot_segment_profile(profile, output_dir)
    summary = summarise(segmented, profile)
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if table_path is not None:
        table_path.parent.mkdir(parents=True, exist_ok=True)
        segmented.to_csv(table_path, index=False, date_format="%Y-%m-%d")
    return summary


def build_from_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience wrapper: order-line frame -> (segmented products, profile)."""
    return build_segments(product_summary(frame))


__all__ = [
    "build_segments",
    "build_from_frame",
    "generate_segments",
    "plot_pareto",
    "plot_segments",
    "plot_segment_profile",
    "summarise",
    "CATEGORY_COLORS",
]
