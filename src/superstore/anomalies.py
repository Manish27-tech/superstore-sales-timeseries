"""Anomaly detection on the weekly series.

A week is anomalous when it deviates from the same week a year earlier by more
than a robust threshold. The median absolute deviation is used rather than a
standard deviation because a handful of very large orders would otherwise
inflate the threshold and hide themselves.

Every flagged week is then explained: which category moved, and whether a single
order line accounts for the gap. On this dataset most spikes are one big order,
which is a much more useful finding than the flag itself.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from superstore.config import FIGURE_DPI, FIGURES, PALETTE, PROFIT, REPORTS, TARGET  # noqa: E402

Z_THRESHOLD = 3.5


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


def detect_weekly_anomalies(
    weekly: pd.DataFrame,
    threshold: float = Z_THRESHOLD,
    min_expected: float = 2_000.0,
    min_deviation: float = 3_000.0,
) -> pd.DataFrame:
    """Robust z-scores against the same week one year earlier.

    Two business-significance guards sit on top of the statistics. A week whose
    year-earlier revenue was only a few hundred dollars can produce a huge ratio
    without being a real event, so a week must have a plausible baseline
    (``min_expected``) and a material absolute gap (``min_deviation``) before it
    can be flagged.
    """
    frame = weekly.copy().reset_index(drop=True)
    frame["expected"] = frame["revenue"].shift(52)
    frame["ratio"] = frame["revenue"] / frame["expected"]

    valid = frame["ratio"].dropna()
    centre = float(valid.median())
    mad = float((valid - centre).abs().median())
    scale = 1.4826 * mad if mad > 0 else float(valid.std())
    frame["z_score"] = (frame["ratio"] - centre) / scale
    frame["deviation"] = frame["revenue"] - frame["expected"]

    eligible = (frame["expected"] >= min_expected) & (
        frame["deviation"].abs() >= min_deviation
    )
    frame["direction"] = np.where(
        eligible & (frame["z_score"] >= threshold), "spike", "normal"
    )
    frame.loc[eligible & (frame["z_score"] <= -threshold), "direction"] = "dip"
    frame["is_anomaly"] = frame["direction"] != "normal"
    frame["eligible"] = eligible
    return frame


def explain_anomalies(frame: pd.DataFrame, anomalies: pd.DataFrame) -> pd.DataFrame:
    """Attach the dominant category and largest order line to each anomaly."""
    rows: list[dict[str, object]] = []
    for _, row in anomalies.iterrows():
        week_start = row["week_start"]
        week_end = week_start + pd.Timedelta(days=6)
        week = frame[(frame["Week Start"] == week_start)]
        category = week.groupby("Category")[TARGET].sum().sort_values(ascending=False)
        largest = week.nlargest(1, TARGET)
        rows.append(
            {
                "week_start": week_start,
                "week_end": week_end,
                "revenue": round(float(row["revenue"]), 2),
                "expected": round(float(row["expected"]), 2),
                "deviation": round(float(row["deviation"]), 2),
                "z_score": round(float(row["z_score"]), 2),
                "direction": row["direction"],
                "top_category": str(category.index[0]) if not category.empty else "",
                "top_category_share": round(
                    float(category.iloc[0] / week[TARGET].sum()), 4
                )
                if not week.empty
                else 0.0,
                "largest_order_value": round(float(largest[TARGET].iloc[0]), 2)
                if not largest.empty
                else 0.0,
                "largest_order_product": str(largest["Product Name"].iloc[0])
                if not largest.empty
                else "",
                "largest_order_share": round(
                    float(largest[TARGET].iloc[0] / week[TARGET].sum()), 4
                )
                if not week.empty
                else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("z_score", ascending=False).reset_index(drop=True)


def plot_anomaly_timeline(
    scored: pd.DataFrame, anomalies: pd.DataFrame, output_dir: Path = FIGURES
) -> Path:
    _style()
    fig, ax = plt.subplots(figsize=(11.5, 4.4))
    ax.plot(scored["week_start"], scored["revenue"], color=PALETTE["actual"], lw=1.2, label="Weekly revenue")
    spikes = anomalies[anomalies["direction"] == "spike"]
    dips = anomalies[anomalies["direction"] == "dip"]
    ax.scatter(spikes["week_start"], spikes["revenue"], color=PALETTE["predicted"], s=42, zorder=3, label="Spike")
    ax.scatter(dips["week_start"], dips["revenue"], color="#7B4B94", s=42, zorder=3, label="Dip")
    for _, row in anomalies.nlargest(2, "z_score").iterrows():
        ax.annotate(
            row["week_start"].strftime("%b %Y"),
            xy=(row["week_start"], row["revenue"]),
            xytext=(10, 14),
            textcoords="offset points",
            fontsize=8,
            color=PALETTE["predicted"],
        )
    ax.set_ylabel("Weekly revenue (USD)")
    ax.set_title("Anomalies are rare and all upward, each traced to a single large order")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    return _save(fig, "17_anomaly_timeline.png", output_dir)


def plot_anomaly_explanations(anomalies: pd.DataFrame, output_dir: Path = FIGURES) -> Path:
    _style()
    frame = anomalies.reindex(anomalies["z_score"].abs().sort_values(ascending=False).index).head(8)
    labels = [date.strftime("%d %b %Y") for date in frame["week_start"]]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    colors = [PALETTE["predicted"] if z > 0 else "#7B4B94" for z in frame["z_score"]]
    axes[0].barh(labels[::-1], frame["z_score"][::-1], color=colors[::-1])
    axes[0].axvline(0, color="black", lw=0.9)
    axes[0].set_xlabel("Robust z-score vs same week last year")
    axes[0].set_title(f"The {len(frame)} flagged weeks, ranked by deviation")

    axes[1].scatter(
        frame["largest_order_share"],
        frame["z_score"],
        s=60,
        color=[PALETTE["predicted"] if z > 0 else "#7B4B94" for z in frame["z_score"]],
    )
    for _, row in frame.iterrows():
        axes[1].annotate(
            row["week_start"].strftime("%b %y"),
            xy=(row["largest_order_share"], row["z_score"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7.5,
        )
    axes[1].axhline(0, color="black", lw=0.9)
    axes[1].set_xlabel("Share of the week's revenue from its largest order line")
    axes[1].set_ylabel("Robust z-score")
    axes[1].xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    axes[1].set_title("Big spikes and big single lines move together")
    return _save(fig, "18_anomaly_explanations.png", output_dir)


def summarise(scored: pd.DataFrame, anomalies: pd.DataFrame) -> dict[str, object]:
    """Anomaly counts plus the detail a reader needs to trust them."""
    return {
        "weeks_scored": int(scored["z_score"].notna().sum()),
        "eligible_weeks": int(scored["eligible"].sum()),
        "threshold": Z_THRESHOLD,
        "min_expected": 2_000.0,
        "min_deviation": 3_000.0,
        "anomalies": int(len(anomalies)),
        "spikes": int((anomalies["direction"] == "spike").sum()),
        "dips": int((anomalies["direction"] == "dip").sum()),
        "share_of_weeks": round(
            float(len(anomalies) / max(int(scored["eligible"].sum()), 1)), 4
        ),
        "largest_single_order_share": round(
            float(anomalies["largest_order_share"].max()), 4
        )
        if not anomalies.empty
        else 0.0,
        "top_anomalies": [
            {
                "week_start": str(row["week_start"].date()),
                "direction": str(row["direction"]),
                "z_score": round(float(row["z_score"]), 2),
                "revenue": round(float(row["revenue"]), 2),
                "expected": round(float(row["expected"]), 2),
                "largest_order_value": round(float(row["largest_order_value"]), 2),
                "largest_product": str(row["largest_order_product"]),
            }
            for _, row in anomalies.head(5).iterrows()
        ],
    }


def generate_anomalies(
    frame: pd.DataFrame,
    weekly: pd.DataFrame,
    output_dir: Path = FIGURES,
    summary_path: Path | None = REPORTS / "anomalies_summary.json",
    table_path: Path | None = None,
) -> dict[str, object]:
    """Score, explain, render and persist the anomaly analysis."""
    scored = detect_weekly_anomalies(weekly)
    anomalies = explain_anomalies(frame, scored[scored["is_anomaly"]])
    plot_anomaly_timeline(scored, anomalies, output_dir)
    plot_anomaly_explanations(anomalies, output_dir)
    summary = summarise(scored, anomalies)
    if table_path is not None:
        table_path.parent.mkdir(parents=True, exist_ok=True)
        anomalies.to_csv(table_path, index=False, date_format="%Y-%m-%d")
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


__all__ = [
    "Z_THRESHOLD",
    "detect_weekly_anomalies",
    "explain_anomalies",
    "generate_anomalies",
    "plot_anomaly_explanations",
    "plot_anomaly_timeline",
    "summarise",
    "PROFIT",
]
