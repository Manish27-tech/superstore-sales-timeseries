"""The end-to-end pipeline, one function per stage.

Running the whole thing takes about a minute on a laptop. Every stage writes its
artefact before the next stage reads it, so a failure leaves a usable partial
state and `python run.py all` is safe to rerun at any point.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from superstore import anomalies, diagnostics, eda, export, forecast, report, segments, timeseries
from superstore.clean import clean_orders, save_orders
from superstore.config import DATA_DASHBOARD, REPORTS
from superstore.data import describe, load_raw
from superstore.features import monthly_revenue, product_summary, weekly_revenue


def _timed(label: str, function: Callable[[], Any]) -> tuple[Any, float]:
    start = time.perf_counter()
    print(f"[pipeline] {label} ...", flush=True)
    result = function()
    elapsed = time.perf_counter() - start
    print(f"[pipeline] {label} done in {elapsed:.1f}s", flush=True)
    return result, elapsed


def build_kpis(
    eda_summary: dict[str, Any],
    timeseries_summary: dict[str, Any],
    segments_summary: dict[str, Any],
    anomalies_summary: dict[str, Any],
    metrics: dict[str, Any],
    diagnostics_summary: dict[str, Any],
) -> dict[str, Any]:
    """One JSON object with everything the dashboard header and about tab need."""
    return {
        "sales": eda_summary,
        "timeseries": timeseries_summary,
        "segments": segments_summary,
        "anomalies": anomalies_summary,
        "forecast": {
            "selected_model": metrics["selected_model"],
            "selection_scores": metrics["selection_scores"],
            "test": metrics["test"],
            "test_all_models": metrics["test_all_models"],
            "improvement_vs_naive_pct": metrics["improvement_vs_naive_pct"],
            "split": metrics["split"],
            "interval_coverage": metrics["interval_coverage"],
            "feature_importance": metrics["feature_importance"],
            "diagnostics": diagnostics_summary,
        },
        "generated_by": "superstore-sales-timeseries",
    }


def run_all(with_report: bool = True) -> dict[str, Any]:
    """Run every stage and return a summary of the outcomes."""
    timings: dict[str, float] = {}

    raw, timings["load"] = _timed("load and verify the pinned dataset", load_raw)
    cleaned, timings["clean"] = _timed(
        "clean and derive calendar fields", lambda: clean_orders(raw)
    )
    orders, audit = cleaned
    save_orders(orders)

    eda_summary, timings["eda"] = _timed("exploratory figures", lambda: eda.generate_eda(orders))

    monthly = monthly_revenue(orders)
    timeseries_summary, timings["timeseries"] = _timed(
        "decomposition", lambda: timeseries.generate_timeseries(monthly)
    )

    products = product_summary(orders)
    start = time.perf_counter()
    print("[pipeline] product segmentation ...", flush=True)
    segmented_frame, _profile = segments.build_segments(products)
    segments_summary = segments.generate_segments(
        products, table_path=REPORTS / "products.csv"
    )
    timings["segments"] = round(time.perf_counter() - start, 2)
    print(f"[pipeline] product segmentation done in {timings['segments']:.1f}s", flush=True)

    weekly = weekly_revenue(orders)
    anomalies_summary, timings["anomalies"] = _timed(
        "weekly anomalies",
        lambda: anomalies.generate_anomalies(
            orders, weekly, table_path=REPORTS / "anomalies.csv"
        ),
    )

    metrics, timings["forecast"] = _timed(
        "13-week forecast", lambda: forecast.train_and_evaluate(weekly)
    )

    scored_forecast = pd.read_csv(REPORTS / "forecast.csv", parse_dates=["week_start"])
    diagnostics_summary, timings["diagnostics"] = _timed(
        "forecast diagnostics",
        lambda: diagnostics.generate_diagnostics(scored_forecast, metrics),
    )

    kpis = build_kpis(
        eda_summary, timeseries_summary, segments_summary, anomalies_summary, metrics, diagnostics_summary
    )
    anomaly_table = pd.read_csv(REPORTS / "anomalies.csv")
    _, timings["export"] = _timed(
        "dashboard feed",
        lambda: export.export_tables(
            orders,
            segmented_frame,
            scored_forecast,
            anomaly_table,
            kpis,
            output_dir=DATA_DASHBOARD,
            metrics=metrics,
        ),
    )

    if with_report:
        _, timings["report"] = _timed(
            "report and model card",
            lambda: report.generate_reports(
                eda_summary,
                timeseries_summary,
                segments_summary,
                anomalies_summary,
                metrics,
                diagnostics_summary,
                audit,
            ),
        )

    total = sum(timings.values())
    print(f"[pipeline] all stages complete in {total:.1f}s", flush=True)
    return {
        "timings": {name: round(value, 2) for name, value in timings.items()},
        "total_seconds": round(total, 2),
        "rows": int(len(orders)),
        "weekly_periods": int(len(weekly)),
        "forecast_rmse": metrics["test"]["rmse"],
        "forecast_model": metrics["selected_model"],
        "dashboard_files": len(list(Path(DATA_DASHBOARD).glob("*"))),
        "dataset": describe(raw),
    }
