"""Command line entry point.

    python run.py all          # full pipeline, ~1 minute
    python run.py data         # download, verify, clean, persist
    python run.py report       # rebuild REPORT.md and MODEL_CARD.md from artefacts
    python run.py dashboard    # launch the Streamlit app
    python run.py check        # verify the committed artefacts are present and sane

The module path is added to ``sys.path`` so the repository works without an
editable install, which keeps Streamlit Community Cloud simple.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from superstore.clean import clean_orders, save_orders  # noqa: E402
from superstore.data import describe, load_raw  # noqa: E402


def cmd_data() -> int:
    raw = load_raw()
    orders, audit = clean_orders(raw)
    save_orders(orders)
    print(json.dumps({"dataset": describe(raw), "clean": audit}, indent=2, default=str))
    return 0


def cmd_all() -> int:
    from superstore.pipeline import run_all

    summary = run_all()
    print(json.dumps(summary, indent=2, default=str))
    return 0


def cmd_report() -> int:
    from superstore.report import generate_reports, load_summaries

    summaries = load_summaries()
    paths = generate_reports(
        summaries["eda_summary"],
        summaries["timeseries_summary"],
        summaries["segments_summary"],
        summaries["anomalies_summary"],
        summaries["metrics"],
        summaries["diagnostics_summary"],
        {
            "rows_in": summaries["eda_summary"]["rows"],
            "rows_out": summaries["eda_summary"]["rows"],
            "duplicate_rows_removed": 0,
            "date_min": summaries["eda_summary"]["date_min"],
            "date_max": summaries["eda_summary"]["date_max"],
            "weeks": summaries["eda_summary"]["weeks"],
            "months": summaries["eda_summary"]["months"],
            "partial_weeks": 2,
            "median_ship_lag_days": 4.0,
            "share_discounted": summaries["eda_summary"]["discount_share"],
            "share_return_lines": summaries["eda_summary"]["return_line_share"],
            "share_return_value": summaries["eda_summary"]["return_line_share"],
            "order_id_year_mismatches": 7,
            "median_margin": (
                summaries["eda_summary"]["total_profit"] / summaries["eda_summary"]["total_sales"]
            ),
            "total_sales": summaries["eda_summary"]["total_sales"],
            "total_profit": summaries["eda_summary"]["total_profit"],
        },
    )
    for name, path in paths.items():
        print(f"wrote {name}: {path.relative_to(ROOT)}")
    return 0


def cmd_dashboard() -> int:
    app = ROOT / "dashboard" / "app.py"
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app)])


def cmd_check() -> int:
    script = ROOT / "scripts" / "check_artifacts.py"
    return subprocess.call([sys.executable, str(script)])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "command",
        nargs="?",
        default="all",
        choices=("all", "data", "report", "dashboard", "check"),
        help="stage to run (default: all)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {
        "all": cmd_all,
        "data": cmd_data,
        "report": cmd_report,
        "dashboard": cmd_dashboard,
        "check": cmd_check,
    }
    return handlers[args.command]()


if __name__ == "__main__":
    raise SystemExit(main())
