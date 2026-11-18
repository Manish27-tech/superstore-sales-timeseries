"""Forecast tab: a 13-week-ahead model, its error, and an honest reading of both."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dashboard.theme import GREY, NAVY, ORANGE, PURPLE, apply_layout, fmt_money, fmt_pct, kpi_cards

MODEL_LABELS = {
    "seasonal_naive": "Seasonal naive",
    "ridge_regression": "Ridge",
    "gradient_boosting": "Gradient boosting",
}


def render(feed: dict) -> None:
    payload = feed["kpis"]["forecast"]
    forecast = feed["forecast"].sort_values("week_start")
    diagnostics = payload["diagnostics"]
    test = payload["test"]

    st.markdown("### Forecasting 13 weeks ahead from a fixed origin")
    st.caption(
        f"Trained on {payload['split']['train_weeks']} weeks ending "
        f"{payload['split']['test_start'][:10]}, scored once on the following "
        f"{payload['split']['test_weeks']} weeks. Selection uses walk-forward error inside the "
        "training period, never the hold-out."
    )

    kpi_cards(
        [
            (
                "Selected model",
                MODEL_LABELS.get(payload["selected_model"], payload["selected_model"]),
                "walk-forward selection",
            ),
            ("Hold-out RMSE", fmt_money(test["rmse"]), f"MASE {test['mase']:.2f}"),
            ("vs seasonal naive", f"{payload['improvement_vs_naive_pct']:.1f}%", "lower RMSE"),
            ("MAPE", f"{test['mape']:.1f}%", "mean absolute % error"),
        ]
    )
    st.caption(
        f"The 5-95% error band covered {fmt_pct(payload['interval_coverage'], 0)} of held-out weeks "
        "against a nominal 90%."
    )

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=forecast["week_start"],
            y=forecast["upper"],
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=forecast["week_start"],
            y=forecast["lower"],
            fill="tonexty",
            fillcolor="rgba(228, 87, 46, 0.15)",
            line={"width": 0},
            name="5-95% error band",
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=forecast["week_start"],
            y=forecast["actual"],
            mode="lines+markers",
            name="Actual",
            line={"color": NAVY, "width": 2.6},
            hovertemplate="%{x|%d %b %Y}<br>$%{y:,.0f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=forecast["week_start"],
            y=forecast["predicted"],
            mode="lines+markers",
            name="Forecast",
            line={"color": ORANGE, "width": 2.2, "dash": "dash"},
            hovertemplate="%{x|%d %b %Y}<br>$%{y:,.0f}<extra></extra>",
        )
    )
    st.plotly_chart(
        apply_layout(figure, 420, "Held-out weeks: shape learned, level missed"),
        width="stretch",
    )

    st.markdown(
        f'<div class="warn"><b>Read the gap, not just the score.</b> The model tracks the seasonal '
        f"rhythm but under-predicts by {fmt_money(abs(test['bias']))} in an average week, because the "
        f"fourth quarter of 2017 set a new record that no pre-October model could know. R² over these "
        f"13 weeks is {test['r2']:.2f}; MASE {test['mase']:.2f} is the fairer comparison, and the "
        "5-95% band covered fewer weeks than it promised.</div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)
    with left:
        comparison = pd.DataFrame(payload["test_all_models"]).T.reset_index()
        comparison.columns = ["model", "r2", "rmse", "mae", "bias", "mape", "n", "mase"]
        comparison["model"] = comparison["model"].map(lambda name: MODEL_LABELS.get(name, name))
        st.markdown("##### Hold-out scorecard")
        st.dataframe(
            comparison[["model", "rmse", "mae", "mape", "mase", "r2"]].style.format(
                {"rmse": "${:,.0f}", "mae": "${:,.0f}", "mape": "{:.1f}%", "mase": "{:.2f}", "r2": "{:.2f}"}
            ),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            f"Selected on mean walk-forward RMSE: Ridge "
            f"{fmt_money(payload['selection_scores']['ridge_regression'])}, gradient boosting "
            f"{fmt_money(payload['selection_scores']['gradient_boosting'])}."
        )

    with right:
        backtest = feed.get("backtest")
        if backtest is None or backtest.empty:
            st.info("Backtest table not exported.")
        else:
            figure = go.Figure()
            for name, colour, label in (
                ("rmse_naive", GREY, "Seasonal naive"),
                ("rmse_ridge", NAVY, "Ridge"),
                ("rmse_gb", ORANGE, "Gradient boosting"),
            ):
                figure.add_trace(
                    go.Bar(name=label, x=backtest["origin"], y=backtest[name], marker_color=colour)
                )
            figure.update_layout(barmode="group")
            figure.update_yaxes(title="RMSE (USD per week)")
            st.plotly_chart(
                apply_layout(figure, 320, "Walk-forward error by fold"),
                width="stretch",
            )

    st.markdown("#### Where the error comes from")
    left, right = st.columns(2)
    with left:
        frame = forecast.copy()
        frame["residual"] = frame["predicted"] - frame["actual"]
        frame["pct_error"] = frame["residual"] / frame["actual"]
        line = go.Figure(
            go.Bar(
                x=frame["horizon"],
                y=frame["pct_error"],
                marker_color=[ORANGE if value < 0 else PURPLE for value in frame["pct_error"]],
                hovertemplate="Week %{x}<br>%{y:.0%}<extra></extra>",
            )
        )
        line.add_hline(y=0, line_color="black", line_width=1)
        line.update_xaxes(title="Weeks ahead")
        line.update_yaxes(tickformat=".0%")
        st.plotly_chart(apply_layout(line, 320, "Percentage error by horizon"), width="stretch")
        st.caption(
            f"Worst week: {diagnostics['worst_week']} at {fmt_pct(diagnostics['worst_pct_error'])}. "
            f"{diagnostics['under_predicted_weeks']} of {diagnostics['weeks']} weeks were forecast too low."
        )

    with right:
        importance = pd.DataFrame(payload["feature_importance"]).head(8).sort_values("importance_mean")
        bar = go.Figure(
            go.Bar(
                x=importance["importance_mean"],
                y=importance["feature"],
                orientation="h",
                marker_color=NAVY,
                error_x={"array": importance["importance_std"]},
                hovertemplate="%{y}<br>+%{x:,.0f} RMSE when shuffled<extra></extra>",
            )
        )
        st.plotly_chart(apply_layout(bar, 320, "Feature importance"), width="stretch")

    st.markdown("#### What would actually improve this")
    st.markdown(
        "- **Shorter horizon.** Error is dominated by the level, not by drift within the 13 weeks.\n"
        "- **Four-week blocks.** Aggregation averages out the copier-sized orders that drive the residual.\n"
        "- **A leading indicator.** The extract has no promotion calendar or pipeline data, which is the "
        "missing variable behind a record quarter.\n"
        "- **Separate volume and value.** Forecast order count and average order value, then multiply; the "
        "lumpiness is much easier to attribute."
    )
    st.caption(
        "Reproduce with `python run.py all`. The model artefact, metrics and figures are all regenerated "
        "from the pinned dataset."
    )
