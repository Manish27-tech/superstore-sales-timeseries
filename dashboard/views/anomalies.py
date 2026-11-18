"""Anomalies tab: the weeks that broke the pattern and the orders behind them."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from dashboard.theme import NAVY, ORANGE, PURPLE, apply_layout, fmt_money, fmt_pct


def render(feed: dict) -> None:
    weekly = feed["weekly"].copy()
    anomalies = feed["anomalies"].copy()
    summary = feed["kpis"]["anomalies"]

    st.markdown("### Rare, upward, and traceable to one order")
    st.caption(
        f"{summary['anomalies']} of {summary['eligible_weeks']} comparable weeks "
        f"({fmt_pct(summary['share_of_weeks'])}) exceed a robust z-score of {summary['threshold']} "
        f"against the same week a year earlier, after requiring a baseline of at least "
        f"{fmt_money(summary['min_expected'])} and a gap of at least {fmt_money(summary['min_deviation'])}."
    )

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=weekly["week_start"],
            y=weekly["revenue"],
            mode="lines",
            name="Weekly revenue",
            line={"color": NAVY, "width": 1.3},
            hovertemplate="%{x|%d %b %Y}<br>$%{y:,.0f}<extra></extra>",
        )
    )
    spikes = anomalies[anomalies["direction"] == "spike"]
    dips = anomalies[anomalies["direction"] == "dip"]
    if not spikes.empty:
        figure.add_trace(
            go.Scatter(
                x=spikes["week_start"],
                y=spikes["revenue"],
                mode="markers",
                name="Spike",
                marker={"color": ORANGE, "size": 11, "line": {"width": 1, "color": "white"}},
                hovertemplate="%{x|%d %b %Y}<br>$%{y:,.0f}<extra></extra>",
            )
        )
    if not dips.empty:
        figure.add_trace(
            go.Scatter(
                x=dips["week_start"],
                y=dips["revenue"],
                mode="markers",
                name="Dip",
                marker={"color": PURPLE, "size": 11},
            )
        )
    st.plotly_chart(
        apply_layout(figure, 400, "Weekly revenue with flagged anomalies"),
        width="stretch",
    )

    table = anomalies.assign(
        week=anomalies["week_start"].dt.strftime("%d %b %Y"),
        gap=anomalies["deviation"],
    )[
        ["week", "direction", "z_score", "revenue", "expected", "gap", "largest_order_value", "largest_order_product"]
    ].rename(
        columns={
            "week": "Week",
            "direction": "Direction",
            "z_score": "z",
            "revenue": "Revenue",
            "expected": "Same week last year",
            "gap": "Gap",
            "largest_order_value": "Largest line",
            "largest_order_product": "Product on that line",
        }
    )
    st.dataframe(
        table.style.format(
            {
                "z": "{:.2f}",
                "Revenue": "${:,.0f}",
                "Same week last year": "${:,.0f}",
                "Gap": "${:,.0f}",
                "Largest line": "${:,.0f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    left, right = st.columns(2)
    with left:
        scatter = go.Figure(
            go.Scatter(
                x=anomalies["largest_order_share"],
                y=anomalies["z_score"],
                mode="markers+text",
                text=[date.strftime("%b %y") for date in anomalies["week_start"]],
                textposition="top center",
                marker={"color": ORANGE, "size": 13},
                hovertemplate="%{text}<br>z = %{y:.1f}<extra></extra>",
            )
        )
        scatter.add_hline(y=0, line_color="black", line_width=1)
        scatter.update_xaxes(tickformat=".0%", title="Share of the week from its largest order line")
        scatter.update_yaxes(title="Robust z-score")
        st.plotly_chart(
            apply_layout(scatter, 340, "Big spikes and big single orders move together"),
            width="stretch",
        )

    with right:
        st.markdown("##### Why this matters")
        st.markdown(
            f"- The largest single order line accounted for "
            f"{fmt_pct(summary['largest_single_order_share'], 0)} of its week's revenue.\n"
            f"- Largest deviation: {summary['top_anomalies'][0]['week_start']} at "
            f"z = {summary['top_anomalies'][0]['z_score']:.1f}.\n"
            "- A spike this concentrated is an order-level event, not a change in demand, so the "
            "response is a pipeline check, not a re-forecast.\n"
            "- Dips are absent: the guards remove low-baseline weeks where a ratio is meaningless."
        )

    st.markdown(
        '<div class="callout"><b>Method.</b> Expected revenue is the same week one year earlier; the '
        "score is (ratio - median) / (1.4826 x MAD). The two business guards stop a quiet week with a "
        "tiny baseline from producing a dramatic but meaningless ratio. The scoring window therefore "
        "starts in 2015, which is why the busiest week of all (March 2014) cannot be scored: it has no "
        "year-earlier comparison.</div>",
        unsafe_allow_html=True,
    )
