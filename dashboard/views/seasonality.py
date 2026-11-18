"""Seasonality tab: what repeats, what trends, and how strong each part is."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.theme import GREY, NAVY, ORANGE, PURPLE, apply_layout, fmt_pct

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def build_heatmap(monthly: pd.DataFrame) -> pd.DataFrame:
    """Month x year matrix indexed to each year's own average."""
    frame = monthly.copy()
    frame["year"] = frame["month_start"].dt.year.astype(str)
    frame["month"] = frame["month_start"].dt.month
    pivot = frame.pivot_table(index="year", columns="month", values="revenue", aggfunc="sum")
    return pivot.div(pivot.mean(axis=1), axis=0)


def render(feed: dict) -> None:
    timeseries = feed["kpis"]["timeseries"]
    monthly = feed["monthly"]
    weekday = feed["weekday"]

    st.markdown("### The same three months carry every year")
    st.caption(
        "Decomposition uses a centred 12-month moving average for the trend and "
        "month-average ratios for the seasonal component."
    )

    left, right = st.columns([1.6, 1])
    with left:
        heat = build_heatmap(monthly)
        figure = go.Figure(
            go.Heatmap(
                z=heat.to_numpy(),
                x=MONTHS,
                y=heat.index.tolist(),
                colorscale="RdYlGn",
                zmin=0.3,
                zmax=1.9,
                text=np.round(heat.to_numpy(), 2),
                texttemplate="%{text:.2f}",
                hovertemplate="%{y} %{x}<br>%{z:.2f}x the year average<extra></extra>",
                colorbar={"title": "x year avg"},
            )
        )
        st.plotly_chart(apply_layout(figure, 360, "Each month against its own year's average"), width="stretch")

    with right:
        factors = timeseries["seasonal_factors"]
        values = [factors[month] for month in MONTHS]
        bar = go.Figure(
            go.Bar(
                x=MONTHS,
                y=values,
                marker_color=[ORANGE if value >= 1 else GREY for value in values],
                hovertemplate="%{x}<br>%{y:.2f}x<extra></extra>",
            )
        )
        bar.add_hline(y=1, line_color="black", line_width=1)
        st.plotly_chart(apply_layout(bar, 360, "Seasonal factor: 1.0 is an average month"), width="stretch")

    columns = st.columns(4)
    columns[0].metric("Strongest month", timeseries["strongest_month"], f"{timeseries['strongest_factor']:.2f}x")
    columns[1].metric("Weakest month", timeseries["weakest_month"], f"{timeseries['weakest_factor']:.2f}x")
    columns[2].metric("Peak-to-trough", f"{timeseries['peak_to_trough']:.1f}x", "seasonal ratio")
    columns[3].metric(
        "Trend growth",
        fmt_pct(timeseries["trend_growth"], 0),
        "over the 24 months with a centred trend",
    )

    left, right = st.columns(2)
    with left:
        line = go.Figure()
        line.add_trace(
            go.Scatter(
                x=monthly["month_start"],
                y=monthly["revenue"].rolling(12, min_periods=12).sum(),
                mode="lines",
                line={"color": NAVY, "width": 2.6},
                hovertemplate="%{x|%b %Y}<br>$%{y:,.0f}<extra></extra>",
            )
        )
        st.plotly_chart(
            apply_layout(line, 340, "Trailing twelve-month revenue"),
            width="stretch",
        )

    with right:
        annual = {int(year): value for year, value in feed["kpis"]["sales"]["annual_revenue"].items()}
        years = [str(year) for year in sorted(annual)]
        growth = [annual[year] for year in sorted(annual)]
        bar = go.Figure(
            go.Bar(
                x=years,
                y=growth,
                marker_color=NAVY,
                text=[f"${value/1000:.0f}k" for value in growth],
                textposition="outside",
                hovertemplate="%{x}<br>$%{y:,.0f}<extra></extra>",
            )
        )
        bar.update_yaxes(range=[0, max(growth) * 1.18])
        st.plotly_chart(apply_layout(bar, 340, "Annual revenue"), width="stretch")

    st.markdown("#### Weekday profile")
    weekday = weekday.copy()
    weekday["share"] = weekday["revenue"] / weekday["revenue"].sum()
    bar = go.Figure(
        go.Bar(
            x=weekday["weekday"],
            y=weekday["revenue"],
            marker_color=[PURPLE if day == "Wednesday" else NAVY for day in weekday["weekday"]],
            text=[fmt_pct(value) for value in weekday["share"]],
            textposition="outside",
            hovertemplate="%{x}<br>$%{y:,.0f}<extra></extra>",
        )
    )
    bar.update_yaxes(range=[0, weekday["revenue"].max() * 1.2])
    st.plotly_chart(apply_layout(bar, 340, "Revenue by weekday of order"), width="stretch")
    st.markdown(
        '<div class="warn"><b>Read this chart with suspicion.</b> Wednesday holds only '
        f'{fmt_pct(feed["kpis"]["sales"]["weekday_line_share"]["Wednesday"])} of order lines against '
        "11-19% for every other weekday. That is a property of the published extract, not the business, "
        "so weekday patterns are shown for completeness and are not used in any model or recommendation."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("#### Quarter-on-quarter view")
    frame = monthly.copy()
    frame["year"] = frame["month_start"].dt.year
    frame["quarter"] = frame["month_start"].dt.quarter
    table = (
        frame.pivot_table(index="year", columns="quarter", values="revenue", aggfunc="sum")
        .round(0)
        .rename(columns=lambda value: f"Q{value}")
    )
    table["Total"] = table.sum(axis=1)
    st.dataframe(
        table.style.format("${:,.0f}"),
        width="stretch",
    )
