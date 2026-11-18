"""Explore tab: filter, inspect and download the aggregates."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dashboard.theme import NAVY, ORANGE, apply_layout, fmt_compact, fmt_money, kpi_cards


def filter_months(monthly: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    return monthly[monthly["month_start"].dt.year.isin(years)]


def render(feed: dict) -> None:
    monthly = feed["monthly"].copy()
    categories = feed["monthly_category"].copy()
    daily = feed["daily"].copy()
    available_years = sorted(monthly["month_start"].dt.year.unique().tolist())

    st.markdown("### Explore the aggregates")
    st.caption(
        "Daily and monthly totals, filtered by year and category. The full cleaned extract is "
        "reproduced by `python run.py data`; it is not committed because it is regenerable."
    )

    years = st.multiselect("Years", available_years, default=available_years, key="explore_years")
    category_names = sorted(categories.columns[1:].tolist())
    selected_categories = st.multiselect(
        "Categories", category_names, default=category_names, key="explore_categories"
    )

    filtered = filter_months(monthly, years or available_years)
    if filtered.empty:
        st.info("No months in that selection.")
        return

    kpi_cards(
        [
            ("Revenue", fmt_compact(filtered["revenue"].sum()), "in the selected months"),
            ("Profit", fmt_compact(filtered["profit"].sum()), "before overheads"),
            (
                "Margin",
                f"{filtered['profit'].sum() / filtered['revenue'].sum() * 100:.1f}%",
                "profit over revenue",
            ),
            ("Orders", f"{int(filtered['orders'].sum()):,}", "grouped by order id"),
        ]
    )

    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=filtered["month_start"],
            y=filtered["revenue"],
            name="Revenue",
            marker_color=NAVY,
            hovertemplate="%{x|%b %Y}<br>$%{y:,.0f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=filtered["month_start"],
            y=filtered["profit"],
            name="Profit",
            mode="lines+markers",
            line={"color": ORANGE, "width": 2},
            hovertemplate="%{x|%b %Y}<br>$%{y:,.0f}<extra></extra>",
        )
    )
    st.plotly_chart(apply_layout(figure, 380, "Revenue and profit by month"), width="stretch")

    if selected_categories:
        chosen = ["month_start"] + list(selected_categories)
        melted = categories.loc[categories["month_start"].dt.year.isin(years or available_years), chosen].melt(
            id_vars="month_start", var_name="category", value_name="revenue"
        )
        area = go.Figure()
        for category, group in melted.groupby("category"):
            area.add_trace(
                go.Scatter(
                    x=group["month_start"],
                    y=group["revenue"],
                    name=category,
                    stackgroup="one",
                    hovertemplate="%{x|%b %Y}<br>$%{y:,.0f}<extra></extra>",
                )
            )
        st.plotly_chart(apply_layout(area, 360, "Revenue by category, stacked"), width="stretch")

    st.markdown("#### Monthly table")
    table = filtered.assign(month=filtered["month_start"].dt.strftime("%b %Y"))[
        ["month", "revenue", "profit", "orders", "units", "margin", "aov"]
    ].rename(
        columns={
            "month": "Month",
            "revenue": "Revenue",
            "profit": "Profit",
            "orders": "Orders",
            "units": "Units",
            "margin": "Margin",
            "aov": "AOV",
        }
    )
    st.dataframe(
        table.style.format(
            {
                "Revenue": "${:,.0f}",
                "Profit": "${:,.0f}",
                "Margin": "{:.1%}",
                "AOV": "${:,.0f}",
            }
        ),
        width="stretch",
        hide_index=True,
        height=340,
    )
    st.download_button(
        "Download monthly totals (CSV)",
        table.to_csv(index=False).encode("utf-8"),
        file_name="superstore_monthly.csv",
        mime="text/csv",
    )

    st.markdown("#### Daily revenue")
    span = daily[daily["date"].dt.year.isin(years or available_years)]
    if not span.empty:
        line = go.Figure(
            go.Scatter(
                x=span["date"],
                y=span["revenue"],
                mode="lines",
                line={"color": NAVY, "width": 1},
                hovertemplate="%{x|%d %b %Y}<br>$%{y:,.0f}<extra></extra>",
            )
        )
        st.plotly_chart(apply_layout(line, 320, "Daily revenue"), width="stretch")
        st.caption(
            f"{(span['revenue'] == 0).mean() * 100:.1f}% of days in this selection record no sale. "
            f"Busiest single day: {span.loc[span['revenue'].idxmax(), 'date'].strftime('%d %b %Y')} at "
            f"{fmt_money(span['revenue'].max())}."
        )
        st.download_button(
            "Download daily revenue (CSV)",
            span.to_csv(index=False).encode("utf-8"),
            file_name="superstore_daily.csv",
            mime="text/csv",
        )

    st.markdown(
        '<div class="callout"><b>Note on aggregation.</b> Weeks are Monday-starting and the two '
        "boundary weeks are excluded from the modelled series so every week covers seven days. "
        "Months are calendar months and are never trimmed.</div>",
        unsafe_allow_html=True,
    )
