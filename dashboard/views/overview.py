"""Landing tab: the four numbers that matter and the shape of the series."""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.theme import (
    CATEGORY_COLORS,
    GREY,
    NAVY,
    ORANGE,
    apply_layout,
    fmt_compact,
    fmt_money,
    fmt_pct,
)


def render(feed: dict) -> None:
    kpis = feed["kpis"]["sales"]
    monthly = feed["monthly"].copy()
    categories = feed["monthly_category"].copy()
    products = feed["products"]

    st.markdown("### Four years of product sales in one view")
    st.caption(
        "9,994 order lines from the Superstore extract, aggregated to Monday-starting weeks. "
        "Every figure below is computed by the pipeline and committed with the code."
    )

    columns = st.columns(5)
    columns[0].metric("Revenue", fmt_compact(kpis["total_sales"]), f"{kpis['yoy_growth']['2017'] * 100:+.1f}% in 2017")
    columns[1].metric("Profit", fmt_compact(kpis["total_profit"]), f"{kpis['margin'] * 100:.1f}% margin")
    columns[2].metric("Orders", f"{kpis['orders']:,}", f"{kpis['customers']:,} customers")
    columns[3].metric("Mean order value", fmt_money(kpis["mean_order_value"]), "median " + fmt_money(kpis["median_order_value"]))
    columns[4].metric("Products", f"{kpis['products']:,}", f"{kpis['products_for_80pct']} make 80% of sales")

    monthly["rolling_12"] = monthly["revenue"].rolling(12, min_periods=6).mean()
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=monthly["month_start"],
            y=monthly["revenue"],
            name="Monthly revenue",
            line={"color": NAVY, "width": 1.6},
            hovertemplate="%{x|%b %Y}<br>$%{y:,.0f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=monthly["month_start"],
            y=monthly["rolling_12"],
            name="12-month average",
            line={"color": ORANGE, "width": 2.6, "dash": "dash"},
            hovertemplate="%{x|%b %Y}<br>$%{y:,.0f}<extra></extra>",
        )
    )
    st.plotly_chart(
        apply_layout(figure, 400, "Revenue compounds, but individual months are lumpy"),
        width="stretch",
    )

    left, right = st.columns([1.35, 1])
    with left:
        melted = categories.melt(id_vars="month_start", var_name="category", value_name="revenue")
        area = px.area(
            melted,
            x="month_start",
            y="revenue",
            color="category",
            color_discrete_map=CATEGORY_COLORS,
            labels={"month_start": "", "revenue": "Revenue"},
        )
        area.update_traces(hovertemplate="%{x|%b %Y}<br>$%{y:,.0f}<extra></extra>")
        st.plotly_chart(
            apply_layout(area, 380, "Technology leads every year; Office Supplies rebounds in 2017"),
            width="stretch",
        )

    with right:
        region = (
            feed["monthly"]
            .assign(month=feed["monthly"]["month_start"].dt.month)
            .groupby("month")["revenue"]
            .mean()
            .reset_index()
        )
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        region["label"] = region["month"].map(lambda value: month_names[value - 1])
        bar = go.Figure(
            go.Bar(
                x=region["label"],
                y=region["revenue"],
                marker_color=[ORANGE if value >= region["revenue"].median() else GREY for value in region["revenue"]],
                hovertemplate="%{x}<br>$%{y:,.0f}<extra></extra>",
            )
        )
        st.plotly_chart(
            apply_layout(bar, 380, "Average revenue by calendar month"),
            width="stretch",
        )

    st.markdown("#### Concentration and mix")
    left, right = st.columns([1, 1])
    with left:
        top = (
            products.groupby("sub_category")["revenue"]
            .sum()
            .sort_values(ascending=True)
            .tail(10)
            .reset_index()
        )
        bar = px.bar(
            top,
            x="revenue",
            y="sub_category",
            orientation="h",
            color="revenue",
            color_continuous_scale=["#C9D6E3", NAVY],
            labels={"revenue": "Revenue", "sub_category": ""},
        )
        bar.update_layout(coloraxis_showscale=False)
        bar.update_traces(hovertemplate="$%{x:,.0f}<extra></extra>")
        st.plotly_chart(apply_layout(bar, 420, "Top ten sub-categories"), width="stretch")

    with right:
        by_segment = products.groupby("segment")["revenue"].sum().reset_index()
        donut = px.pie(
            by_segment,
            names="segment",
            values="revenue",
            hole=0.55,
            color="segment",
            color_discrete_map={
                "Anchor products": NAVY,
                "Rising products": "#55A868",
                "Steady mid-tier": "#DDB75D",
                "Long tail": GREY,
            },
        )
        donut.update_traces(textinfo="percent", hovertemplate="%{label}<br>$%{value:,.0f}<extra></extra>")
        st.plotly_chart(apply_layout(donut, 420, "Where revenue sits by product segment"), width="stretch")

    st.markdown(
        f'<div class="callout">Seasonality explains <b>{fmt_pct(feed["kpis"]["timeseries"]["seasonal_strength"], 0)}</b> '
        f'of the detrended variance, and the trend explains '
        f'<b>{fmt_pct(feed["kpis"]["timeseries"]["trend_strength"], 0)}</b> of the deseasonalised variance. '
        f'But {fmt_pct(kpis["quiet_days_share"], 0)} of calendar days record no sale at all, and a single '
        f'copier order can double a quiet week &mdash; which is why the forecast tab reports an honest, '
        f'not-flattering, error.</div>',
        unsafe_allow_html=True,
    )

    with st.expander("How to read this dashboard"):
        st.markdown(
            "- **Overview** is the executive summary: level, growth, mix.\n"
            "- **Seasonality** decomposes the series into trend and calendar effects.\n"
            "- **Products** ranks the catalogue and splits it into behavioural segments.\n"
            "- **Forecast** shows a 13-week-ahead model, its errors, and why they happen.\n"
            "- **Anomalies** explains the weeks that broke the pattern.\n"
            "- **Explore** lets you filter and download the underlying aggregates."
        )
