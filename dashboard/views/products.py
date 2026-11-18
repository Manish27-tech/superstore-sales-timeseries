"""Products tab: concentration, behaviour and the searchable catalogue."""

from __future__ import annotations

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.theme import GREY, NAVY, ORANGE, SEGMENT_COLORS, apply_layout, fmt_money, fmt_pct


def pareto_frame(products):
    """Cumulative revenue share, computed on the sorted catalogue."""
    frame = products.sort_values("revenue", ascending=False).reset_index(drop=True)
    frame["rank"] = np.arange(1, len(frame) + 1)
    frame["cumulative"] = frame["revenue"].cumsum() / frame["revenue"].sum()
    return frame


def render(feed: dict) -> None:
    products = feed["products"]
    summary = feed["kpis"]["segments"]

    st.markdown("### A fifth of the catalogue pays for the rest")
    st.caption(
        f"{summary['products_for_80pct']} of {summary['products']:,} products "
        f"({fmt_pct(summary['products_for_80pct'] / summary['products'])}) generate 80% of revenue. "
        "ABC classes come from that curve; behavioural segments come from a k-means fit on scale, growth, "
        "volatility and margin."
    )

    columns = st.columns(4)
    for column, name in zip(columns, ("A", "B", "C"), strict=False):
        column.metric(
            f"Class {name}",
            f"{summary['abc_products'][name]:,} products",
            f"{fmt_pct(summary['abc_revenue_share'][name])} of revenue",
        )
    columns[3].metric("Anchor products", f"{summary['segments']['Anchor products']['products']:,}", f"{fmt_pct(summary['anchor_share'])} of revenue")

    left, right = st.columns([1.1, 1])
    with left:
        frame = pareto_frame(products)
        line = go.Figure()
        line.add_trace(
            go.Scatter(
                x=frame["rank"],
                y=frame["cumulative"],
                mode="lines",
                line={"color": NAVY, "width": 2.4},
                hovertemplate="Rank %{x}<br>%{y:.0%} of revenue<extra></extra>",
            )
        )
        marker = int((frame["cumulative"] <= 0.8).sum() + 1)
        line.add_hline(y=0.8, line_dash="dash", line_color=ORANGE)
        line.add_vline(x=marker, line_dash="dash", line_color=ORANGE)
        line.add_annotation(
            x=marker,
            y=0.55,
            text=f"{marker} products → 80%",
            showarrow=True,
            arrowhead=2,
            ax=60,
            ay=0,
            font={"color": ORANGE},
        )
        line.update_yaxes(tickformat=".0%", range=[0, 1.02])
        st.plotly_chart(apply_layout(line, 380, "Cumulative revenue share"), width="stretch")

    with right:
        by_segment = (
            products.groupby("segment")
            .agg(revenue=("revenue", "sum"), products=("product_id", "size"))
            .reset_index()
        )
        bar = px.bar(
            by_segment,
            x="segment",
            y="revenue",
            color="segment",
            color_discrete_map=SEGMENT_COLORS,
            text="products",
            labels={"segment": "", "revenue": "Revenue", "products": "products"},
        )
        bar.update_traces(texttemplate="%{text} products", textposition="outside")
        st.plotly_chart(apply_layout(bar, 380, "Revenue by behavioural segment"), width="stretch")

    st.markdown("#### Every product, positioned by scale and growth")
    categories = sorted(products["category"].unique())
    selected_categories = st.multiselect("Category", categories, default=categories, key="product_categories")
    selected_segments = st.multiselect(
        "Segment", sorted(products["segment"].unique()), default=sorted(products["segment"].unique()), key="product_segments"
    )
    filtered = products[
        products["category"].isin(selected_categories) & products["segment"].isin(selected_segments)
    ]

    if filtered.empty:
        st.info("No products match that combination of filters.")
    else:
        scatter = px.scatter(
            filtered,
            x="revenue",
            y="growth",
            color="segment",
            color_discrete_map=SEGMENT_COLORS,
            size="orders",
            size_max=22,
            hover_name="product_name",
            hover_data={"revenue": ":,.0f", "margin": ":.1%", "abc_class": True, "growth": ":.0%"},
            labels={"revenue": "Revenue (log scale)", "growth": "Growth: last 4 vs first 4 months"},
        )
        scatter.update_xaxes(type="log")
        scatter.update_yaxes(tickformat=".0%", range=[-1, 2.5])
        st.plotly_chart(apply_layout(scatter, 460), width="stretch")

    st.markdown("#### Catalogue table")
    search = st.text_input("Search product names", "")
    table = filtered.copy()
    if search:
        table = table[table["product_name"].str.contains(search, case=False, na=False)]
    table = table.sort_values("revenue", ascending=False)
    display = table[
        [
            "product_name",
            "category",
            "sub_category",
            "revenue",
            "margin",
            "growth",
            "orders",
            "abc_class",
            "segment",
        ]
    ].rename(
        columns={
            "product_name": "Product",
            "category": "Category",
            "sub_category": "Sub-category",
            "revenue": "Revenue",
            "margin": "Margin",
            "growth": "Growth",
            "orders": "Orders",
            "abc_class": "ABC",
            "segment": "Segment",
        }
    )
    st.caption(f"{len(display):,} products shown")
    st.dataframe(
        display.style.format({"Revenue": "${:,.0f}", "Margin": "{:.1%}", "Growth": "{:+.0%}"}),
        width="stretch",
        height=420,
    )
    st.download_button(
        "Download this table (CSV)",
        display.to_csv(index=False).encode("utf-8"),
        file_name="superstore_products.csv",
        mime="text/csv",
    )

    st.markdown(
        '<div class="callout"><b>What the segments mean.</b> Anchors are large and stable; risers grew '
        "sharply between their first and last four active months; the steady mid-tier is small but "
        "consistent; the long tail is most of the catalogue by count and a small fraction of revenue. "
        "Growth is measured within each product's own life, so a 2016 launch is not penalised for "
        "being absent in 2014.</div>",
        unsafe_allow_html=True,
    )
