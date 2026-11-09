"""Streamlit dashboard: streamlit run app.py"""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from priceopt import (baseline_cadences, current_state, estimate_elasticities, generate_sales, load_sales,
                      optimize_markdown, portfolio_summary, recommend_prices, simulate)
from priceopt.optimizer import project, unconstrained_optimum

st.set_page_config(page_title="Price Elasticity Optimizer", layout="wide")

ACTION_COLORS = alt.Scale(domain=["Raise", "Lower", "Hold"], range=["#2a9d8f", "#e76f51", "#9aa5b1"])


@st.cache_data
def demo_data():
    return generate_sales()


@st.cache_data
def run_estimation(sales: pd.DataFrame, shrink: bool):
    return estimate_elasticities(sales, shrink=shrink), current_state(sales)


# ── Sidebar ──────────────────────────────────────────────────────────────
st.sidebar.title("Settings")
source = st.sidebar.radio("Data", ["Demo data (synthetic)", "Upload CSV"])
truth = None
if source == "Upload CSV":
    upload = st.sidebar.file_uploader("Weekly sales CSV", type="csv")
    st.sidebar.caption("Columns: week, sku, category, price, units, unit_cost, [promo], [competitor_price]")
    if upload is None:
        st.info("Upload a sales CSV to begin, or switch to demo data.")
        st.stop()
    try:
        sales = load_sales(upload)
    except ValueError as e:
        st.error(str(e))
        st.stop()
else:
    sales, truth = demo_data()

st.sidebar.subheader("Guardrails")
max_change = st.sidebar.slider("Max price move", 0.05, 0.30, 0.15, 0.01, format="±%.2f")
objective = st.sidebar.selectbox("Optimize for", ["margin", "revenue"], format_func=str.title)
min_margin = st.sidebar.slider("Minimum gross margin", 0.0, 0.6, 0.0, 0.05, format="%.2f")
max_se = st.sidebar.slider("Max std. error to act on", 0.1, 1.0, 0.5, 0.05)
shrink = st.sidebar.toggle("Category shrinkage", value=True,
                           help="Pull noisy SKU estimates toward their category average")

est, state = run_estimation(sales, shrink)
recs = recommend_prices(est, state, max_change, objective, min_margin, max_se)
summary = portfolio_summary(recs)

# ── Header ───────────────────────────────────────────────────────────────
st.title("Price Elasticity & Markdown Optimizer")
st.caption(f"{sales['sku'].nunique()} SKUs · {sales['category'].nunique()} categories · "
           f"{sales['week'].min():%b %Y} – {sales['week'].max():%b %Y}"
           + (" · synthetic demo data with known true elasticities" if truth is not None else ""))

k1, k2, k3, k4 = st.columns(4)
k1.metric("Weekly gross margin", f"${summary['weekly_margin_projected']:,.0f}",
          f"{summary['margin_lift_pct']:+.1f}%")
k2.metric("Annualized margin lift", f"${summary['annualized_margin_lift']:,.0f}")
k3.metric("Revenue change", f"{summary['revenue_change_pct']:+.1f}%")
k4.metric("Actions", f"{summary['raise']} ↑  {summary['lower']} ↓  {summary['hold']} =")

tab_recs, tab_est, tab_sku, tab_md = st.tabs(
    ["Price recommendations", "Elasticity estimates", "SKU deep dive", "Markdown planner"])

# ── Recommendations ──────────────────────────────────────────────────────
with tab_recs:
    chart = alt.Chart(recs).mark_bar().encode(
        x=alt.X("margin_lift:Q", title="Weekly margin lift ($)"),
        y=alt.Y("sku:N", sort="-x", title=None),
        color=alt.Color("action:N", scale=ACTION_COLORS, title="Action"),
        tooltip=["sku", "category", "action", alt.Tooltip("current_price", format="$,.2f"),
                 alt.Tooltip("recommended_price", format="$,.2f"), alt.Tooltip("margin_lift", format="$,.0f"),
                 "rationale"],
    ).properties(height=max(300, 18 * len(recs)))
    st.altair_chart(chart, width="stretch")

    by_cat = recs.groupby("category").agg(
        skus=("sku", "count"),
        avg_elasticity=("elasticity", "mean"),
        current_margin=("current_margin", "sum"),
        projected_margin=("projected_margin", "sum"),
    )
    by_cat["margin_lift_pct"] = (by_cat["projected_margin"] / by_cat["current_margin"] - 1) * 100
    st.subheader("By category")
    st.dataframe(by_cat.style.format({"avg_elasticity": "{:.2f}", "current_margin": "${:,.0f}",
                                      "projected_margin": "${:,.0f}", "margin_lift_pct": "{:+.1f}%"}),
                 width="stretch")

    st.subheader("All SKUs")
    st.dataframe(recs, width="stretch", hide_index=True, column_config={
        c: st.column_config.NumberColumn(format="$%.2f") for c in
        ["unit_cost", "current_price", "recommended_price", "current_revenue", "projected_revenue",
         "current_margin", "projected_margin", "margin_lift"]})
    st.download_button("Download recommendations CSV", recs.to_csv(index=False), "price_recommendations.csv")

# ── Elasticity estimates ─────────────────────────────────────────────────
with tab_est:
    plot = est.dropna(subset=["elasticity_shrunk"]).copy()
    if truth is not None:
        plot = plot.merge(truth[["sku", "true_elasticity"]], on="sku")
        mae_raw = (plot["elasticity"] - plot["true_elasticity"]).abs().mean()
        mae_shr = (plot["elasticity_shrunk"] - plot["true_elasticity"]).abs().mean()
        cover = plot["true_elasticity"].between(plot["ci_low"], plot["ci_high"]).mean()
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean abs. error (raw)", f"{mae_raw:.3f}")
        c2.metric("Mean abs. error (shrunk)", f"{mae_shr:.3f}", f"{(mae_shr / mae_raw - 1) * 100:+.0f}%",
                  delta_color="inverse")
        c3.metric("True value inside 95% CI", f"{cover:.0%}")

    base = alt.Chart(plot).encode(y=alt.Y("sku:N", sort=alt.EncodingSortField("elasticity_shrunk"), title=None))
    layers = [
        base.mark_rule(strokeWidth=2, opacity=0.6).encode(
            x=alt.X("ci_low:Q", title="Own-price elasticity"), x2="ci_high:Q",
            color=alt.Color("category:N", title="Category")),
        base.mark_point(filled=True, size=70).encode(
            x="elasticity_shrunk:Q", color="category:N",
            tooltip=["sku", "category", "elasticity", "elasticity_shrunk", "std_error_shrunk", "status"]),
        base.mark_tick(color="#555", thickness=1, size=10).encode(x="elasticity:Q"),
    ]
    if truth is not None:
        layers.append(base.mark_point(shape="diamond", color="black", size=50, filled=False)
                      .encode(x="true_elasticity:Q", tooltip=["sku", "true_elasticity"]))
    rule = alt.Chart(pd.DataFrame({"x": [-1]})).mark_rule(strokeDash=[4, 4], color="gray").encode(x="x:Q")
    st.altair_chart(alt.layer(*layers, rule).properties(height=max(320, 18 * len(plot))),
                    width="stretch")
    st.caption("Dot = shrunk estimate with 95% CI · tick = raw SKU estimate"
               + (" · ◇ = true elasticity" if truth is not None else "")
               + " · dashed line = unit elasticity (−1): left of it, demand is elastic")
    st.dataframe(est, width="stretch", hide_index=True)

# ── SKU deep dive ────────────────────────────────────────────────────────
with tab_sku:
    sku = st.selectbox("SKU", recs["sku"], index=int(recs["margin_lift"].idxmax()))
    r = recs.set_index("sku").loc[sku]
    s = state.set_index("sku").loc[sku]
    hist = sales[sales["sku"] == sku]

    headline = (f"**{r['action']}** ${r['current_price']:,.2f} → **${r['recommended_price']:,.2f}** "
                f"({r['price_change_pct']:+.1f}%) · elasticity {r['elasticity']:.2f}  \n{r['rationale']}")
    st.markdown(headline.replace("$", r"\$"))   # unescaped $ pairs render as LaTeX

    left, right = st.columns(2)
    if np.isfinite(r["elasticity"]) and r["elasticity"] < 0:
        prices = np.linspace(max(s["unit_cost"] * 1.01, s["current_price"] * 0.6), s["current_price"] * 1.4, 200)
        p = project(prices, s["current_price"], s["baseline_units"], r["elasticity"], s["unit_cost"])
        curve = pd.DataFrame({"price": prices, "Gross margin": p["margin"], "Revenue": p["revenue"]}).melt(
            "price", var_name="metric", value_name="weekly_$")
        band = alt.Chart(pd.DataFrame({"lo": [s["current_price"] * (1 - max_change)],
                                       "hi": [s["current_price"] * (1 + max_change)]})).mark_rect(
            opacity=0.08, color="#2a9d8f").encode(x="lo:Q", x2="hi:Q")
        lines = alt.Chart(curve).mark_line().encode(
            x=alt.X("price:Q", title="Price ($)", scale=alt.Scale(zero=False)),
            y=alt.Y("weekly_$:Q", title="Weekly $"), color="metric:N")
        marks = alt.Chart(pd.DataFrame({"price": [s["current_price"], r["recommended_price"]],
                                        "label": ["Current", "Recommended"]})).mark_rule(
            strokeDash=[5, 3]).encode(x="price:Q", color=alt.Color("label:N", title=None,
                                      scale=alt.Scale(range=["#9aa5b1", "#e9c46a"])))
        left.markdown("**Projected weekly revenue and margin vs price** (shaded = guardrail)")
        left.altair_chart(alt.layer(band, lines, marks).resolve_scale(color="independent"),
                          width="stretch")
        p_star = unconstrained_optimum(r["elasticity"], s["unit_cost"])
        left.caption(f"Unconstrained profit-max price: "
                     + (f"${p_star:,.2f}" if np.isfinite(p_star) else "none (inelastic demand)"))

    scatter = alt.Chart(hist).mark_circle(opacity=0.6).encode(
        x=alt.X("price:Q", title="Price ($)", scale=alt.Scale(
            zero=False, nice=False, domain=[hist["price"].min() * 0.97, hist["price"].max() * 1.03])),
        y=alt.Y("units:Q", title="Weekly units"),
        color=alt.Color("promo:N", title="Promo", scale=alt.Scale(range=["#264653", "#e76f51"])),
        tooltip=["week", "price", "units", "promo"])
    right.markdown("**Observed price vs units**")
    right.altair_chart(scatter, width="stretch")

    hist_long = hist.melt("week", ["price", "units"])
    st.altair_chart(alt.Chart(hist_long).mark_line().encode(
        x="week:T", y=alt.Y("value:Q", title=None), row=alt.Row("variable:N", title=None)
    ).properties(height=140).resolve_scale(y="independent"), width="stretch")

# ── Markdown planner ─────────────────────────────────────────────────────
with tab_md:
    st.markdown("Clearance plan for a fixed inventory position. Dynamic programming checks every "
                "cadence where markdowns only deepen, then compares it with common rules of thumb.")
    md_sku = st.selectbox("Prefill from SKU", recs["sku"], key="md_sku")
    mr = recs.set_index("sku").loc[md_sku]
    ms = state.set_index("sku").loc[md_sku]
    e_default = float(np.clip(mr["elasticity"], -6, -0.1)) if np.isfinite(mr["elasticity"]) else -2.0

    c1, c2, c3, c4 = st.columns(4)
    weeks = c1.number_input("Weeks left in season", 2, 26, 12)
    inventory = c2.number_input("Units on hand", 1, 5000, int(min(max(ms["baseline_units"] * weeks * 1.3, 10), 5000)))
    full_price = c3.number_input("Full price ($)", 1.0, 100000.0, float(ms["current_price"]))
    unit_cost = c4.number_input("Unit cost ($)", 0.0, 100000.0, float(ms["unit_cost"]))
    c5, c6, c7, c8 = st.columns(4)
    base_units = c5.number_input("Weekly demand at full price", 0.1, 5000.0, float(round(ms["baseline_units"], 1)))
    elasticity = c6.number_input("Elasticity", -6.0, -0.1, round(e_default, 2), 0.1)
    decay = c7.slider("Weekly demand fade", 0.0, 0.15, 0.04, 0.01, help="How fast interest fades as the season ages")
    salvage = c8.number_input("Salvage value per unit ($)", 0.0, float(full_price), float(round(unit_cost * 0.3, 2)))

    plan = optimize_markdown(int(inventory), int(weeks), full_price, base_units, elasticity, unit_cost,
                             salvage, decay)
    compare = [{"Strategy": "Optimized (DP)", "Revenue + salvage": plan.total_value,
                "Gross margin": plan.gross_margin, "Sell-through": plan.sell_through_pct,
                "Leftover units": plan.leftover_units}]
    for name, cadence in baseline_cadences(int(weeks)).items():
        b = simulate(cadence, int(inventory), full_price, base_units, elasticity, unit_cost, salvage, decay)
        compare.append({"Strategy": name, "Revenue + salvage": b.total_value, "Gross margin": b.gross_margin,
                        "Sell-through": b.sell_through_pct, "Leftover units": b.leftover_units})
    compare = pd.DataFrame(compare)
    best_rule = compare.iloc[1:]["Gross margin"].max()

    m1, m2, m3 = st.columns(3)
    m1.metric("Optimized gross margin", f"${plan.gross_margin:,.0f}",
              f"{'+' if plan.gross_margin >= best_rule else '−'}${abs(plan.gross_margin - best_rule):,.0f} "
              "vs best rule of thumb")
    m2.metric("Sell-through", f"{plan.sell_through_pct:.0f}%")
    m3.metric("Leftover units", f"{plan.leftover_units:,}")

    sched = plan.schedule
    bars = alt.Chart(sched).mark_bar(color="#2a9d8f", opacity=0.7).encode(
        x=alt.X("week:O", title="Week"), y=alt.Y("inventory_end:Q", title="Inventory at week end"),
        tooltip=list(sched.columns))
    line = alt.Chart(sched).mark_line(point=True, color="#e76f51").encode(
        x="week:O", y=alt.Y("markdown_pct:Q", title="Markdown %", scale=alt.Scale(domain=[0, 70])))
    st.altair_chart(alt.layer(bars, line).resolve_scale(y="independent"), width="stretch")

    left, right = st.columns(2)
    left.dataframe(sched, hide_index=True, width="stretch")
    right.dataframe(compare.style.format({"Revenue + salvage": "${:,.0f}", "Gross margin": "${:,.0f}",
                                          "Sell-through": "{:.0f}%"}), hide_index=True,
                    width="stretch")
