import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import duckdb
import os
import sys

# Ensure local imports work cleanly
sys.path.append(os.path.abspath("."))
from marketing_incrementality_mmm.models.budget_optimizer import BudgetOptimizer
from marketing_incrementality_mmm.models.transformations import hill_saturation, hill_derivative
from marketing_incrementality_mmm.reports.audit_generator import AuditReportGenerator

st.set_page_config(
    page_title="Meridian-Lite | Incrementality & MMM Cockpit",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card { background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 15px; text-align: center; }
    .kpi-title { font-size: 12px; color: #64748b; font-weight: 600; text-transform: uppercase; }
    .kpi-val { font-size: 24px; font-weight: 700; color: #1e293b; margin: 4px 0; }
    .kpi-sub { font-size: 11px; color: #16a34a; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# 1. Fitted Model Parameters (from Milestone 3)
FITTED_PARAMS = {
    "Meta_Prospecting":    {"lambda_decay": 0.284, "k_half_sat": 2023.6, "gamma": 1.801, "beta_max": 3934.4},
    "Meta_Retargeting":    {"lambda_decay": 0.523, "k_half_sat": 2023.6, "gamma": 0.600, "beta_max": 3962.5},
    "Google_Brand_Search": {"lambda_decay": 0.145, "k_half_sat": 2023.6, "gamma": 0.600, "beta_max": 3968.4},
    "Google_PMax_NonBrand":{"lambda_decay": 0.167, "k_half_sat": 2023.6, "gamma": 0.600, "beta_max": 3938.0},
    "TikTok_Ads":          {"lambda_decay": 0.374, "k_half_sat": 2023.6, "gamma": 1.362, "beta_max": 3950.7},
    "YouTube_Awareness":   {"lambda_decay": 0.461, "k_half_sat": 2023.6, "gamma": 1.235, "beta_max": 3953.8}
}

CURRENT_DAILY_SPEND = {
    "Meta_Prospecting": 2289.55,
    "Meta_Retargeting": 800.89,
    "Google_Brand_Search": 627.72,
    "Google_PMax_NonBrand": 1985.19,
    "TikTok_Ads": 1260.27,
    "YouTube_Awareness": 1121.37
}

CLAIMED_ROAS = {
    "Meta_Prospecting": 2.85,
    "Meta_Retargeting": 5.40,
    "Google_Brand_Search": 6.20,
    "Google_PMax_NonBrand": 3.10,
    "TikTok_Ads": 2.10,
    "YouTube_Awareness": 1.45
}

TOTAL_CURRENT_BUDGET = sum(CURRENT_DAILY_SPEND.values())

@st.cache_resource
def get_optimizer():
    return BudgetOptimizer(FITTED_PARAMS)

optimizer = get_optimizer()

# Sidebar Controls
st.sidebar.title("⚖️ Meridian-Lite MMM")
st.sidebar.markdown("**Enterprise Marketing Mix Modeling & Budget Optimization Platform**")
st.sidebar.divider()
st.sidebar.markdown("### Client Profile")
st.sidebar.info("**Brand:** Aura Wellness DTC\n\n**GMV:** $12.4M / year\n\n**Channels:** 6 Paid Networks")

# Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Executive Summary & Attribution",
    "📈 Saturation & Marginal Response",
    "🎛️ Live Scenario Simulator",
    "📑 Executive Audit Exporter"
])

# ==========================================
# TAB 1: EXECUTIVE SUMMARY & ATTRIBUTION
# ==========================================
with tab1:
    st.markdown("## Executive Performance & Attribution Audit")
    st.markdown("Independent Bayesian Econometric evaluation isolating true incremental lift vs. ad network claimed returns.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""<div class="metric-card"><div class="kpi-title">Model Explanatory Power</div><div class="kpi-val">90.53%</div><div class="kpi-sub">R² = 0.9053</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="metric-card"><div class="kpi-title">Model Tracking Error</div><div class="kpi-val">3.57%</div><div class="kpi-sub">MAPE = 3.57%</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="metric-card"><div class="kpi-title">Total Daily Ad Spend</div><div class="kpi-val">$8,085</div><div class="kpi-sub">Across 6 Paid Channels</div></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""<div class="metric-card"><div class="kpi-title">Identified Misallocation</div><div class="kpi-val">$173,077</div><div class="kpi-sub">+100% to +158% Over-attribution</div></div>""", unsafe_allow_html=True)

    st.divider()

    # Attribution Gap Bar Chart
    attr_data = []
    for ch in optimizer.channels:
        spend = CURRENT_DAILY_SPEND[ch] * 728
        rev = optimizer.channel_daily_revenue(CURRENT_DAILY_SPEND[ch], ch) * 728
        iroas = rev / spend
        claimed = CLAIMED_ROAS[ch]
        gap = ((claimed - iroas) / iroas) * 100
        attr_data.append({
            "Channel": ch,
            "Total Spend ($)": spend,
            "Incremental Revenue ($)": rev,
            "True iROAS": round(iroas, 2),
            "Claimed ROAS": claimed,
            "Over-Attribution Gap": f"{gap:+.1f}%"
        })
    df_attr = pd.DataFrame(attr_data)

    c_left, c_right = st.columns([1.2, 1.0])
    with c_left:
        st.markdown("### The Commercial Attribution Gap")
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(df_attr))
        w = 0.35
        ax.bar(x - w/2, df_attr["Claimed ROAS"], w, label="In-Platform Claimed ROAS", color="#f97316", alpha=0.9)
        ax.bar(x + w/2, df_attr["True iROAS"], w, label="True Incremental ROAS (MMM)", color="#16a34a", alpha=0.9)
        ax.axhline(1.0, color="black", linestyle="--", alpha=0.6, label="Break-Even (1.0x)")
        ax.set_xticks(x)
        ax.set_xticklabels(df_attr["Channel"], rotation=25, ha="right", fontsize=9)
        ax.set_ylabel("ROAS Multiple")
        ax.legend(loc="upper right", fontsize=8.5)
        st.pyplot(fig)

    with c_right:
        st.markdown("### Channel Attribution Audit Breakdown")
        st.dataframe(df_attr[["Channel", "True iROAS", "Claimed ROAS", "Over-Attribution Gap"]], use_container_width=True, hide_index=True)
        st.warning("**Key Finding:** Google P-Max claims 3.10x ROAS but drives only **1.20x incremental return**, actively destroying profit margins at current spend levels.")

# ==========================================
# TAB 2: SATURATION CURVES & MARGINAL ROAS
# ==========================================
with tab2:
    st.markdown("## Diminishing Returns & Hill Saturation Dynamics")
    st.markdown("Non-linear response curves illustrating where each dollar enters diminishing marginal returns.")

    spend_grid = np.linspace(0, 5000, 300)
    selected_ch = st.selectbox("Select Marketing Channel to Inspect:", optimizer.channels, index=3) # Default Google P-Max

    p = FITTED_PARAMS[selected_ch]
    rev_curve = p["beta_max"] * hill_saturation(spend_grid, half_sat_k=p["k_half_sat"], shape_gamma=p["gamma"])
    mroas_curve = p["beta_max"] * hill_derivative(spend_grid, half_sat_k=p["k_half_sat"], shape_gamma=p["gamma"])

    curr_s = CURRENT_DAILY_SPEND[selected_ch]
    curr_rev = optimizer.channel_daily_revenue(curr_s, selected_ch)
    curr_m = optimizer.channel_marginal_roas(curr_s, selected_ch)

    c1, c2 = st.columns(2)
    with c1:
        fig1, ax1 = plt.subplots(figsize=(8, 4.5))
        ax1.plot(spend_grid, rev_curve, color="#2563eb", lw=2.2, label="Revenue Curve")
        ax1.scatter([curr_s], [curr_rev], color="#dc2626", s=90, zorder=5, label=f"Current Spend (${curr_s:,.0f}/day)")
        ax1.scatter([p["k_half_sat"]], [p["beta_max"]*0.5], color="#16a34a", s=70, zorder=5, label=f"Half-Sat K (${p['k_half_sat']:,.0f})")
        ax1.set_title(f"{selected_ch}: Hill Revenue Saturation", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Daily Spend ($)")
        ax1.set_ylabel("Incremental Daily Revenue ($)")
        ax1.legend(loc="lower right", fontsize=8.5)
        st.pyplot(fig1)

    with c2:
        fig2, ax2 = plt.subplots(figsize=(8, 4.5))
        ax2.plot(spend_grid, mroas_curve, color="#7c3aed", lw=2.2, label="Marginal ROAS (dRev/dSpend)")
        ax2.scatter([curr_s], [curr_m], color="#dc2626", s=90, zorder=5, label=f"Current mROAS ({curr_m:.2f}x)")
        ax2.axhline(1.0, color="black", linestyle="--", alpha=0.7, label="mROAS Break-Even (1.0x)")
        ax2.set_title(f"{selected_ch}: Marginal Return Curve", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Daily Spend ($)")
        ax2.set_ylabel("Marginal ROAS")
        ax2.legend(loc="upper right", fontsize=8.5)
        st.pyplot(fig2)

    st.info(f"**Diagnostic Summary for {selected_ch}:** At current spend of **${curr_s:,.2f}/day**, the marginal ROAS is **{curr_m:.2f}x**. Spending an additional $100 returns **${curr_m*100:,.2f}** in gross revenue.")

# ==========================================
# TAB 3: LIVE SCENARIO SIMULATOR
# ==========================================
with tab3:
    st.markdown("## Live Budget Optimization Simulator")
    st.markdown("Use the slider to set a target budget. The SLSQP solver reallocates spend according to the Equi-Marginal Principle.")

    sim_budget = st.slider("Target Total Daily Marketing Budget ($):", min_value=3000, max_value=15000, value=int(TOTAL_CURRENT_BUDGET), step=250)

    opt_res = optimizer.optimize_budget(total_daily_budget=float(sim_budget))
    df_opt = opt_res["breakdown"]

    current_total_rev = sum(optimizer.channel_daily_revenue(CURRENT_DAILY_SPEND[c], c) for c in optimizer.channels)
    opt_total_rev = opt_res["total_optimal_revenue"]
    lift = opt_total_rev - current_total_rev

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Total Optimal Revenue", f"${opt_total_rev:,.2f}/day")
    with col_b:
        st.metric("Incremental Uplift vs Baseline", f"{lift:+,.2f}/day", delta=f"{lift*365.25:+,.0f}/year")
    with col_c:
        st.metric("Optimal Portfolio mROAS", f"{df_opt['Marginal ROAS (mROAS)'].iloc[0]:.2f}x", "Equi-marginal balance")

    st.markdown("### Recommended Capital Allocation Breakdown")
    st.dataframe(df_opt, use_container_width=True, hide_index=True)

# ==========================================
# TAB 4: EXECUTIVE AUDIT EXPORTER
# ==========================================
with tab4:
    st.markdown("## Board-Ready Audit Report Exporter")
    st.markdown("Generate and download the executive HTML audit deliverable with one click.")

    # Calculate baseline audit comparison
    opt_solution = optimizer.optimize_budget(total_daily_budget=TOTAL_CURRENT_BUDGET)
    df_realloc = []
    for ch in optimizer.channels:
        curr_s = CURRENT_DAILY_SPEND[ch]
        curr_r = optimizer.channel_daily_revenue(curr_s, ch)
        opt_row = opt_solution["breakdown"][opt_solution["breakdown"]["Channel"] == ch].iloc[0]
        opt_s = opt_row["Optimal Daily Spend ($)"]
        opt_r = opt_row["Projected Daily Revenue ($)"]
        df_realloc.append({
            "Channel": ch,
            "Current Spend ($/day)": curr_s,
            "Optimal Spend ($/day)": opt_s,
            "Recommended Shift ($)": opt_s - curr_s,
            "Current mROAS": optimizer.channel_marginal_roas(curr_s, ch),
            "Optimal mROAS": opt_row["Marginal ROAS (mROAS)"],
            "Incremental Lift ($/day)": opt_r - curr_r
        })
    df_realloc_table = pd.DataFrame(df_realloc)
    daily_gain = opt_solution["total_optimal_revenue"] - current_total_rev

    generator = AuditReportGenerator(
        company_name="Aura Wellness DTC",
        diagnostics={"r2_score": 0.9053, "mape_pct": 3.57, "rmse": 1914.87},
        df_attribution=df_attr,
        df_reallocation=df_realloc_table,
        annual_uplift=daily_gain * 365.25,
        daily_uplift=daily_gain
    )

    report_html = generator.generate_html()
    report_md = generator.generate_markdown()

    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            label="📥 Download Executive HTML Audit Report",
            data=report_html,
            file_name="Aura_Wellness_Incrementality_Audit.html",
            mime="text/html",
            use_container_width=True
        )
    with d2:
        st.download_button(
            label="📄 Download Markdown Pitch Deck Summary",
            data=report_md,
            file_name="Aura_Wellness_Incrementality_Audit.md",
            mime="text/markdown",
            use_container_width=True
        )

    st.divider()
    st.markdown("### Report In-Browser Preview")
    st.components.v1.html(report_html, height=650, scrolling=True)
