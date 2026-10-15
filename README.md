# ⚖️ Meridian-Lite: Enterprise Marketing Mix Modeling & Budget Optimization Platform

An end-to-end econometric measurement and capital allocation platform designed to solve post-iOS 14.5 ad-attribution failure. Bypasses deterministic tracking (cookies/pixels) by implementing **Bayesian Non-Linear Marketing Mix Modeling (MMM)**, **Geometric Adstock Carryover**, **Hill Saturation Dynamics**, and **Constrained Non-Linear Capital Optimization (SLSQP)**.

---

## 📑 Table of Contents
- [Executive Commercial Summary](#executive-commercial-summary)
- [The Commercial Attribution Gap](#the-commercial-attribution-gap)
- [System Architecture & Engineering Flow](#system-architecture--engineering-flow)
- [Mathematical Framework](#mathematical-framework)
  - [1. Geometric Adstock Carryover](#1-geometric-adstock-carryover)
  - [2. Hill Saturation Function](#2-hill-saturation-function)
  - [3. Constrained Budget Optimization (SLSQP)](#3-constrained-budget-optimization-slsqp)
- [DuckDB Star-Schema Data Mart](#duckdb-star-schema-data-mart)
- [Model Diagnostics & Benchmark Recovery](#model-diagnostics--benchmark-recovery)
- [Interactive Streamlit Cockpit](#interactive-streamlit-cockpit)
- [Project Directory Structure](#project-directory-structure)
- [Automated Test Suite (CI/CD)](#automated-test-suite-cicd)
- [Quickstart Guide](#quickstart-guide)

---

## Executive Commercial Summary

| Metric / Dimension | Performance | Strategic Business Takeaway |
| :--- | :---: | :--- |
| **Model Explanatory Power ($R^2$)** | **90.53%** | Explains over 90% of daily revenue variance across 728 continuous days (104 weeks / 2 years). |
| **Mean Absolute Pct Error (MAPE)** | **3.57%** | Daily forecast accuracy within ±$1,914 on an e-commerce business averaging $27,000+/day. |
| **Identified Capital Misallocation** | **$173,077 / yr** | Reallocating budget away from saturated search into video/social unlocks **+4.96%** in incremental revenue with **$0 added ad spend**. |
| **Google P-Max Attribution Gap** | **+158.3%** | Platform claims a 3.10x ROAS, but actual incremental return is **1.20x**, destroying profit margins at current scale. |
| **YouTube Awareness Attribution Gap** | **-29.3%** | Last-click analytics undervalue YouTube (1.45x reported vs. **2.05x true iROAS**) due to high memory carryover ($\lambda = 0.461$). |

---

## The Commercial Attribution Gap

Advertising walled gardens (Meta Ads Manager, Google Ads) operate on attribution windows that claim credit for organic brand momentum. The table below contrasts reported in-platform figures against the independent Bayesian MMM audit:

| Channel | 2-Year Total Spend | Incremental Revenue Driven | True iROAS | In-Platform Claimed ROAS | Over-Attribution Gap | Net P&L Margin Contribution (at 45% Margin) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Google Brand Search** | $456,984 | $1,173,228 | **2.57x** | 6.20x | **+141.2%** | +$70,968 (Modest positive) |
| **Google P-Max / NonBrand** | $1,445,216 | $1,732,912 | **1.20x** | 3.10x | **+158.3%** | **-$665,406 (Capital Destruction)** |
| **Meta Retargeting** | $583,051 | $1,570,603 | **2.69x** | 5.40x | **+100.7%** | +$123,720 (Modest positive) |
| **Meta Prospecting** | $1,666,792 | $2,270,064 | **1.36x** | 2.85x | **+109.6%** | **-$645,263 (Capital Destruction)** |
| **TikTok Ads** | $917,480 | $1,614,856 | **1.76x** | 2.10x | **+19.3%** | -$190,795 (Under Margin Threshold) |
| **YouTube Awareness** | $816,360 | $1,676,678 | **2.05x** | 1.45x | **-29.3%** | -$61,855 (Breakeven Seed Engine) |

---

## System Architecture & Engineering Flow

```text
[ Multi-Channel Paid Ad Spend & Order Logs (728 Days) ]
                         │
                         ▼
[ DuckDB OLAP Analytical Ingestion Layer ]
      ├── dim_marketing_channel      (SCD Type 1 dimensions, funnel tiers)
      ├── dim_calendar               (Date keys, seasonality, peak holiday flags)
      ├── fact_daily_channel_spend   (Spend, clicks, impressions, CPC, CPM)
      ├── fact_daily_revenue         (Total orders, gross revenue, baseline split)
      └── mart_mmm_matrix.parquet    (Pivoted time-series feature mart)
                         │
                         ▼
[ Applied Mathematics & Non-Linear Transformations ]
      ├── Geometric Adstock: x*_t = x_t + λ · x*_{t-1}
      ├── Hill Saturation: S(x*) = (x*)^γ / (K^γ + (x*)^γ)
      └── First-Order Analytic Derivative: dS/dx = γ K^γ (x*)^(γ-1) / (K^γ + (x*)^γ)^2
                         │
                         ▼
[ Model Estimation & Parameter Recovery (L-BFGS-B MAP Loss) ]
      ├── Simultaneous 33-parameter regularized optimization
      └── R² = 0.9053, MAPE = 3.57%, RMSE = $1,914.87
                         │
                         ▼
[ Capital Optimization & Decision Systems (SLSQP) ]
      ├── Equi-marginal allocation: mROAS_m = λ = 0.73
      ├── Reallocation: Siphon $1,245/day from P-Max into TikTok & YouTube
      └── Yield: +$473.86/day (+$173,077/year) with ZERO budget increase
                         │
                         ▼
[ Production Delivery & Visualization (Streamlit) ]
      ├── Interactive budget sliders & real-time re-optimization
      ├── Saturation curves & marginal return derivatives
      └── Automated 1-Click Executive HTML/Markdown Audit Exporter
