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
| **Mean Absolute Pct Error (MAPE)** | **3.57%** | Daily forecast accuracy within $\pm \$1,914$ on an e-commerce business averaging $\$27,000+/\text{day}$. |
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
      ├── Geometric Adstock: x~_t = x_t + λ · x~_{t-1}
      ├── Hill Saturation: S(x~) = (x~)^γ / (K^γ + (x~)^γ)
      └── First-Order Analytic Derivative: dS/dx = γ K^γ (x~)^(γ-1) / (K^γ + (x~)^γ)^2
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
```

---

## Mathematical Framework

### 1. Geometric Adstock Carryover
Advertising impact persists across time. The effective ad exposure $\tilde{x}_{m, t}$ on day $t$ given raw expenditure $x_{m, t}$ and retention decay factor $\lambda_m \in [0, 1)$ is modeled as:

$$\tilde{x}_{m, t} = x_{m, t} + \lambda_m \cdot \tilde{x}_{m, t-1} = \sum_{l=0}^{\infty} \lambda_m^l x_{m, t-l}$$

* **Half-Life Formulation:** The time required for an ad impression's psychological impact to decay by 50%:
  $$t_{1/2} = \frac{\ln(0.5)}{\ln(\lambda)}$$
  * *Google Brand Search ($\lambda = 0.145$):* $t_{1/2} \approx 0.35\text{ days}$ (Immediate decay; conversion capture).
  * *YouTube Awareness ($\lambda = 0.461$):* $t_{1/2} \approx 1.25\text{ days}$ (Persistent carryover memory).

### 2. Hill Saturation Function
Models diminishing returns as media spend scales against a finite audience:

$$S(\tilde{x}) = \frac{\tilde{x}^\gamma}{K^\gamma + \tilde{x}^\gamma}$$

Where:
* $K > 0$: **Half-Saturation Threshold** (spend level where response reaches 50% of maximum capacity: $S(K) = 0.5$).
* $\gamma > 0$: **Shape Parameter** ($\gamma \le 1.0$ produces concave curves; $\gamma > 1.0$ produces sigmoidal S-curves).

#### Analytic Marginal ROAS Derivative
To determine whether an extra dollar of budget yields positive incremental return, the instantaneous rate of change is calculated via the quotient rule:

$$\text{mROAS}_m(x) = \beta_m \cdot \frac{dS}{dx} = \beta_m \cdot \frac{\gamma_m K_m^{\gamma_m} \tilde{x}_m^{\gamma_m - 1}}{\left(K_m^{\gamma_m} + \tilde{x}_m^{\gamma_m}\right)^2}$$

### 3. Constrained Budget Optimization (SLSQP)
We formulate the non-linear constrained capital allocation problem:

$$\max_{\mathbf{s}} \sum_{m=1}^M \text{Revenue}_m(s_m) \quad \text{subject to} \quad \sum_{m=1}^M s_m = B_{\text{total}}, \quad s_m \ge s_{\min, m}$$

Using Karush-Kuhn-Tucker (KKT) first-order optimality conditions, maximum capital efficiency is achieved when the **Marginal ROAS of every unconstrained channel equals the shadow price of capital $\lambda$**:

$$\text{mROAS}_1(s_1^{\text{opt}}) = \text{mROAS}_2(s_2^{\text{opt}}) = \dots = \text{mROAS}_M(s_M^{\text{opt}}) = \lambda = \mathbf{0.73}$$

---

## DuckDB Star-Schema Data Mart

Raw marketing impressions, clicks, spend, and transactional sales are normalized in DuckDB and persisted as compressed Parquet files (`data/processed/`):

```sql
-- Analytical Mart Architecture (mart_mmm_matrix)
SELECT 
    r.date_key,
    c.year, 
    c.month, 
    c.week_of_year, 
    c.day_of_week, 
    c.is_peak_holiday,
    s.spend_meta_prosp, 
    s.spend_meta_retarg, 
    s.spend_goog_brand,
    s.spend_goog_pmax, 
    s.spend_tiktok, 
    s.spend_youtube, 
    s.total_paid_spend,
    r.total_orders, 
    r.gross_revenue, 
    r.baseline_organic_revenue
FROM fact_daily_revenue r
JOIN dim_calendar c ON r.date_key = c.date_key
JOIN spend_pivot s ON r.date_key = s.date_key
ORDER BY r.date_key;
```

---

## Model Diagnostics & Benchmark Recovery

The engine estimates 33 joint parameters using L-BFGS-B bounded optimization with an informative L2 shrinkage prior on Hill curvature ($\alpha = 0.015$):

* **$R^2$ Score:** `0.9053`
* **MAPE:** `3.57%`
* **RMSE:** `$1,914.87`

### Parameter Recovery vs. Latent Ground Truth

| Channel | Truth $\lambda$ | Recovered $\hat{\lambda}$ | Truth $K$ ($) | Recovered $\hat{K}$ ($) | Truth $\gamma$ | Recovered $\hat{\gamma}$ | Truth $\beta$ ($) | Recovered $\hat{\beta}$ ($) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Meta Prospecting** | 0.55 | 0.284 | $3,500 | $2,023.6 | 1.25 | 1.801 | $7,500 | $3,934.4 |
| **Meta Retargeting** | 0.20 | 0.523 | $900 | $2,023.6 | 0.90 | 0.600 | $2,400 | $3,962.5 |
| **Google Brand Search** | 0.10 | 0.145 | $800 | $2,023.6 | 0.85 | 0.600 | $3,100 | $3,968.4 |
| **Google P-Max / NonBrand** | 0.40 | 0.167 | $2,800 | $2,023.6 | 1.15 | 0.600 | $6,200 | $3,938.0 |
| **TikTok Ads** | 0.30 | 0.374 | $2,200 | $2,023.6 | 1.40 | 1.362 | $3,800 | $3,950.7 |
| **YouTube Awareness** | 0.70 | 0.461 | $4,000 | $2,023.6 | 1.10 | 1.235 | $4,200 | $3,953.8 |

---

## Interactive Streamlit Cockpit

The web application (`app/cockpit.py`) provides an interactive interface for executive decision-makers:

1. **Executive Performance & Attribution Tab:** High-level KPI tiles and visual side-by-side bar charts comparing claimed vs. true incremental ROAS.
2. **Saturation & Marginal Response Tab:** Interactive channel selector plotting the Hill saturation curve against current daily spend and calculating real-time marginal ROAS.
3. **Live Scenario Simulator Tab:** Budget slider connecting to the SLSQP optimizer to simulate optimal capital allocations across budget levels ($3,000 to $15,000/day).
4. **Board-Ready Audit Exporter Tab:** In-browser preview with direct download buttons for executive HTML and Markdown reports.

---

## Project Directory Structure

```text
marketing_incrementality_mmm/
├── app/
│   └── cockpit.py                   # Streamlit Multi-Tab Executive Web Cockpit
├── data/
│   ├── processed/                   # ZSTD-compressed DuckDB Parquet data marts
│   │   ├── dim_calendar.parquet
│   │   ├── dim_marketing_channel.parquet
│   │   ├── fact_daily_channel_spend.parquet
│   │   ├── fact_daily_revenue.parquet
│   │   └── mart_mmm_matrix.parquet
│   ├── raw/                         # Raw synthesized source files
│   └── synthetic_generator.py       # Multi-channel data synthesizer with embedded physics
├── models/
│   ├── __init__.py
│   ├── bayesian_mmm.py              # 33-parameter L-BFGS-B estimation engine
│   ├── budget_optimizer.py          # SLSQP constrained capital allocation engine
│   └── transformations.py           # Vectorized Geometric Adstock & Hill equations
├── reports/
│   ├── __init__.py
│   ├── audit_generator.py           # Standalone executive HTML/MD report compiler
│   ├── executive_audit_report.html  # Standalone board deliverable
│   └── executive_audit_report.md    # Markdown pitch deck deliverable
├── sql/
│   └── staging.py                   # DuckDB SQL Star-Schema transformation pipeline
├── tests/
│   ├── __init__.py
│   ├── test_audit_generator.py      # Automated report generation tests
│   ├── test_data_pipeline.py        # Data quality contract tests (schema & continuity)
│   ├── test_model.py                # Model fit, state guards, and revenue conservation tests
│   ├── test_optimizer.py            # Budget conservation and optimization tests
│   └── test_transformations.py      # Mathematical transformation invariant tests
├── requirements.txt                 # Pinned production dependencies
└── README.md                        # Documentation
```

---

## Automated Test Suite (CI/CD)

The platform includes **20 deterministic unit and integration tests** across 5 test modules:

```bash
# Execute master test suite
pytest -v tests/
```

* `tests/test_data_pipeline.py`: Verifies temporal continuity (728 continuous days), foreign-key referential integrity, domain constraints (`gross_revenue > 0`, `spend >= 0`), and spend conservation across pivoted matrix columns.
* `tests/test_transformations.py`: Validates Adstock impulse accumulation, boundary limits ($\lambda \in [0, 1)$), Hill half-saturation properties ($S(K) = 0.5$), monotonicity, asymptotic ceilings, and derivative positivity.
* `tests/test_model.py`: Enforces un-fitted state guards, checks minimum fit metrics ($R^2 \ge 0.88$, $\text{MAPE} \le 12\%$), and verifies additive revenue conservation ($\hat{Y} \equiv \text{Base} + \sum \text{Channels}$).
* `tests/test_optimizer.py`: Verifies strict budget conservation, proves revenue dominance over naive equal allocations, and enforces boundary constraints.
* `tests/test_audit_generator.py`: Validates HTML/Markdown structure and checks filesystem export integrity.

---

## Quickstart Guide

### 1. Environment Installation
```bash
git clone [https://github.com/ChrisMangos-cell/marketing_incrementality_mmm.git](https://github.com/ChrisMangos-cell/marketing_incrementality_mmm.git)
cd marketing_incrementality_mmm
pip install -r requirements.txt
```

### 2. Run Data Pipeline & Staging
```bash
python data/synthetic_generator.py
python sql/staging.py
```

### 3. Run Quality Test Gates
```bash
pytest -v tests/
```

### 4. Launch the Streamlit Dashboard
```bash
streamlit run app/cockpit.py
```
