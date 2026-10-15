import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_ground_truth_dataset(
    start_date: str = "2024-01-01",
    n_days: int = 728,
    random_seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Synthesizes 2 years of multi-channel ad spend and e-commerce revenue
    with embedded ground-truth Adstock and Hill parameters.
    """
    np.random.seed(random_seed)
    date_range = pd.date_range(start=start_date, periods=n_days, freq="D")

    # 1. Channel Definitions & Ground-Truth Physics
    # lambda_decay: Adstock memory [0, 1)
    # k_half_sat: Spend ($/day) where response reaches 50% capacity
    # gamma: S-curve slope parameter
    # beta_max: Max incremental daily revenue cap
    channel_configs = {
        "Meta_Prospecting": {
            "channel_id": "CH_META_PROSP",
            "category": "Paid Social",
            "base_spend": 2200.0,
            "spend_volatility": 450.0,
            "cpm": 18.50,
            "ctr": 0.014,
            "lambda_decay": 0.55,
            "k_half_sat": 3500.0,
            "gamma": 1.25,
            "beta_max": 7500.0
        },
        "Meta_Retargeting": {
            "channel_id": "CH_META_RETARG",
            "category": "Paid Social",
            "base_spend": 750.0,
            "spend_volatility": 180.0,
            "cpm": 26.00,
            "ctr": 0.028,
            "lambda_decay": 0.20,
            "k_half_sat": 900.0,     # Saturates very quickly
            "gamma": 0.90,
            "beta_max": 2400.0
        },
        "Google_Brand_Search": {
            "channel_id": "CH_GOOG_BRAND",
            "category": "Paid Search",
            "base_spend": 600.0,
            "spend_volatility": 120.0,
            "cpm": 42.00,
            "ctr": 0.085,
            "lambda_decay": 0.10,     # Near-zero carryover
            "k_half_sat": 800.0,      # Fast saturation (captures existing intent)
            "gamma": 0.85,
            "beta_max": 3100.0
        },
        "Google_PMax_NonBrand": {
            "channel_id": "CH_GOOG_PMAX",
            "category": "Paid Search",
            "base_spend": 1900.0,
            "spend_volatility": 380.0,
            "cpm": 22.00,
            "ctr": 0.021,
            "lambda_decay": 0.40,
            "k_half_sat": 2800.0,
            "gamma": 1.15,
            "beta_max": 6200.0
        },
        "TikTok_Ads": {
            "channel_id": "CH_TIKTOK",
            "category": "Paid Social",
            "base_spend": 1200.0,
            "spend_volatility": 500.0,
            "cpm": 12.00,
            "ctr": 0.011,
            "lambda_decay": 0.30,
            "k_half_sat": 2200.0,
            "gamma": 1.40,
            "beta_max": 3800.0
        },
        "YouTube_Awareness": {
            "channel_id": "CH_YOUTUBE",
            "category": "Video",
            "base_spend": 1100.0,
            "spend_volatility": 300.0,
            "cpm": 15.00,
            "ctr": 0.007,
            "lambda_decay": 0.70,     # Long memory carryover
            "k_half_sat": 4000.0,     # High saturation ceiling
            "gamma": 1.10,
            "beta_max": 4200.0
        }
    }

    # 2. Exogenous Macro Foundations
    trend_factor = 1.0 + (np.linspace(0, 0.25, n_days)) # 25% organic growth across 2 years

    # Day of week multiplier (Saturday/Sunday dip, Tuesday/Wednesday peak)
    dow_multipliers = {0: 1.05, 1: 1.10, 2: 1.08, 3: 1.02, 4: 0.96, 5: 0.88, 6: 0.91}
    dow_factor = np.array([dow_multipliers[d.weekday()] for d in date_range])

    # Holiday demand shocks
    holiday_factor = np.ones(n_days)
    for i, d in enumerate(date_range):
        # Black Friday / Cyber Monday surge (Late November)
        if d.month == 11 and 23 <= d.day <= 30:
            holiday_factor[i] = 2.40
        # Christmas holiday run-up
        elif d.month == 12 and 10 <= d.day <= 22:
            holiday_factor[i] = 1.65
        # Post-Christmas hangover
        elif d.month == 12 and 25 <= d.day <= 31:
            holiday_factor[i] = 0.75

    base_organic_revenue = 11000.0 * trend_factor * dow_factor * holiday_factor

    # 3. Simulate Spend & Transform through Adstock + Hill Functions
    spend_records = []
    adstocked_signals = {}
    incremental_channel_revenue = {}

    for ch_name, cfg in channel_configs.items():
        # Introduce spend spikes during peak holiday seasons
        spend_holiday_boost = np.where(holiday_factor > 1.2, 1.75, 1.0)
        daily_spend = np.clip(
            np.random.normal(cfg["base_spend"], cfg["spend_volatility"], n_days) * spend_holiday_boost,
            a_min=100.0,
            a_max=None
        )

        # Calculate impressions & clicks based on CPM and CTR
        impressions = (daily_spend / cfg["cpm"]) * 1000.0
        clicks = impressions * cfg["ctr"]

        for t in range(n_days):
            spend_records.append({
                "date": date_range[t],
                "channel_id": cfg["channel_id"],
                "channel_name": ch_name,
                "spend": round(float(daily_spend[t]), 2),
                "impressions": int(impressions[t]),
                "clicks": int(clicks[t])
            })

        # Recursive Geometric Adstock computation
        adstock_series = np.zeros(n_days)
        decay = cfg["lambda_decay"]
        for t in range(n_days):
            if t == 0:
                adstock_series[t] = daily_spend[t]
            else:
                adstock_series[t] = daily_spend[t] + (decay * adstock_series[t-1])
        adstocked_signals[ch_name] = adstock_series

        # Hill Saturation Transformation
        K = cfg["k_half_sat"]
        gamma = cfg["gamma"]
        hill_transformed = (adstock_series ** gamma) / ((K ** gamma) + (adstock_series ** gamma))

        # Incremental revenue driven by channel
        channel_rev = cfg["beta_max"] * hill_transformed * dow_factor * np.sqrt(holiday_factor)
        incremental_channel_revenue[ch_name] = channel_rev

    df_spend = pd.DataFrame(spend_records)

    # 4. Synthesize Total Conversions & Orders
    total_incremental = sum(incremental_channel_revenue.values())
    unexplained_noise = np.random.normal(0, 850.0, n_days)
    total_gross_revenue = base_organic_revenue + total_incremental + unexplained_noise

    # Average Order Value (AOV) ~$115
    aov = 115.0 + np.random.normal(0, 3.0, n_days)
    total_orders = np.round(total_gross_revenue / aov).astype(int)

    revenue_records = []
    for t in range(n_days):
        revenue_records.append({
            "date": date_range[t],
            "total_orders": int(total_orders[t]),
            "gross_revenue": round(float(total_gross_revenue[t]), 2),
            "baseline_organic_revenue": round(float(base_organic_revenue[t]), 2),
            "true_incremental_revenue": round(float(total_incremental[t]), 2),
            "noise_component": round(float(unexplained_noise[t]), 2)
        })

    df_revenue = pd.DataFrame(revenue_records)

    return df_spend, df_revenue, channel_configs

if __name__ == "__main__":
    df_spend, df_revenue, configs = generate_ground_truth_dataset()
    df_spend.to_parquet("marketing_incrementality_mmm/data/raw/raw_spend.parquet", index=False)
    df_revenue.to_parquet("marketing_incrementality_mmm/data/raw/raw_revenue.parquet", index=False)
    print(f"Dataset generated: {len(df_spend)} spend rows across {len(df_revenue)} days.")
