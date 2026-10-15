import duckdb
import os

def build_star_schema(
    raw_spend_path: str = "marketing_incrementality_mmm/data/raw/raw_spend.parquet",
    raw_rev_path: str = "marketing_incrementality_mmm/data/raw/raw_revenue.parquet",
    output_dir: str = "marketing_incrementality_mmm/data/processed"
):
    conn = duckdb.connect(database=":memory:")

    # 1. Ingest raw parquet sources
    conn.execute(f"CREATE VIEW stg_spend AS SELECT * FROM read_parquet('{raw_spend_path}')")
    conn.execute(f"CREATE VIEW stg_rev AS SELECT * FROM read_parquet('{raw_rev_path}')")

    # 2. Build Dimension Tables
    conn.execute("""
        CREATE TABLE dim_marketing_channel AS
        SELECT DISTINCT
            channel_id,
            channel_name,
            CASE
                WHEN channel_name LIKE 'Meta%' THEN 'Paid Social'
                WHEN channel_name LIKE 'Google%' THEN 'Paid Search'
                WHEN channel_name = 'TikTok_Ads' THEN 'Paid Social'
                WHEN channel_name = 'YouTube_Awareness' THEN 'Video'
                ELSE 'Other'
            END AS channel_category,
            CASE
                WHEN channel_name LIKE '%Retargeting%' OR channel_name LIKE '%Brand%' THEN 'Lower Funnel'
                ELSE 'Upper / Mid Funnel'
            END AS funnel_tier
        FROM stg_spend
        ORDER BY channel_id;
    """)

    conn.execute("""
        CREATE TABLE dim_calendar AS
        SELECT
            date::DATE AS date_key,
            EXTRACT(year FROM date) AS year,
            EXTRACT(month FROM date) AS month,
            EXTRACT(week FROM date) AS week_of_year,
            EXTRACT(dow FROM date) AS day_of_week,
            DAYNAME(date) AS day_name,
            CASE
                WHEN EXTRACT(month FROM date) = 11 AND EXTRACT(day FROM date) BETWEEN 23 AND 30 THEN 1
                WHEN EXTRACT(month FROM date) = 12 AND EXTRACT(day FROM date) BETWEEN 10 AND 22 THEN 1
                ELSE 0
            END AS is_peak_holiday
        FROM stg_rev
        ORDER BY date_key;
    """)

    # 3. Build Fact Tables
    conn.execute("""
        CREATE TABLE fact_daily_channel_spend AS
        SELECT
            date::DATE AS date_key,
            channel_id,
            spend,
            impressions,
            clicks,
            ROUND(spend / NULLIF(clicks, 0), 2) AS calculated_cpc,
            ROUND((spend / NULLIF(impressions, 0)) * 1000.0, 2) AS calculated_cpm
        FROM stg_spend
        ORDER BY date_key, channel_id;
    """)

    conn.execute("""
        CREATE TABLE fact_daily_revenue AS
        SELECT
            date::DATE AS date_key,
            total_orders,
            gross_revenue,
            baseline_organic_revenue,
            true_incremental_revenue,
            noise_component,
            ROUND(gross_revenue / NULLIF(total_orders, 0), 2) AS calculated_aov
        FROM stg_rev
        ORDER BY date_key;
    """)

    # 4. Build Pivoted MMM Analytics Mart
    conn.execute("""
        CREATE TABLE mart_mmm_matrix AS
        WITH spend_pivot AS (
            SELECT
                date_key,
                SUM(CASE WHEN channel_id = 'CH_META_PROSP' THEN spend ELSE 0 END) AS spend_meta_prosp,
                SUM(CASE WHEN channel_id = 'CH_META_RETARG' THEN spend ELSE 0 END) AS spend_meta_retarg,
                SUM(CASE WHEN channel_id = 'CH_GOOG_BRAND' THEN spend ELSE 0 END) AS spend_goog_brand,
                SUM(CASE WHEN channel_id = 'CH_GOOG_PMAX' THEN spend ELSE 0 END) AS spend_goog_pmax,
                SUM(CASE WHEN channel_id = 'CH_TIKTOK' THEN spend ELSE 0 END) AS spend_tiktok,
                SUM(CASE WHEN channel_id = 'CH_YOUTUBE' THEN spend ELSE 0 END) AS spend_youtube,
                SUM(spend) AS total_paid_spend
            FROM fact_daily_channel_spend
            GROUP BY date_key
        )
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
            r.baseline_organic_revenue,
            r.true_incremental_revenue
        FROM fact_daily_revenue r
        JOIN dim_calendar c ON r.date_key = c.date_key
        JOIN spend_pivot s ON r.date_key = s.date_key
        ORDER BY r.date_key;
    """)

    # Export Tables to Optimized Parquet Marts
    tables = [
        "dim_marketing_channel",
        "dim_calendar",
        "fact_daily_channel_spend",
        "fact_daily_revenue",
        "mart_mmm_matrix"
    ]
    for tbl in tables:
        target_path = os.path.join(output_dir, f"{tbl}.parquet")
        conn.execute(f"COPY {tbl} TO '{target_path}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        row_count = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        print(f"Staged {tbl} -> {target_path} ({row_count:,} rows)")

if __name__ == "__main__":
    build_star_schema()
