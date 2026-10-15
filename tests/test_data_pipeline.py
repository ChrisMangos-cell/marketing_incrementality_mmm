import pytest
import duckdb
import pandas as pd

@pytest.fixture(scope="module")
def conn():
    connection = duckdb.connect(database=":memory:")
    connection.execute("CREATE VIEW mart AS SELECT * FROM read_parquet('marketing_incrementality_mmm/data/processed/mart_mmm_matrix.parquet')")
    connection.execute("CREATE VIEW fact_spend AS SELECT * FROM read_parquet('marketing_incrementality_mmm/data/processed/fact_daily_channel_spend.parquet')")
    connection.execute("CREATE VIEW dim_channel AS SELECT * FROM read_parquet('marketing_incrementality_mmm/data/processed/dim_marketing_channel.parquet')")
    yield connection
    connection.close()

def test_temporal_continuity(conn):
    """Assert exactly 728 uninterrupted days with step size of 1 day."""
    dates_df = conn.execute("SELECT date_key FROM mart ORDER BY date_key").fetchdf()
    dates = pd.to_datetime(dates_df["date_key"])
    assert len(dates) == 728, f"Expected 728 days, found {len(dates)}"
    date_diffs = dates.diff().dropna()
    assert (date_diffs == pd.Timedelta(days=1)).all(), "Temporal gap detected in calendar sequence!"

def test_referential_integrity(conn):
    """Ensure all channel foreign keys in fact_spend map cleanly to dim_channel."""
    orphan_count = conn.execute("""
        SELECT COUNT(*)
        FROM fact_spend f
        LEFT JOIN dim_channel d ON f.channel_id = d.channel_id
        WHERE d.channel_id IS NULL
    """).fetchone()[0]
    assert orphan_count == 0, f"Found {orphan_count} orphaned channel foreign keys!"

def test_domain_constraints(conn):
    """Validate physical boundaries: non-negative spend and positive revenue."""
    violations = conn.execute("""
        SELECT COUNT(*)
        FROM mart
        WHERE gross_revenue <= 0
           OR total_paid_spend < 0
           OR spend_meta_prosp < 0
           OR spend_goog_brand < 0
    """).fetchone()[0]
    assert violations == 0, f"Detected {violations} rows violating domain boundaries!"

def test_spend_conservation(conn):
    """Assert sum of column components exactly matches total_paid_spend."""
    mismatches = conn.execute("""
        SELECT COUNT(*)
        FROM mart
        WHERE ROUND(spend_meta_prosp + spend_meta_retarg + spend_goog_brand +
                    spend_goog_pmax + spend_tiktok + spend_youtube, 2) != ROUND(total_paid_spend, 2)
    """).fetchone()[0]
    assert mismatches == 0, f"Detected {mismatches} rows with spend reconciliation mismatches!"
