import pytest
import numpy as np
import pandas as pd
import duckdb
from marketing_incrementality_mmm.models.bayesian_mmm import MarketingMixModel

@pytest.fixture(scope="module")
def sample_data():
    conn = duckdb.connect()
    df = conn.execute("SELECT * FROM read_parquet('marketing_incrementality_mmm/data/processed/mart_mmm_matrix.parquet') LIMIT 180").fetchdf()
    return df

@pytest.fixture(scope="module")
def channels():
    return [
        "Meta_Prospecting",
        "Meta_Retargeting",
        "Google_Brand_Search",
        "Google_PMax_NonBrand",
        "TikTok_Ads",
        "YouTube_Awareness"
    ]

def test_unfitted_model_guard(channels):
    """Ensure accessing attribution on an unfitted model raises ValueError."""
    model = MarketingMixModel(channel_names=channels)
    with pytest.raises(ValueError):
        model.get_attribution_summary()

def test_model_fit_and_metrics(sample_data, channels):
    """Fit model on slice and ensure diagnostics meet minimum quality thresholds."""
    model = MarketingMixModel(channel_names=channels)
    model.fit(sample_data)

    assert model.is_fitted is True
    assert model.diagnostics["r2_score"] >= 0.88, f"R2 below acceptable threshold: {model.diagnostics['r2_score']}"
    assert model.diagnostics["mape_pct"] <= 12.0, f"MAPE too high: {model.diagnostics['mape_pct']}%"

def test_revenue_conservation(sample_data, channels):
    """Verify that baseline + sum of channels identically equals total predicted revenue."""
    model = MarketingMixModel(channel_names=channels)
    model.fit(sample_data)

    base = model.decomposition["base_organic"]
    media_sum = sum(model.decomposition["channels"].values())
    reconstructed_pred = base + media_sum

    np.testing.assert_array_almost_equal(
        model.diagnostics["y_pred"],
        reconstructed_pred,
        decimal=4,
        err_msg="Revenue decomposition does not sum to total predicted revenue!"
    )
