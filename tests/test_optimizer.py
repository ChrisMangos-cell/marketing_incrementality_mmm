import pytest
import numpy as np
from marketing_incrementality_mmm.models.budget_optimizer import BudgetOptimizer

@pytest.fixture(scope="module")
def mock_params():
    return {
        "Meta_Prospecting":    {"k_half_sat": 3500.0, "gamma": 1.25, "beta_max": 7500.0},
        "Meta_Retargeting":    {"k_half_sat": 900.0,  "gamma": 0.90, "beta_max": 2400.0},
        "Google_Brand_Search": {"k_half_sat": 800.0,  "gamma": 0.85, "beta_max": 3100.0},
        "Google_PMax_NonBrand":{"k_half_sat": 2800.0, "gamma": 1.15, "beta_max": 6200.0},
        "TikTok_Ads":          {"k_half_sat": 2200.0, "gamma": 1.40, "beta_max": 3800.0},
        "YouTube_Awareness":   {"k_half_sat": 4000.0, "gamma": 1.10, "beta_max": 4200.0}
    }

def test_budget_conservation(mock_params):
    """Assert that the sum of allocated spend strictly matches the target budget."""
    optimizer = BudgetOptimizer(mock_params)
    target_budget = 10000.0
    solution = optimizer.optimize_budget(total_daily_budget=target_budget)

    total_allocated = solution["breakdown"]["Optimal Daily Spend ($)"].sum()
    assert pytest.approx(total_allocated, rel=1e-4) == target_budget

def test_optimization_lift_over_equal_split(mock_params):
    """Verify that optimal allocation yields strictly higher revenue than an equal split."""
    optimizer = BudgetOptimizer(mock_params)
    target_budget = 8000.0

    # Revenue from naive equal allocation
    naive_spend = np.full(6, target_budget / 6.0)
    naive_revenue = optimizer.total_daily_revenue(naive_spend)

    # Revenue from SLSQP optimization
    sol = optimizer.optimize_budget(total_daily_budget=target_budget)
    assert sol["total_optimal_revenue"] > naive_revenue

def test_spend_bounds_enforced(mock_params):
    """Verify channel bounds [s_min, s_max] are strictly enforced."""
    optimizer = BudgetOptimizer(mock_params)
    min_spend = 300.0
    max_spend = 4000.0
    sol = optimizer.optimize_budget(
        total_daily_budget=9000.0,
        min_spend_per_channel=min_spend,
        max_spend_per_channel=max_spend
    )

    spends = sol["breakdown"]["Optimal Daily Spend ($)"].values
    assert np.all(spends >= min_spend - 1e-4)
    assert np.all(spends <= max_spend + 1e-4)
