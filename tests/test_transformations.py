import pytest
import numpy as np
from marketing_incrementality_mmm.models.transformations import (
    geometric_adstock,
    hill_saturation,
    hill_derivative
)

def test_adstock_zero_decay():
    """When lambda = 0, effective spend must identically match raw spend."""
    raw_spend = np.array([100.0, 250.0, 0.0, 400.0])
    transformed = geometric_adstock(raw_spend, decay_rate=0.0)
    np.testing.assert_array_almost_equal(raw_spend, transformed)

def test_adstock_impulse_accumulation():
    """Verify that an initial single spend pulse decays geometrically."""
    pulse = np.array([100.0, 0.0, 0.0, 0.0])
    decay = 0.5
    transformed = geometric_adstock(pulse, decay_rate=decay)
    expected = np.array([100.0, 50.0, 25.0, 12.5])
    np.testing.assert_array_almost_equal(transformed, expected)

def test_adstock_boundary_rejection():
    """Ensure decay parameters outside [0, 1) are rejected."""
    with pytest.raises(ValueError):
        geometric_adstock([100.0], decay_rate=1.0)
    with pytest.raises(ValueError):
        geometric_adstock([100.0], decay_rate=-0.1)

def test_hill_half_saturation_property():
    """When x == K, the Hill response must equal exactly 0.5."""
    K = 1500.0
    for gamma in [0.75, 1.0, 1.5, 2.0]:
        response = hill_saturation([K], half_sat_k=K, shape_gamma=gamma)
        assert pytest.approx(response[0], rel=1e-5) == 0.5

def test_hill_monotonicity():
    """Response must be strictly non-decreasing with respect to spend."""
    spend_grid = np.linspace(0, 10000, 200)
    response = hill_saturation(spend_grid, half_sat_k=2000.0, shape_gamma=1.2)
    diffs = np.diff(response)
    assert np.all(diffs >= 0.0), "Hill function violated monotonic non-decreasing property!"

def test_hill_asymptotic_ceiling():
    """Response must approach, but never exceed 1.0 at high spend."""
    huge_spend = np.array([1e5, 1e6, 1e7])
    response = hill_saturation(huge_spend, half_sat_k=1000.0, shape_gamma=1.0)
    assert np.all(response < 1.0)
    assert np.all(response > 0.99)

def test_hill_derivative_positivity():
    """First derivative must be non-negative everywhere (positive marginal returns)."""
    spend_grid = np.linspace(10, 5000, 100)
    slopes = hill_derivative(spend_grid, half_sat_k=1500.0, shape_gamma=1.2)
    assert np.all(slopes >= 0.0), "Derivative indicated negative marginal return!"
