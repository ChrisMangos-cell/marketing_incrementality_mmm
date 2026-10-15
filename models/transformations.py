import numpy as np
import pandas as pd
from typing import Union

def geometric_adstock(
    spend: Union[np.ndarray, pd.Series, list],
    decay_rate: float,
    normalize: bool = False
) -> np.ndarray:
    """
    Computes Geometric Adstock carryover decay:
    x*_t = x_t + decay_rate * x*_{t-1}

    Parameters:
    - spend: Array of daily or weekly spend values (non-negative).
    - decay_rate: Retention factor lambda in [0, 1).
                  0.0 = zero memory; 0.8 = long-term persistence.
    - normalize: If True, weights are normalized such that total area under
                 the impulse response equals 1 (spend scale preservation).

    Returns:
    - np.ndarray of adstocked effective exposure values.
    """
    if not 0.0 <= decay_rate < 1.0:
        raise ValueError(f"decay_rate must be in [0, 1), received {decay_rate}")

    x = np.asarray(spend, dtype=np.float64)
    if np.any(x < 0):
        raise ValueError("Spend array cannot contain negative values.")

    n = len(x)
    adstocked = np.zeros(n, dtype=np.float64)

    current_acc = 0.0
    for t in range(n):
        current_acc = x[t] + (decay_rate * current_acc)
        adstocked[t] = current_acc

    if normalize and decay_rate > 0.0:
        # Scale by (1 - lambda) so infinite steady-state equals unit input
        adstocked = adstocked * (1.0 - decay_rate)

    return adstocked


def hill_saturation(
    spend: Union[np.ndarray, pd.Series, list],
    half_sat_k: float,
    shape_gamma: float
) -> np.ndarray:
    """
    Computes Hill function diminishing marginal returns:
    S(x) = (x^gamma) / (K^gamma + x^gamma)

    Parameters:
    - spend: Array of spend or adstocked exposure values (non-negative).
    - half_sat_k: Half-saturation expenditure threshold K > 0 (where S(K) = 0.5).
    - shape_gamma: Curvature parameter gamma > 0.
                   gamma < 1: Concave everywhere (steep initial slope).
                   gamma > 1: S-shaped (inflection point with slow start).

    Returns:
    - np.ndarray bounded in [0.0, 1.0).
    """
    if half_sat_k <= 0:
        raise ValueError(f"half_sat_k must be strictly positive, received {half_sat_k}")
    if shape_gamma <= 0:
        raise ValueError(f"shape_gamma must be strictly positive, received {shape_gamma}")

    x = np.asarray(spend, dtype=np.float64)
    if np.any(x < 0):
        raise ValueError("Spend array cannot contain negative values.")

    # Prevent zero division and floating overflow
    k_pow = np.power(float(half_sat_k), shape_gamma)
    x_pow = np.power(x, shape_gamma)

    # Handle x = 0 cleanly
    denominator = k_pow + x_pow
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(x == 0.0, 0.0, x_pow / denominator)

    return result


def hill_derivative(
    spend: Union[np.ndarray, pd.Series, list],
    half_sat_k: float,
    shape_gamma: float
) -> np.ndarray:
    """
    Computes the first analytical derivative of the Hill function:
    dS/dx = (gamma * K^gamma * x^(gamma - 1)) / (K^gamma + x^gamma)^2

    Used to derive Marginal Return (mROAS) and optimal spend frontiers.
    """
    if half_sat_k <= 0 or shape_gamma <= 0:
        raise ValueError("Parameters half_sat_k and shape_gamma must be strictly positive.")

    x = np.asarray(spend, dtype=np.float64)
    k_pow = np.power(float(half_sat_k), shape_gamma)

    # Edge case: x = 0 with gamma <= 1 can produce division by zero or infinite slope
    with np.errstate(divide='ignore', invalid='ignore'):
        numerator = shape_gamma * k_pow * np.power(x, shape_gamma - 1.0)
        denominator = np.square(k_pow + np.power(x, shape_gamma))
        derivative = np.where(x <= 0.0, 0.0, numerator / denominator)

    return np.nan_to_num(derivative, nan=0.0, posinf=0.0, neginf=0.0)
