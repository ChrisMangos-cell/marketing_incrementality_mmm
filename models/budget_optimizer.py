import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Tuple
from marketing_incrementality_mmm.models.transformations import hill_saturation, hill_derivative

class BudgetOptimizer:
    def __init__(self, channel_params: Dict[str, Dict[str, float]]):
        """
        Initializes the optimizer with fitted channel Hill parameters.
        channel_params[channel] = {'k_half_sat': float, 'gamma': float, 'beta_max': float, ...}
        """
        self.channel_params = channel_params
        self.channels = list(channel_params.keys())
        self.n_channels = len(self.channels)

    def channel_daily_revenue(self, spend: float, channel: str) -> float:
        """Computes steady-state incremental daily revenue for a single channel."""
        p = self.channel_params[channel]
        k = p["k_half_sat"]
        gamma = p["gamma"]
        beta = p["beta_max"]
        # Steady-state exposure under continuous daily spend
        sat = hill_saturation(np.array([spend]), half_sat_k=k, shape_gamma=gamma)[0]
        return float(beta * sat)

    def channel_marginal_roas(self, spend: float, channel: str) -> float:
        """Computes instantaneous marginal ROAS (dRev/dSpend) at an exact spend level."""
        p = self.channel_params[channel]
        k = p["k_half_sat"]
        gamma = p["gamma"]
        beta = p["beta_max"]
        slope = hill_derivative(np.array([spend]), half_sat_k=k, shape_gamma=gamma)[0]
        return float(beta * slope)

    def total_daily_revenue(self, spend_vector: np.ndarray) -> float:
        """Objective function: Total incremental daily revenue across all channels."""
        total_rev = 0.0
        for i, ch in enumerate(self.channels):
            total_rev += self.channel_daily_revenue(spend_vector[i], ch)
        return total_rev

    def optimize_budget(
        self,
        total_daily_budget: float,
        min_spend_per_channel: float = 200.0,
        max_spend_per_channel: float = 6000.0
    ) -> Dict:
        """
        Solves constrained non-linear optimization using SLSQP:
        max Σ Revenue(s_i) subject to Σ s_i = total_daily_budget
        """
        # Initial guess: Naive equal split
        x0 = np.full(self.n_channels, total_daily_budget / self.n_channels)

        # Budget constraint: Sum of channel spends must identically equal total_daily_budget
        constraints = ({
            "type": "eq",
            "fun": lambda s: np.sum(s) - total_daily_budget
        })

        # Bounded domains for each channel (guard against negative spend or over-concentration)
        bounds = [(min_spend_per_channel, max_spend_per_channel) for _ in range(self.n_channels)]

        # Scipy minimizes, so we minimize the negative total revenue
        def neg_revenue(s):
            return -self.total_daily_revenue(s)

        res = minimize(
            neg_revenue,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 1000}
        )

        optimal_spends = res.x
        optimal_revenue = -res.fun

        breakdown = []
        for i, ch in enumerate(self.channels):
            s = optimal_spends[i]
            rev = self.channel_daily_revenue(s, ch)
            mroas = self.channel_marginal_roas(s, ch)
            breakdown.append({
                "Channel": ch,
                "Optimal Daily Spend ($)": round(float(s), 2),
                "Projected Daily Revenue ($)": round(float(rev), 2),
                "Simulated aROAS": round(float(rev / s), 2),
                "Marginal ROAS (mROAS)": round(float(mroas), 2)
            })

        return {
            "total_budget": float(total_daily_budget),
            "total_optimal_revenue": round(float(optimal_revenue), 2),
            "blended_optimal_roas": round(float(optimal_revenue / total_daily_budget), 2),
            "breakdown": pd.DataFrame(breakdown)
        }

    def compute_efficient_frontier(
        self,
        budget_range: np.ndarray,
        min_spend_per_channel: float = 200.0,
        max_spend_per_channel: float = 6000.0
    ) -> pd.DataFrame:
        """
        Sweeps across budget levels to generate the Efficient Investment Frontier.
        """
        frontier_rows = []
        for b in budget_range:
            sol = self.optimize_budget(
                total_daily_budget=b,
                min_spend_per_channel=min_spend_per_channel,
                max_spend_per_channel=max_spend_per_channel
            )
            frontier_rows.append({
                "Total Daily Budget ($)": round(float(b), 2),
                "Max Incremental Revenue ($)": sol["total_optimal_revenue"],
                "Blended aROAS": sol["blended_optimal_roas"]
            })
        return pd.DataFrame(frontier_rows)
