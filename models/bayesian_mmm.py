import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Tuple
from marketing_incrementality_mmm.models.transformations import geometric_adstock, hill_saturation

class MarketingMixModel:
    def __init__(self, channel_names: List[str]):
        self.channel_names = channel_names
        self.n_channels = len(channel_names)
        self.channel_cols = [
            f"spend_{c.lower()}" for c in [
                "meta_prosp", "meta_retarg", "goog_brand", "goog_pmax", "tiktok", "youtube"
            ]
        ]
        self.is_fitted = False
        self.fitted_params = {}
        self.diagnostics = {}
        self.decomposition = {}

    def _predict_components(
        self,
        params: np.ndarray,
        spend_matrix: np.ndarray,
        dow_matrix: np.ndarray,
        holiday_vec: np.ndarray,
        trend_vec: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Decomposes revenue into baseline organic, macro controls, and individual channels.
        """
        baseline = params[0]
        trend_slope = params[1]
        dow_coeffs = np.append(params[2:8], 1.0) # Pin Sunday to 1.0 for identifiability
        holiday_coeff = params[8]

        n_days = spend_matrix.shape[0]
        dow_factor = np.dot(dow_matrix, dow_coeffs)

        # 1. Base Organic Component
        base_organic = baseline * (1.0 + trend_slope * trend_vec) * dow_factor * (1.0 + holiday_coeff * (holiday_vec - 1.0))

        # 2. Channel Incremental Components
        channel_contributions = {}
        total_media = np.zeros(n_days)
        offset = 9

        for m, ch_name in enumerate(self.channel_names):
            decay = params[offset + 4*m]
            k_sat = params[offset + 4*m + 1]
            gamma = params[offset + 4*m + 2]
            beta  = params[offset + 4*m + 3]

            raw_spend = spend_matrix[:, m]
            adstocked = geometric_adstock(raw_spend, decay_rate=decay)
            saturated = hill_saturation(adstocked, half_sat_k=k_sat, shape_gamma=gamma)

            # Channel daily lift
            ch_lift = beta * saturated * dow_factor * np.sqrt(holiday_vec)
            channel_contributions[ch_name] = ch_lift
            total_media += ch_lift

        total_pred = base_organic + total_media
        return total_pred, base_organic, channel_contributions

    def fit(self, df: pd.DataFrame) -> "MarketingMixModel":
        """
        Fits non-linear parameters via constrained L-BFGS-B optimization.
        """
        n_days = len(df)
        spend_matrix = df[self.channel_cols].to_numpy(dtype=np.float64)
        dow_dummies = pd.get_dummies(df["day_of_week"], dtype=float).to_numpy()
        holiday_vec = np.where(df["is_peak_holiday"] == 1, 2.0, 1.0)
        trend_vec = np.linspace(0, 1, n_days)
        y_true = df["gross_revenue"].to_numpy(dtype=np.float64)

        # Parameter Layout:
        # [0]: baseline, [1]: trend, [2:8]: 6 DoW, [8]: holiday
        # For each channel: lambda (decay), K (half-sat), gamma (shape), beta (capacity)
        init_base = [10500.0, 0.22] + [1.05, 1.10, 1.08, 1.02, 0.95, 0.88] + [0.85]
        bounds = [(6000.0, 18000.0), (0.0, 0.50)] + [(0.6, 1.4)]*6 + [(0.1, 2.0)]

        init_channels = []
        for m in range(self.n_channels):
            init_channels.extend([0.35, 2000.0, 1.1, 4000.0])
            bounds.extend([
                (0.02, 0.85),       # lambda in [0.02, 0.85]
                (400.0, 8000.0),    # K in [$400, $8,000]
                (0.60, 2.00),       # gamma in [0.60, 2.00]
                (800.0, 12000.0)    # beta in [$800, $12,000]
            ])

        x0 = np.array(init_base + init_channels)

        def objective(params):
            y_pred, _, _ = self._predict_components(params, spend_matrix, dow_dummies, holiday_vec, trend_vec)
            residuals = y_true - y_pred
            mse = np.mean(np.square(residuals))
            # Priors penalty (shrink extreme gamma parameters toward 1.0)
            reg = 0.015 * np.sum(np.square(params[11::4] - 1.0))
            return mse + reg

        res = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1500, "ftol": 1e-7, "disp": False}
        )

        p = res.x
        self.fitted_params["baseline"] = float(p[0])
        self.fitted_params["trend_slope"] = float(p[1])
        self.fitted_params["holiday_coeff"] = float(p[8])

        channel_results = {}
        offset = 9
        for m, ch_name in enumerate(self.channel_names):
            channel_results[ch_name] = {
                "lambda_decay": round(float(p[offset + 4*m]), 3),
                "k_half_sat": round(float(p[offset + 4*m + 1]), 1),
                "gamma": round(float(p[offset + 4*m + 2]), 3),
                "beta_max": round(float(p[offset + 4*m + 3]), 1)
            }
        self.fitted_params["channels"] = channel_results

        # Diagnostic & Decomposition Capture
        y_fitted, base_organic, ch_contribs = self._predict_components(
            res.x, spend_matrix, dow_dummies, holiday_vec, trend_vec
        )

        ss_res = np.sum(np.square(y_true - y_fitted))
        ss_tot = np.sum(np.square(y_true - np.mean(y_true)))
        r2 = 1.0 - (ss_res / ss_tot)
        mape = np.mean(np.abs((y_true - y_fitted) / y_true)) * 100.0
        rmse = np.sqrt(np.mean(np.square(y_true - y_fitted)))

        self.diagnostics = {
            "r2_score": round(float(r2), 4),
            "mape_pct": round(float(mape), 2),
            "rmse": round(float(rmse), 2),
            "y_pred": y_fitted,
            "y_true": y_true,
            "residuals": y_true - y_fitted
        }

        self.decomposition = {
            "base_organic": base_organic,
            "channels": ch_contribs,
            "spend_matrix": spend_matrix
        }
        self.is_fitted = True
        return self

    def get_attribution_summary(self) -> pd.DataFrame:
        """
        Produces commercial attribution table: Spend, Incremental Revenue, True iROAS.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before running attribution.")

        rows = []
        for m, ch_name in enumerate(self.channel_names):
            total_spend = np.sum(self.decomposition["spend_matrix"][:, m])
            total_revenue_driven = np.sum(self.decomposition["channels"][ch_name])
            iroas = total_revenue_driven / total_spend

            rows.append({
                "Channel": ch_name,
                "Total Spend ($)": round(total_spend, 2),
                "Incremental Revenue ($)": round(total_revenue_driven, 2),
                "True iROAS": round(iroas, 2),
                "Fitted λ": self.fitted_params["channels"][ch_name]["lambda_decay"],
                "Fitted K ($)": self.fitted_params["channels"][ch_name]["k_half_sat"],
                "Fitted γ": self.fitted_params["channels"][ch_name]["gamma"],
                "Fitted β ($)": self.fitted_params["channels"][ch_name]["beta_max"]
            })

        return pd.DataFrame(rows)
