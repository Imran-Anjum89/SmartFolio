"""
Risk and Covariance Engine for SmartFolio.
Computes empirical sample covariance, robust Ledoit-Wolf shrinkage covariance,
and portfolio risk metrics.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.covariance import LedoitWolf

class RiskService:
    def __init__(self, periods_per_year: int = 252):
        self.periods_per_year = periods_per_year

    def compute_asset_returns_matrix(self, price_series_dict: Dict[str, pd.Series]) -> pd.DataFrame:
        """Align dates across all asset price series and compute daily percentage returns."""
        df_prices = pd.DataFrame(price_series_dict).dropna()
        df_returns = df_prices.pct_change().dropna()
        return df_returns

    def compute_sample_covariance(self, returns_df: pd.DataFrame) -> np.ndarray:
        """Standard empirical sample covariance matrix (annualized)."""
        cov_sample = returns_df.cov().values * self.periods_per_year
        return cov_sample

    def compute_ledoit_wolf_covariance(self, returns_df: pd.DataFrame) -> Tuple[np.ndarray, float]:
        """
        Ledoit-Wolf shrinkage covariance estimator.
        Shrinks sample covariance toward constant correlation target:
        Sigma_LW = (1 - shrinkage) * S + shrinkage * F.
        Returns (annualized_cov_matrix, shrinkage_intensity).
        """
        lw = LedoitWolf()
        lw.fit(returns_df)
        cov_lw = lw.covariance_ * self.periods_per_year
        shrinkage = float(lw.shrinkage_)
        return cov_lw, shrinkage

    def compute_portfolio_risk(self, weights: np.ndarray, cov_matrix: np.ndarray) -> float:
        """Portfolio annualized volatility: sigma_p = sqrt(w^T Sigma w)."""
        weights = np.asarray(weights, dtype=float)
        port_variance = float(weights.T @ cov_matrix @ weights)
        return float(np.sqrt(max(port_variance, 0.0)))

    def compute_correlation_matrix(self, returns_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Return correlation matrix as a nested dict for easy JSON serialization."""
        corr = returns_df.corr().round(4)
        return corr.to_dict()
