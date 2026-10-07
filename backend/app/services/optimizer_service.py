"""
Markowitz Portfolio Optimizer Service for SmartFolio.
Solves constrained mean-variance portfolio optimization problems:
- Maximum Sharpe Ratio
- Minimum Volatility
- Risk-Aware Optimization (Utility balancing with risk aversion lambda)
Enforces: sum(w_i) = 1.0, 0 <= w_i <= 0.40.
"""
import numpy as np
import scipy.optimize as sco
from typing import Dict, List, Tuple, Any, Optional
from config.settings import W_MIN, W_MAX, RISK_FREE_RATE_ANNUAL, RISK_PROFILES

class PortfolioOptimizerService:
    def __init__(
        self,
        risk_free_rate: float = RISK_FREE_RATE_ANNUAL,
        w_min: float = W_MIN,
        w_max: float = W_MAX
    ):
        self.rf = risk_free_rate
        self.w_min = w_min
        self.w_max = w_max

    def optimize_portfolio(
        self,
        expected_returns: Dict[str, float],
        cov_matrix: np.ndarray,
        strategy: str = "max_sharpe",
        risk_profile: str = "moderate",
        total_capital: float = 100000.0,
        prev_weights: Optional[np.ndarray] = None,
        cost_penalty_gamma: float = 0.0,
        fee_rate: float = 0.0015,
        sector_map: Optional[Dict[str, str]] = None,
        max_sector_weight: float = 0.35
    ) -> Dict[str, Any]:
        """
        Optimize asset allocation weights based on expected returns and covariance.
        Strategies: 'max_sharpe', 'min_volatility', 'risk_aware'.
        Includes Cost-Aware optimization with turnover penalty (Section 17).
        """
        symbols = list(expected_returns.keys())
        n = len(symbols)
        if n == 0:
            raise ValueError("No assets provided for optimization.")

        mu = np.array([expected_returns[s] for s in symbols], dtype=float)
        
        # Initial guess: equal weights or previous weights
        init_w = prev_weights if (prev_weights is not None and len(prev_weights) == n) else np.full(n, 1.0 / n)
        bounds = tuple((self.w_min, self.w_max) for _ in range(n))
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        # Optional Sector Constraints (Section 18)
        if sector_map is not None:
            unique_sectors = set(sector_map.get(s, "Other") for s in symbols)
            for sec in unique_sectors:
                sec_indices = [i for i, s in enumerate(symbols) if sector_map.get(s, "Other") == sec]
                if len(sec_indices) > 0 and len(sec_indices) < n:
                    constraints.append({
                        "type": "ineq",
                        "fun": lambda w, idxs=sec_indices: max_sector_weight - np.sum(w[idxs])
                    })

        # Risk Aversion parameter from profile
        profile_cfg = RISK_PROFILES.get(risk_profile.lower(), RISK_PROFILES["moderate"])
        risk_aversion = profile_cfg["risk_aversion"]

        w_old = prev_weights if (prev_weights is not None and len(prev_weights) == n) else np.zeros(n)

        if strategy == "min_volatility":
            def objective(w):
                cost_term = (cost_penalty_gamma * fee_rate * np.sum(np.abs(w - w_old))) if cost_penalty_gamma > 0 else 0.0
                return float(w.T @ cov_matrix @ w + cost_term)
        elif strategy == "risk_aware":
            def objective(w):
                port_ret = float(w.T @ mu)
                port_var = float(w.T @ cov_matrix @ w)
                cost_term = (cost_penalty_gamma * fee_rate * np.sum(np.abs(w - w_old))) if cost_penalty_gamma > 0 else 0.0
                # Maximize w^T mu - (lambda / 2) w^T Sigma w - gamma * Cost -> Minimize negative
                return -(port_ret - (risk_aversion / 2.0) * port_var - cost_term)
        else:  # default 'max_sharpe'
            def objective(w):
                port_ret = float(w.T @ mu)
                port_vol = float(np.sqrt(max(w.T @ cov_matrix @ w, 1e-8)))
                cost_term = (cost_penalty_gamma * fee_rate * np.sum(np.abs(w - w_old))) if cost_penalty_gamma > 0 else 0.0
                # Maximize Sharpe Ratio minus turnover friction
                return -((port_ret - self.rf - cost_term) / port_vol)

        # Solve optimization
        opt_res = sco.minimize(
            objective,
            init_w,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-9}
        )

        if not opt_res.success:
            # Fallback to equal weighting if solver has difficulty with edge cases
            raw_weights = init_w
        else:
            raw_weights = opt_res.x

        # Clean numerical precision and re-normalize
        weights = np.clip(raw_weights, self.w_min, self.w_max)
        weights = weights / np.sum(weights)

        # Portfolio metrics
        port_ret = float(weights.T @ mu)
        port_vol = float(np.sqrt(max(weights.T @ cov_matrix @ weights, 1e-8)))
        port_sharpe = float((port_ret - self.rf) / port_vol) if port_vol > 0 else 0.0

        # Build monetary asset breakdown table
        asset_allocations = []
        for i, sym in enumerate(symbols):
            w = float(weights[i])
            invested_amount = round(w * total_capital, 2)
            asset_allocations.append({
                "symbol": sym,
                "weight": round(w, 4),
                "weight_percentage": round(w * 100.0, 2),
                "investment_amount": invested_amount,
                "expected_return": round(float(mu[i]), 4)
            })

        # Sort asset breakdown by weight descending
        asset_allocations.sort(key=lambda x: x["weight"], reverse=True)

        turnover_val = float(np.sum(np.abs(weights - w_old))) if prev_weights is not None else 0.0
        est_cost = float(turnover_val * fee_rate * total_capital)

        return {
            "strategy": strategy,
            "risk_profile": risk_profile,
            "risk_aversion": risk_aversion,
            "total_capital": total_capital,
            "expected_portfolio_return": round(port_ret, 4),
            "portfolio_volatility": round(port_vol, 4),
            "sharpe_ratio": round(port_sharpe, 4),
            "turnover": round(turnover_val, 4),
            "estimated_cost_inr": round(est_cost, 2),
            "cost_aware": bool(cost_penalty_gamma > 0),
            "constraints": {
                "sum_weights": round(float(np.sum(weights)), 4),
                "max_weight_constraint": self.w_max,
                "min_weight_constraint": self.w_min
            },
            "allocations": asset_allocations
        }
