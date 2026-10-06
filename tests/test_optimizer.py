"""
Unit Tests for Ledoit-Wolf Covariance and Constrained Markowitz Portfolio Optimization.
"""
import pytest
import numpy as np
import pandas as pd
from backend.app.services.risk_service import RiskService
from backend.app.services.optimizer_service import PortfolioOptimizerService

@pytest.fixture
def sample_market_data():
    np.random.seed(42)
    n = 200
    symbols = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ITC.NS"]
    returns_dict = {s: np.random.normal(0.0005, 0.015, n) for s in symbols}
    df_returns = pd.DataFrame(returns_dict)
    return symbols, df_returns

def test_ledoit_wolf_covariance(sample_market_data):
    symbols, df_returns = sample_market_data
    risk_service = RiskService()
    cov_lw, shrinkage = risk_service.compute_ledoit_wolf_covariance(df_returns)
    assert cov_lw.shape == (len(symbols), len(symbols))
    assert 0.0 <= shrinkage <= 1.0
    # Positive semi-definiteness: all eigenvalues >= 0
    eigenvals = np.linalg.eigvalsh(cov_lw)
    assert (eigenvals >= -1e-8).all()

def test_markowitz_constraints(sample_market_data):
    symbols, df_returns = sample_market_data
    risk_service = RiskService()
    opt_service = PortfolioOptimizerService()
    cov_lw, _ = risk_service.compute_ledoit_wolf_covariance(df_returns)

    exp_returns = {s: float(df_returns[s].mean() * 252) for s in symbols}
    
    # Test Max Sharpe Strategy
    res_sharpe = opt_service.optimize_portfolio(exp_returns, cov_lw, strategy="max_sharpe", total_capital=100000.0)
    allocations = res_sharpe["allocations"]

    total_weight = sum(a["weight"] for a in allocations)
    total_invested = sum(a["investment_amount"] for a in allocations)

    assert np.isclose(total_weight, 1.0, atol=1e-3), f"Total weight {total_weight} must be 1.0"
    assert np.isclose(total_invested, 100000.0, atol=5.0)

    # Box constraint: max weight <= 0.40
    for a in allocations:
        assert a["weight"] <= 0.4001, f"Weight {a['weight']} exceeded 40% cap"
        assert a["weight"] >= 0.0, f"Weight {a['weight']} was negative (short-selling prohibited)"

def test_risk_profiles_behavior(sample_market_data):
    symbols, df_returns = sample_market_data
    risk_service = RiskService()
    opt_service = PortfolioOptimizerService()
    cov_lw, _ = risk_service.compute_ledoit_wolf_covariance(df_returns)
    exp_returns = {s: float(df_returns[s].mean() * 252) for s in symbols}

    res_cons = opt_service.optimize_portfolio(exp_returns, cov_lw, strategy="risk_aware", risk_profile="conservative")
    res_aggr = opt_service.optimize_portfolio(exp_returns, cov_lw, strategy="risk_aware", risk_profile="aggressive")

    # Conservative portfolio should have volatility <= Aggressive portfolio
    assert res_cons["portfolio_volatility"] <= res_aggr["portfolio_volatility"] + 1e-4
