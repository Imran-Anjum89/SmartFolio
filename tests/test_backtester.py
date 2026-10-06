"""
Unit Tests for Walk-Forward Strategy Backtesting and Statistical Validation.
"""
import pytest
import numpy as np
from backend.app.services.data_service import MarketDataService
from backend.app.services.backtest_service import BacktestService
from ml.evaluation.evaluate import bootstrap_sharpe_difference

def test_bootstrap_sharpe_difference():
    # Strategy A clearly outperforms Strategy B deterministically
    ret_a = np.array([0.002] * 100 + [0.0015] * 100)
    ret_b = np.array([0.0001] * 100 + [-0.0002] * 100)

    delta, (ci_low, ci_high) = bootstrap_sharpe_difference(ret_a, ret_b, n_bootstraps=300)
    assert delta > 0
    assert ci_low < ci_high

def test_walk_forward_backtest_execution():
    data_service = MarketDataService()
    backtester = BacktestService(data_service=data_service)

    # Run on small universe subset for speed
    test_symbols = ["TCS.NS", "RELIANCE.NS", "INFY.NS"]
    res = backtester.run_walk_forward_backtest(
        symbols=test_symbols,
        start_date="2022-01-01",
        end_date="2023-01-01",
        rebalance_days=21,
        lookback_window_days=100
    )

    assert "strategies" in res
    assert len(res["strategies"]) == 7
    assert "E0_EqualWeight" in res["strategies"]
    assert "E6_SmartFolioAdaptive" in res["strategies"]
    assert "metrics_summary" in res
    assert "cumulative_wealth" in res
    assert "statistical_validation" in res

    # Verify that metrics are populated
    e6_metrics = res["metrics_summary"]["E6_SmartFolioAdaptive"]
    assert "sharpe_ratio" in e6_metrics
    assert "max_drawdown" in e6_metrics
    assert "annualized_volatility" in e6_metrics
