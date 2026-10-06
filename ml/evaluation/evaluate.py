"""
Comprehensive Evaluation Metrics for SmartFolio.
Separates statistical prediction evaluation from decision-quality portfolio performance metrics,
and implements block bootstrap confidence intervals.
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from config.settings import RANDOM_SEED, RISK_FREE_RATE_ANNUAL

def compute_prediction_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute statistical prediction accuracy metrics.
    - MAE: Mean Absolute Error
    - RMSE: Root Mean Squared Error
    - R2: Coefficient of determination
    - Directional Accuracy: Proportion of time predicted return sign matches actual sign
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - (ss_res / (ss_tot + 1e-8)))
    
    # Directional accuracy: sign agreement
    correct_dir = np.sign(y_true) == np.sign(y_pred)
    dir_acc = float(np.mean(correct_dir)) if len(correct_dir) > 0 else 0.5
    
    return {
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "r2": round(r2, 4),
        "directional_accuracy": round(dir_acc, 4)
    }

def compute_portfolio_metrics(
    portfolio_returns: np.ndarray,
    periods_per_year: int = 252,
    risk_free_rate: float = RISK_FREE_RATE_ANNUAL
) -> Dict[str, float]:
    """
    Compute financial portfolio decision-quality metrics:
    - Annualized Return
    - Annualized Volatility
    - Sharpe Ratio
    - Sortino Ratio (downside deviation based)
    - Maximum Drawdown (MDD)
    - Calmar Ratio
    """
    returns = np.asarray(portfolio_returns, dtype=float)
    if len(returns) == 0 or np.all(returns == 0):
        return {
            "cumulative_return": 0.0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0
        }
    
    cum_return = float(np.prod(1.0 + returns) - 1.0)
    n_years = len(returns) / periods_per_year
    ann_return = float((1.0 + cum_return) ** (1.0 / max(n_years, 0.01)) - 1.0) if cum_return > -1.0 else -1.0
    
    ann_vol = float(np.std(returns, ddof=1) * np.sqrt(periods_per_year)) if len(returns) > 1 else 0.0
    
    # Sharpe Ratio
    rf_daily = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess_returns = returns - rf_daily
    sharpe = float(np.mean(excess_returns) / (np.std(returns, ddof=1) + 1e-8) * np.sqrt(periods_per_year)) if ann_vol > 0 else 0.0
    
    # Sortino Ratio (Downside deviation)
    downside_returns = np.minimum(returns - rf_daily, 0.0)
    downside_std = float(np.sqrt(np.mean(downside_returns ** 2)) * np.sqrt(periods_per_year))
    sortino = float((ann_return - risk_free_rate) / (downside_std + 1e-8)) if downside_std > 0 else 0.0
    
    # Maximum Drawdown (MDD)
    cum_wealth = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(cum_wealth)
    drawdowns = (cum_wealth - peak) / (peak + 1e-8)
    max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
    
    # Calmar Ratio
    calmar = float(ann_return / (abs(max_dd) + 1e-8)) if max_dd < 0 else 0.0
    
    return {
        "cumulative_return": round(cum_return, 4),
        "annualized_return": round(ann_return, 4),
        "annualized_volatility": round(ann_vol, 4),
        "sharpe_ratio": round(sharpe, 4),
        "sortino_ratio": round(sortino, 4),
        "max_drawdown": round(max_dd, 4),
        "calmar_ratio": round(calmar, 4)
    }

def bootstrap_sharpe_difference(
    returns_a: np.ndarray,
    returns_b: np.ndarray,
    n_bootstraps: int = 1000,
    block_size: int = 5,
    random_state: int = RANDOM_SEED
) -> Tuple[float, Tuple[float, float]]:
    """
    Stationary block bootstrap to compute 95% confidence interval for Delta Sharpe (Sharpe_A - Sharpe_B).
    Returns (delta_sharpe, (ci_lower, ci_upper)).
    """
    np.random.seed(random_state)
    n = min(len(returns_a), len(returns_b))
    if n < 20:
        return 0.0, (0.0, 0.0)
    
    ret_a = returns_a[:n]
    ret_b = returns_b[:n]
    
    sharpe_a_orig = compute_portfolio_metrics(ret_a)["sharpe_ratio"]
    sharpe_b_orig = compute_portfolio_metrics(ret_b)["sharpe_ratio"]
    delta_orig = sharpe_a_orig - sharpe_b_orig
    
    diff_bootstraps = []
    num_blocks = n // block_size
    
    for _ in range(n_bootstraps):
        # Sample starting indices for blocks
        block_starts = np.random.randint(0, n - block_size + 1, size=num_blocks)
        indices = np.concatenate([np.arange(st, st + block_size) for st in block_starts])
        
        sample_a = ret_a[indices]
        sample_b = ret_b[indices]
        
        s_a = compute_portfolio_metrics(sample_a)["sharpe_ratio"]
        s_b = compute_portfolio_metrics(sample_b)["sharpe_ratio"]
        diff_bootstraps.append(s_a - s_b)
        
    ci_lower = float(np.percentile(diff_bootstraps, 2.5))
    ci_upper = float(np.percentile(diff_bootstraps, 97.5))
    
    return round(delta_orig, 4), (round(ci_lower, 4), round(ci_upper, 4))
