"""
Command-Line Experiment Runner for SmartFolio (Weeks 1 to 6).
Evaluates foundational research experiments:
  E0: Equal Weight Benchmark (1/N)
  E1: Classical Markowitz (Historical Mean + Sample Covariance)
  E2: Markowitz + Ledoit-Wolf Shrinkage (Historical Mean + Ledoit-Wolf Covariance)
  E3: XGBoost + Markowitz (XGBoost Regressor + Ledoit-Wolf Covariance)
  E4: PyTorch LSTM + Markowitz (PyTorch LSTM + Ledoit-Wolf Covariance)

Usage:
    python run_experiment.py --experiment E0
    python run_experiment.py --experiment E3
"""
import sys
import os
import argparse
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.app.services.data_service import MarketDataService
from backend.app.services.risk_service import RiskService
from backend.app.services.ml_service import MLService
from backend.app.services.optimizer_service import PortfolioOptimizerService
from backend.app.services.explainer_service import AllocationExplainerService
from config.settings import RANDOM_SEED, set_seed

def main():
    parser = argparse.ArgumentParser(description="SmartFolio Experiment Runner (Weeks 1 to 6)")
    parser.add_argument("--experiment", type=str, default="E3", choices=["E0", "E1", "E2", "E3", "E4"],
                        help="Experiment to run: E0, E1, E2, E3, E4")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for reproducibility")
    parser.add_argument("--symbols", nargs="+", default=None, help="Stock symbols (default: core NIFTY 50 assets)")
    parser.add_argument("--output_file", type=str, default=None, help="Optional output JSON filepath")

    args = parser.parse_args()
    set_seed(args.seed)

    print("==================================================================")
    print(f"  SmartFolio Research Runner — Experiment: {args.experiment}")
    print(f"  Scope: Weeks 1 to 6 | Seed: {args.seed}")
    print("==================================================================")

    data_service = MarketDataService()
    risk_service = RiskService()
    ml_service = MLService()
    optimizer_service = PortfolioOptimizerService()
    explainer_service = AllocationExplainerService()

    target_symbols = args.symbols if args.symbols else data_service.symbols[:6]
    print(f"Loading market data for: {', '.join(target_symbols)}...")

    price_series_dict = {}
    symbols_raw_data = {}
    for s in target_symbols:
        df, src = data_service.fetch_stock_data(s)
        symbols_raw_data[s] = df
        if "date" in df.columns and "close" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            price_series_dict[s] = df.set_index("date")["close"]

    returns_df = risk_service.compute_asset_returns_matrix(price_series_dict)

    # 1. Covariances
    cov_sample = risk_service.compute_sample_covariance(returns_df)
    cov_lw, shrinkage_delta = risk_service.compute_ledoit_wolf_covariance(returns_df)

    # 2. Expected Returns
    hist_mu = {s: float(returns_df[s].mean() * 252) for s in target_symbols}

    if args.experiment == "E0":
        # E0: Equal Weight
        n = len(target_symbols)
        weights = np.full(n, 1.0 / n)
        mu_vec = np.array([hist_mu[s] for s in target_symbols])
        port_ret = float(weights @ mu_vec)
        port_vol = float(np.sqrt(weights @ cov_sample @ weights))
        port_sharpe = (port_ret - 0.065) / port_vol if port_vol > 0 else 0.0
        allocations = [{"symbol": s, "weight": round(1.0 / n, 4)} for s in target_symbols]
        res = {
            "experiment": "E0_EqualWeight",
            "expected_portfolio_return": round(port_ret, 4),
            "portfolio_volatility": round(port_vol, 4),
            "sharpe_ratio": round(port_sharpe, 4),
            "allocations": allocations
        }

    elif args.experiment == "E1":
        # E1: Classical Markowitz (Hist mean + Sample cov)
        res = optimizer_service.optimize_portfolio(hist_mu, cov_sample, strategy="max_sharpe")
        res["experiment"] = "E1_ClassicalMarkowitz"

    elif args.experiment == "E2":
        # E2: Markowitz + Ledoit-Wolf Shrinkage
        res = optimizer_service.optimize_portfolio(hist_mu, cov_lw, strategy="max_sharpe")
        res["experiment"] = "E2_Markowitz_LedoitWolf"
        res["shrinkage_intensity"] = round(shrinkage_delta, 4)

    elif args.experiment == "E3":
        # E3: XGBoost + Markowitz (Ledoit-Wolf)
        xgb_mu = ml_service.predict_expected_returns(symbols_raw_data, model_name="xgboost")
        res = optimizer_service.optimize_portfolio(xgb_mu, cov_lw, strategy="max_sharpe")
        res["experiment"] = "E3_XGBoost_Markowitz"
        res["model_used"] = "XGBoost Regressor (Week 5)"

    elif args.experiment == "E4":
        # E4: PyTorch LSTM + Markowitz (Ledoit-Wolf)
        lstm_mu = ml_service.predict_expected_returns(symbols_raw_data, model_name="lstm")
        res = optimizer_service.optimize_portfolio(lstm_mu, cov_lw, strategy="max_sharpe")
        res["experiment"] = "E4_PyTorchLSTM_Markowitz"
        res["model_used"] = "PyTorch Sequential LSTM (Week 6)"

    # Add allocation explainability
    explained = explainer_service.explain_allocations(
        allocations=res["allocations"],
        expected_returns=hist_mu,
        cov_matrix=cov_lw,
        symbols=target_symbols
    )
    res["allocations"] = explained

    print(f"\n--- Results for {res.get('experiment', args.experiment)} ---")
    print(f"  Expected Annual Return : {res['expected_portfolio_return'] * 100:.2f}%")
    print(f"  Portfolio Volatility   : {res['portfolio_volatility'] * 100:.2f}%")
    print(f"  Sharpe Ratio           : {res['sharpe_ratio']:.4f}")
    print("\n--- Asset Allocation & Explanations ---")
    for a in res["allocations"]:
        print(f"  {a['symbol']:<12}: {a['weight']*100:>5.1f}% | {'; '.join(a.get('explanation', []))}")

    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\nSaved results to {args.output_file}")

if __name__ == "__main__":
    main()
