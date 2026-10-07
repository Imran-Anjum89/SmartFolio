"""
Global Settings and Reproducibility Configuration for SmartFolio.
"""
import os
import random
import numpy as np
import torch

# Reproducibility Seed
RANDOM_SEED = 42

def set_seed(seed: int = RANDOM_SEED):
    """Enforce deterministic behavior across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Initialize seed immediately on module import
set_seed(RANDOM_SEED)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CACHE_DIR = os.path.join(BASE_DIR, "data", "cache")
DB_PATH = os.path.join(BASE_DIR, "smartfolio.db")
UNIVERSE_CONFIG_PATH = os.path.join(BASE_DIR, "config", "universe.json")

# Database Configuration (SQLite default with PostgreSQL support)
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DB_PATH}")

# ML & Feature Hyperparameters
FORECAST_HORIZON_DAYS = 5  # Future return prediction target: (P_{t+5} / P_t) - 1
LOOKBACK_WINDOW_LSTM = 20   # 20 trading days sequence for LSTM
FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_21d",
    "sma_20",
    "sma_50",
    "ema_20",
    "rsi_14",
    "roc_10",
    "macd",
    "macd_signal",
    "volatility_21d",
    "volatility_63d"
]

# Portfolio Constraints
W_MIN = 0.0                      # No short selling
W_MAX = 0.40                     # Maximum allocation per asset (40% cap)
RISK_FREE_RATE_ANNUAL = 0.065     # 6.5% Indian 10-year G-Sec benchmark risk-free rate

# Indian Equity Market Transaction Costs (STT + SEBI + Brokerage + Slippage)
# STT on delivery sale = 0.1%, Brokerage ~ 0.03%, Exchange/SEBI/GST ~ 0.02% -> Approx 15 bps (0.15%)
TRANSACTION_COST_BPS = 15.0      # 15 basis points (0.15% per trade)

# Risk Profile Aversion Parameters (lambda in risk-aware objective: max w^T mu - (lambda/2) w^T Sigma w)
RISK_PROFILES = {
    "conservative": {"risk_aversion": 5.0, "description": "Prioritizes capital preservation & minimum variance"},
    "moderate": {"risk_aversion": 2.0, "description": "Balances expected return and portfolio volatility"},
    "aggressive": {"risk_aversion": 0.5, "description": "Maximizes expected return while accepting higher risk"}
}
