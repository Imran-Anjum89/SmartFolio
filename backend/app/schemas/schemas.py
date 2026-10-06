"""
Pydantic Schemas for SmartFolio REST API.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

# Stock Schemas
class AssetInfo(BaseModel):
    symbol: str
    name: str
    sector: str
    currency: str

class UniverseResponse(BaseModel):
    universe_name: str
    version: str
    description: str
    assets: List[AssetInfo]

# Portfolio Optimization Schemas
class PortfolioOptimizeRequest(BaseModel):
    symbols: List[str] = Field(default=["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ITC.NS"])
    total_capital: float = Field(default=100000.0, ge=1000.0)
    strategy: str = Field(default="max_sharpe", description="max_sharpe | min_volatility | risk_aware")
    risk_profile: str = Field(default="moderate", description="conservative | moderate | aggressive")
    model_name: str = Field(default="ensemble", description="baseline | xgboost | lstm | ensemble")

class AssetAllocation(BaseModel):
    symbol: str
    weight: float
    weight_percentage: float
    investment_amount: float
    expected_return: float
    explanation: Optional[List[str]] = None

class PortfolioOptimizeResponse(BaseModel):
    portfolio_id: str
    strategy: str
    risk_profile: str
    risk_aversion: float
    total_capital: float
    expected_portfolio_return: float
    portfolio_volatility: float
    sharpe_ratio: float
    model_used: str
    model_version: str
    allocations: List[AssetAllocation]

# Prediction & ML Schemas
class ModelComparisonResponse(BaseModel):
    symbol: str
    version: str
    best_model: str
    metrics: Dict[str, Dict[str, float]]
    feature_importance: Dict[str, float]

# Feedback Schemas
class PredictionFeedbackRequest(BaseModel):
    symbol: str
    prediction_date: str
    predicted_return: float
    actual_return: float

class RetrainRequest(BaseModel):
    symbol: str = "TCS.NS"

# Backtest Schemas
class BacktestRequest(BaseModel):
    symbols: Optional[List[str]] = None
    start_date: str = "2021-01-01"
    rebalance_days: int = 21
    lookback_window_days: int = 252
