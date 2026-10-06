"""
Walk-Forward Strategy Backtesting API Endpoints for SmartFolio.
"""
from fastapi import APIRouter, HTTPException
from backend.app.schemas.schemas import BacktestRequest
from backend.app.services.data_service import MarketDataService
from backend.app.services.backtest_service import BacktestService

router = APIRouter(prefix="/backtesting", tags=["Backtesting"])

data_service = MarketDataService()
backtest_service = BacktestService(data_service=data_service)

@router.post("/run")
def run_backtest_endpoint(req: BacktestRequest):
    """
    Execute walk-forward backtesting comparison (E0 to E6) with turnover and transaction costs.
    Returns cumulative wealth curves and statistical bootstrap validation.
    """
    try:
        results = backtest_service.run_walk_forward_backtest(
            symbols=req.symbols,
            start_date=req.start_date,
            rebalance_days=req.rebalance_days,
            lookback_window_days=req.lookback_window_days
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtesting execution failed: {str(e)}")
