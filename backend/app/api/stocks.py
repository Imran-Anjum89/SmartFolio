"""
Stock Market Data and Technical Indicator Endpoints for SmartFolio.
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List
from backend.app.services.data_service import MarketDataService
from backend.app.services.feature_service import FeatureService
from backend.app.schemas.schemas import UniverseResponse

router = APIRouter(prefix="/stocks", tags=["Stocks"])

data_service = MarketDataService()
feature_service = FeatureService()

@router.get("/universe", response_model=UniverseResponse)
def get_universe():
    """Retrieve predefined liquid NIFTY 50 stock universe."""
    return data_service.universe_config

@router.get("/history/{symbol}")
def get_stock_history(
    symbol: str,
    start_date: str = "2022-01-01",
    end_date: Optional[str] = None
):
    """Retrieve historical OHLCV series for a stock."""
    try:
        df, data_source = data_service.fetch_stock_data(symbol, start_date=start_date, end_date=end_date)
        records = df.to_dict(orient="records")
        for r in records:
            if "date" in r and hasattr(r["date"], "strftime"):
                r["date"] = r["date"].strftime("%Y-%m-%d")
        return {
            "symbol": symbol,
            "data_source": data_source,
            "total_bars": len(records),
            "history": records
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/indicators/{symbol}")
def get_stock_indicators(symbol: str):
    """Retrieve computed technical indicators for charting (RSI, MACD, Moving Averages)."""
    try:
        df, data_source = data_service.fetch_stock_data(symbol)
        df_feat = feature_service.build_features_for_stock(df, include_target=False)
        records = df_feat.to_dict(orient="records")
        for r in records:
            if "date" in r and hasattr(r["date"], "strftime"):
                r["date"] = r["date"].strftime("%Y-%m-%d")
        return {
            "symbol": symbol,
            "data_source": data_source,
            "indicators": records[-100:]  # Return latest 100 bars for responsive frontend rendering
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
