"""
Unit Tests for Market Data Service and Data Source Governance.
"""
import pytest
from backend.app.services.data_service import MarketDataService

def test_universe_loading():
    service = MarketDataService()
    universe = service.universe_config
    assert "universe_name" in universe
    assert len(service.symbols) >= 5
    assert "RELIANCE.NS" in service.symbols
    assert "TCS.NS" in service.symbols

def test_fetch_stock_data_source_tagging():
    service = MarketDataService()
    df, data_source = service.fetch_stock_data("TCS.NS", start_date="2023-01-01", end_date="2023-06-01")
    assert not df.empty
    assert len(df) > 50
    assert data_source in ["REAL_YFINANCE", "SYNTHETIC_SIMULATED"]
    assert "close" in df.columns
    assert "open" in df.columns
    assert "volume" in df.columns
