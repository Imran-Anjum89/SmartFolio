"""
Market Data Service for SmartFolio.
Handles yfinance data ingestion, local Parquet/CSV caching, realistic synthetic fallback,
and mandatory data_source metadata tagging.
"""
import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from config.settings import UNIVERSE_CONFIG_PATH, CACHE_DIR, RANDOM_SEED

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.makedirs(CACHE_DIR, exist_ok=True)

class MarketDataService:
    def __init__(self, universe_path: str = UNIVERSE_CONFIG_PATH):
        self.universe_path = universe_path
        self.universe_config = self._load_universe()
        self.symbols = [a["symbol"] for a in self.universe_config.get("assets", [])]

    def _load_universe(self) -> Dict:
        """Load the pre-established stock universe configuration."""
        if os.path.exists(self.universe_path):
            with open(self.universe_path, "r") as f:
                return json.load(f)
        return {"universe_name": "DEFAULT", "assets": []}

    def fetch_stock_data(
        self,
        symbol: str,
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None,
        force_refresh: bool = False
    ) -> Tuple[pd.DataFrame, str]:
        """
        Fetch OHLCV data for a given symbol.
        Returns (DataFrame, data_source) where data_source is either 'REAL_YFINANCE' or 'SYNTHETIC_SIMULATED'.
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        cache_file = os.path.join(CACHE_DIR, f"{symbol.replace('.', '_')}_{start_date}_{end_date}.parquet")
        
        # 1. Try Loading from Local Parquet Cache
        if not force_refresh and os.path.exists(cache_file):
            try:
                df = pd.read_parquet(cache_file)
                data_source = df.attrs.get("data_source", "REAL_YFINANCE")
                logger.info(f"Loaded {symbol} from cache ({data_source}) with {len(df)} rows.")
                return df, data_source
            except Exception as e:
                logger.warning(f"Failed to read cache for {symbol}: {e}")

        # 2. Try Fetching from yfinance
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date, end=end_date, interval="1d", auto_adjust=True)
            
            if not hist.empty and len(hist) > 50:
                hist = hist.reset_index()
                # Rename standard columns
                hist.columns = [c.lower() for c in hist.columns]
                # Ensure date is string/datetime without timezone for clean parquet serialization
                if "date" in hist.columns:
                    hist["date"] = pd.to_datetime(hist["date"]).dt.tz_localize(None)
                
                required_cols = ["date", "open", "high", "low", "close", "volume"]
                hist = hist[[c for c in required_cols if c in hist.columns]]
                hist.attrs["data_source"] = "REAL_YFINANCE"
                
                # Save to cache
                hist.to_parquet(cache_file, index=False)
                logger.info(f"Successfully fetched real market data for {symbol} via yfinance ({len(hist)} rows).")
                return hist, "REAL_YFINANCE"
        except Exception as e:
            logger.warning(f"yfinance fetch failed for {symbol}: {e}. Falling back to realistic synthetic generator.")

        # 3. Fallback to Realistic Synthetic Data Generator
        df_synthetic = self._generate_realistic_synthetic_stock(symbol, start_date, end_date)
        df_synthetic.attrs["data_source"] = "SYNTHETIC_SIMULATED"
        df_synthetic.to_parquet(cache_file, index=False)
        return df_synthetic, "SYNTHETIC_SIMULATED"

    def _generate_realistic_synthetic_stock(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Generates realistic Geometric Brownian Motion (GBM) market data with stochastic volatility.
        Guaranteed to be tagged with data_source = 'SYNTHETIC_SIMULATED'.
        """
        np.random.seed(abs(hash(symbol) + RANDOM_SEED) % (2**31))
        
        dates = pd.date_range(start=start_date, end=end_date, freq="B")  # Business days
        n = len(dates)
        
        # Base parameters reflecting Indian equities
        s0 = 1000.0 + (abs(hash(symbol)) % 2500)
        mu = 0.12 / 252.0  # 12% annual drift
        vol = (0.18 + (abs(hash(symbol)) % 15) * 0.01) / np.sqrt(252.0)  # ~20% annual vol
        
        daily_returns = np.random.normal(mu, vol, n)
        # Add realistic fat-tailed jumps occasionally
        jump_indices = np.random.choice(n, size=int(n * 0.03), replace=False)
        daily_returns[jump_indices] += np.random.normal(0, vol * 2.5, len(jump_indices))
        
        price_path = s0 * np.exp(np.cumsum(daily_returns))
        
        close = price_path
        high = close * (1.0 + np.abs(np.random.normal(0, vol * 0.5, n)))
        low = close * (1.0 - np.abs(np.random.normal(0, vol * 0.5, n)))
        open_p = (low + high) / 2.0 + np.random.normal(0, vol * 0.2, n)
        volume = np.random.lognormal(mean=14, sigma=0.5, size=n).astype(int)
        
        df = pd.DataFrame({
            "date": dates,
            "open": np.round(open_p, 2),
            "high": np.round(high, 2),
            "low": np.round(low, 2),
            "close": np.round(close, 2),
            "volume": volume
        })
        return df

    def fetch_universe_data(
        self,
        symbols: Optional[List[str]] = None,
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None
    ) -> Dict[str, Tuple[pd.DataFrame, str]]:
        """
        Fetch OHLCV data for all specified symbols or full universe.
        Returns dict mapping symbol -> (DataFrame, data_source).
        """
        target_symbols = symbols if symbols else self.symbols
        results = {}
        for s in target_symbols:
            df, src = self.fetch_stock_data(s, start_date=start_date, end_date=end_date)
            results[s] = (df, src)
        return results
