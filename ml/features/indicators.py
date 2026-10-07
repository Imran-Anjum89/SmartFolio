"""
Technical Feature Engineering Pipeline for SmartFolio.
Computes return, trend, momentum, MACD, and volatility indicators without look-ahead bias.
"""
import pandas as pd
import numpy as np
from typing import List, Tuple

def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate financial and technical indicators on OHLCV dataframe.
    Expects columns: ['date', 'open', 'high', 'low', 'close', 'volume'] (case-insensitive).
    Returns dataframe with computed features strictly using information up to time t.
    """
    df = df.copy()
    # Normalize column names
    df.columns = [c.lower() for c in df.columns]
    
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
    
    close = df["close"]
    
    # 1. Return Features (Lagged, backward-looking)
    df["return_1d"] = close.pct_change(1)
    df["return_5d"] = close.pct_change(5)
    df["return_21d"] = close.pct_change(21)
    
    # 2. Trend Features (Moving averages normalized by close price)
    sma_20 = close.rolling(window=20).mean()
    sma_50 = close.rolling(window=50).mean()
    ema_20 = close.ewm(span=20, adjust=False).mean()
    
    df["sma_20"] = (close - sma_20) / (sma_20 + 1e-8)
    df["sma_50"] = (close - sma_50) / (sma_50 + 1e-8)
    df["ema_20"] = (close - ema_20) / (ema_20 + 1e-8)
    
    # 3. Momentum Features: RSI 14
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    df["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))
    
    # Rate of Change (ROC 10)
    df["roc_10"] = close.pct_change(10) * 100.0
    
    # 4. MACD Features (12-day EMA - 26-day EMA, and 9-day signal EMA)
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    
    df["macd"] = macd_line / (close + 1e-8)
    df["macd_signal"] = signal_line / (close + 1e-8)
    
    # 5. Risk / Volatility Features (Annualized rolling standard deviation of 1d returns)
    df["volatility_21d"] = df["return_1d"].rolling(window=21).std() * np.sqrt(252)
    df["volatility_63d"] = df["return_1d"].rolling(window=63).std() * np.sqrt(252)
    
    # Average True Range (ATR 14)
    if "high" in df.columns and "low" in df.columns:
        prev_close = close.shift(1)
        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - prev_close).abs()
        tr3 = (df["low"] - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr_14"] = (tr.rolling(window=14).mean()) / (close + 1e-8)
    else:
        df["atr_14"] = df["volatility_21d"] / np.sqrt(252)

    return df

FEATURE_GROUPS = {
    "price": ["return_1d", "return_5d", "return_21d"],
    "technical": ["sma_20", "sma_50", "ema_20", "rsi_14", "roc_10", "macd", "macd_signal"],
    "volatility": ["volatility_21d", "volatility_63d", "atr_14"],
    "market": ["nifty_return_1d", "nifty_volatility_21d", "vix_level"],
    "relative": ["rel_return_nifty_5d", "rel_volatility_nifty"],
    "sector": ["sector_rel_return_5d"]
}

def generate_target(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """
    Generate future return target: FutureReturn_{h} = (Close_{t+h} / Close_t) - 1.
    Strictly shifts forward so that target_h represents returns over next h trading days.
    """
    df = df.copy()
    df[f"target_{horizon}d"] = df["close"].shift(-horizon) / df["close"] - 1.0
    return df

def generate_multi_horizon_targets(df: pd.DataFrame, horizons: List[int] = [1, 5, 20]) -> pd.DataFrame:
    """
    Generate multiple future return horizons (e.g. 1d, 5d, 20d) for multi-horizon research (Section 8).
    """
    df = df.copy()
    for h in horizons:
        df[f"target_{h}d"] = df["close"].shift(-h) / df["close"] - 1.0
    return df

def prepare_feature_target_dataset(df: pd.DataFrame, feature_cols: List[str], horizon: int = 5) -> pd.DataFrame:
    """
    Full pipeline to compute technical indicators, target, and drop NaN boundary rows.
    """
    df_feat = compute_technical_indicators(df)
    df_with_target = generate_target(df_feat, horizon=horizon)
    # Drop rows where features or target are NaN
    valid_cols = feature_cols + [f"target_{horizon}d", "close", "date"] if "date" in df.columns else feature_cols + [f"target_{horizon}d", "close"]
    cleaned = df_with_target.dropna(subset=[c for c in valid_cols if c in df_with_target.columns]).reset_index(drop=True)
    return cleaned
