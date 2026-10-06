"""
Unit and Integration Tests for Technical Feature Engineering and Leakage Checks.
"""
import pytest
import pandas as pd
import numpy as np
from ml.features.indicators import compute_technical_indicators, generate_target, prepare_feature_target_dataset
from ml.features.leakage_checks import verify_feature_leakage, verify_target_alignment
from config.settings import FEATURE_COLUMNS, FORECAST_HORIZON_DAYS

@pytest.fixture
def sample_ohlcv():
    np.random.seed(42)
    n = 150
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    close = 100.0 * np.exp(np.cumsum(np.random.normal(0.0005, 0.015, n)))
    high = close * 1.01
    low = close * 0.99
    open_p = (high + low) / 2.0
    volume = np.random.randint(100000, 500000, n)
    return pd.DataFrame({
        "date": dates,
        "open": open_p,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume
    })

def test_compute_technical_indicators(sample_ohlcv):
    df_feat = compute_technical_indicators(sample_ohlcv)
    for col in FEATURE_COLUMNS:
        assert col in df_feat.columns, f"Missing feature column: {col}"
    # Verify RSI is within bounds [0, 100]
    valid_rsi = df_feat["rsi_14"].dropna()
    assert (valid_rsi >= 0.0).all() and (valid_rsi <= 100.0).all()

def test_generate_target(sample_ohlcv):
    df_target = generate_target(sample_ohlcv, horizon=FORECAST_HORIZON_DAYS)
    target_col = f"target_{FORECAST_HORIZON_DAYS}d"
    assert target_col in df_target.columns
    # Check manual alignment at row 10
    expected = (sample_ohlcv.loc[10 + FORECAST_HORIZON_DAYS, "close"] / sample_ohlcv.loc[10, "close"]) - 1.0
    actual = df_target.loc[10, target_col]
    assert np.isclose(expected, actual)

def test_leakage_checks(sample_ohlcv):
    assert verify_feature_leakage(sample_ohlcv, FEATURE_COLUMNS) is True
    assert verify_target_alignment(sample_ohlcv, horizon=FORECAST_HORIZON_DAYS) is True
