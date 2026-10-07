"""
Unit Tests for Machine Learning & Deep Learning Models.
"""
import pytest
import numpy as np
import pandas as pd
from ml.models.baseline import HistoricalMeanBaseline
from ml.models.xgboost_model import SmartFolioXGBoost
from ml.models.lstm_model import SmartFolioLSTM
from ml.evaluation.evaluate import compute_prediction_metrics
from config.settings import FEATURE_COLUMNS

@pytest.fixture
def sample_dataset():
    np.random.seed(42)
    n = 120
    X = pd.DataFrame(np.random.randn(n, len(FEATURE_COLUMNS)), columns=FEATURE_COLUMNS)
    # Target return has slight linear dependence on first feature plus noise
    y = pd.Series(0.05 * X[FEATURE_COLUMNS[0]] + np.random.normal(0, 0.02, n))
    return X, y

def test_historical_mean_baseline(sample_dataset):
    X, y = sample_dataset
    model = HistoricalMeanBaseline().fit(X[:80], y[:80])
    preds = model.predict(X[80:])
    assert len(preds) == len(X) - 80
    assert np.isclose(preds[0], y[:80].mean())

def test_xgboost_model(sample_dataset):
    X, y = sample_dataset
    model = SmartFolioXGBoost(n_estimators=30, max_depth=3).fit(X[:80], y[:80])
    preds = model.predict(X[80:])
    assert len(preds) == len(X) - 80
    importances = model.get_feature_importance()
    assert len(importances) == len(FEATURE_COLUMNS)
    metrics = compute_prediction_metrics(y[80:].values, preds)
    assert "rmse" in metrics and "mae" in metrics

def test_lstm_model(sample_dataset):
    X, y = sample_dataset
    model = SmartFolioLSTM(epochs=5, lookback_window=10).fit(X[:80], y[:80])
    preds = model.predict(X[80:])
    assert len(preds) == len(X) - 80
    assert not np.isnan(preds).any()
