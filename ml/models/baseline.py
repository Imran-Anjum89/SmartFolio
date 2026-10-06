"""
Historical Mean Baseline Return Forecaster.
Provides a standard non-ML benchmark estimating future return via rolling historical mean.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any

class HistoricalMeanBaseline:
    def __init__(self, rolling_window: int = 63):
        self.rolling_window = rolling_window
        self.mean_return_ = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HistoricalMeanBaseline":
        """Fit baseline by computing the historical mean target return."""
        self.mean_return_ = float(y.mean()) if len(y) > 0 else 0.0
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict constant historical mean return for all instances."""
        return np.full(shape=(len(X),), fill_value=self.mean_return_)

    def get_params(self) -> Dict[str, Any]:
        return {"model_type": "HistoricalMean", "rolling_window": self.rolling_window, "mean_return": self.mean_return_}
