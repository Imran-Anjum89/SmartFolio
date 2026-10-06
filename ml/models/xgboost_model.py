"""
XGBoost Return Forecasting Model for SmartFolio.
Models non-linear feature interactions on tabular financial technical indicators.
"""
import xgboost as xgb
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from config.settings import RANDOM_SEED

class SmartFolioXGBoost:
    def __init__(
        self,
        n_estimators: int = 150,
        max_depth: int = 4,
        learning_rate: float = 0.03,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = RANDOM_SEED
    ):
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state,
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "n_jobs": -1
        }
        self.model = xgb.XGBRegressor(**self.params)
        self.feature_names: List[str] = []
        self.feature_importances_: Dict[str, float] = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[List[tuple]] = None,
        verbose: bool = False
    ) -> "SmartFolioXGBoost":
        """Fit XGBoost model and store feature importances."""
        self.feature_names = list(X.columns)
        if eval_set:
            self.model.fit(X, y, eval_set=eval_set, verbose=verbose)
        else:
            self.model.fit(X, y, verbose=verbose)

        # Record feature importances
        raw_importances = self.model.feature_importances_
        self.feature_importances_ = {
            feat: float(imp) for feat, imp in zip(self.feature_names, raw_importances)
        }
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Forecast future return percentages."""
        return self.model.predict(X)

    def get_feature_importance(self) -> Dict[str, float]:
        """Return dictionary of feature name -> importance score."""
        return dict(sorted(self.feature_importances_.items(), key=lambda item: item[1], reverse=True))
