"""
Machine Learning Service for SmartFolio.
Manages model training across assets, model selection, versioning, and expected return forecasting.
"""
import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from config.settings import FEATURE_COLUMNS, FORECAST_HORIZON_DAYS, RANDOM_SEED
from ml.models.baseline import HistoricalMeanBaseline
from ml.models.xgboost_model import SmartFolioXGBoost
from ml.models.lstm_model import SmartFolioLSTM
from ml.models.ensemble import SmartFolioEnsemble
from ml.evaluation.evaluate import compute_prediction_metrics
from backend.app.services.feature_service import FeatureService

logger = logging.getLogger(__name__)

class MLService:
    def __init__(self, feature_service: Optional[FeatureService] = None):
        self.feature_service = feature_service or FeatureService()
        self.models_cache: Dict[str, Dict[str, Any]] = {}
        self.current_model_version: str = "v1.0"
        self.active_model_name: str = "ensemble"  # default champion candidate

    def train_models_for_symbol(
        self,
        symbol: str,
        df_raw: pd.DataFrame,
        train_ratio: float = 0.8
    ) -> Dict[str, Any]:
        """
        Train Baseline, XGBoost, LSTM, and Ensemble for a given symbol using strict time-series split.
        Returns trained models, evaluation metrics, and feature importances.
        """
        df_feat = self.feature_service.build_features_for_stock(df_raw, include_target=True)
        target_col = f"target_{FORECAST_HORIZON_DAYS}d"
        
        if len(df_feat) < 80:
            raise ValueError(f"Insufficient samples for {symbol}: only {len(df_feat)} rows available.")

        # Temporal split (no random shuffling!)
        split_idx = int(len(df_feat) * train_ratio)
        val_idx = int(len(df_feat) * (train_ratio + 0.1))  # 10% validation, remaining 10% test
        
        train_df = df_feat.iloc[:split_idx]
        val_df = df_feat.iloc[split_idx:val_idx]
        test_df = df_feat.iloc[val_idx:] if val_idx < len(df_feat) else df_feat.iloc[split_idx:]

        X_train, y_train = train_df[FEATURE_COLUMNS], train_df[target_col]
        X_val, y_val = val_df[FEATURE_COLUMNS], val_df[target_col]
        X_test, y_test = test_df[FEATURE_COLUMNS], test_df[target_col]

        # 1. Historical Mean Baseline
        base_model = HistoricalMeanBaseline().fit(X_train, y_train)
        base_preds = base_model.predict(X_test)
        base_metrics = compute_prediction_metrics(y_test.values, base_preds)

        # 2. XGBoost
        xgb_model = SmartFolioXGBoost().fit(X_train, y_train)
        xgb_preds = xgb_model.predict(X_test)
        xgb_metrics = compute_prediction_metrics(y_test.values, xgb_preds)

        # 3. PyTorch LSTM
        lstm_model = SmartFolioLSTM(epochs=18).fit(X_train, y_train)
        lstm_preds = lstm_model.predict(X_test)
        lstm_metrics = compute_prediction_metrics(y_test.values, lstm_preds)

        # 4. Ensemble
        ensemble_model = SmartFolioEnsemble().fit(X_train, y_train, X_val, y_val)
        ens_preds = ensemble_model.predict(X_test)
        ens_metrics = compute_prediction_metrics(y_test.values, ens_preds)

        # Determine best performing model on test set
        models_eval = {
            "baseline": base_metrics["rmse"],
            "xgboost": xgb_metrics["rmse"],
            "lstm": lstm_metrics["rmse"],
            "ensemble": ens_metrics["rmse"]
        }
        best_model_name = min(models_eval, key=models_eval.get)

        trained_bundle = {
            "symbol": symbol,
            "version": self.current_model_version,
            "models": {
                "baseline": base_model,
                "xgboost": xgb_model,
                "lstm": lstm_model,
                "ensemble": ensemble_model
            },
            "metrics": {
                "baseline": base_metrics,
                "xgboost": xgb_metrics,
                "lstm": lstm_metrics,
                "ensemble": ens_metrics
            },
            "best_model": best_model_name,
            "feature_importance": xgb_model.get_feature_importance(),
            "latest_features": df_feat[FEATURE_COLUMNS].iloc[[-1]]
        }

        self.models_cache[symbol] = trained_bundle
        return trained_bundle

    def predict_expected_returns(
        self,
        symbols_data: Dict[str, pd.DataFrame],
        model_name: str = "ensemble"
    ) -> Dict[str, float]:
        """
        Generate expected return forecasts mu_hat for each asset using the specified model.
        Returns dict of symbol -> expected_return (annualized).
        """
        expected_returns = {}
        for symbol, df_raw in symbols_data.items():
            if symbol not in self.models_cache:
                self.train_models_for_symbol(symbol, df_raw)

            bundle = self.models_cache[symbol]
            latest_feat = bundle["latest_features"]
            
            selected_model = bundle["models"].get(model_name, bundle["models"]["ensemble"])
            pred_5d = float(selected_model.predict(latest_feat)[0])
            
            # Annualize the 5-day return prediction: (1 + r_5d)^(252 / 5) - 1
            ann_return = float((1.0 + pred_5d) ** (252.0 / FORECAST_HORIZON_DAYS) - 1.0) if pred_5d > -0.9 else -0.9
            # Cap extreme predictions within financial realism (-50% to +100% annual)
            ann_return = float(np.clip(ann_return, -0.50, 1.00))
            expected_returns[symbol] = round(ann_return, 4)

        return expected_returns
