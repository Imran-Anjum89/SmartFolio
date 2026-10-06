"""
Feature Service for SmartFolio.
Prepares processed training and inference feature matrices with data integrity validation.
"""
import pandas as pd
from typing import Dict, List, Optional
from config.settings import FEATURE_COLUMNS, FORECAST_HORIZON_DAYS
from ml.features.indicators import prepare_feature_target_dataset, compute_technical_indicators
from ml.features.leakage_checks import verify_feature_leakage, verify_target_alignment

class FeatureService:
    def __init__(self, feature_cols: List[str] = FEATURE_COLUMNS, horizon: int = FORECAST_HORIZON_DAYS):
        self.feature_cols = feature_cols
        self.horizon = horizon

    def build_features_for_stock(self, df: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
        """
        Build technical features and optionally target. Drops NaN rows from lookback window.
        """
        if include_target:
            return prepare_feature_target_dataset(df, self.feature_cols, horizon=self.horizon)
        else:
            df_feat = compute_technical_indicators(df)
            valid_cols = self.feature_cols + ["close", "date"] if "date" in df_feat.columns else self.feature_cols + ["close"]
            return df_feat.dropna(subset=[c for c in valid_cols if c in df_feat.columns]).reset_index(drop=True)

    def validate_leakage_safety(self, df: pd.DataFrame) -> bool:
        """Run temporal integrity and leakage assertions on the dataset."""
        verify_feature_leakage(df, self.feature_cols)
        verify_target_alignment(df, horizon=self.horizon)
        return True
