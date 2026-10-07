"""
Explainability Module for XGBoost and Feature Attribution (Section 29).
Computes normalized feature importance rankings and contributions.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any

class FeatureAttributionExplainer:
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names

    def get_feature_importances(self, xgb_model) -> List[Dict[str, Any]]:
        """
        Extract normalized feature importances from trained XGBoost model.
        """
        if hasattr(xgb_model, "feature_importances_"):
            raw_imp = xgb_model.feature_importances_
        elif hasattr(xgb_model, "model") and hasattr(xgb_model.model, "feature_importances_"):
            raw_imp = xgb_model.model.feature_importances_
        else:
            raw_imp = np.ones(len(self.feature_names)) / len(self.feature_names)

        total = np.sum(raw_imp) or 1.0
        norm_imp = raw_imp / total

        results = []
        for name, score in zip(self.feature_names, norm_imp):
            results.append({
                "feature": name,
                "importance": round(float(score), 4),
                "importance_pct": round(float(score * 100.0), 2)
            })

        results.sort(key=lambda x: x["importance"], reverse=True)
        return results
