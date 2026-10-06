"""
Ensemble Forecaster and Champion-Challenger Model Selection Engine for SmartFolio.
Implements validation-weighted ensembling and the portfolio-level Champion-Challenger promotion gate.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from ml.models.baseline import HistoricalMeanBaseline
from ml.models.xgboost_model import SmartFolioXGBoost
from ml.models.lstm_model import SmartFolioLSTM
from ml.evaluation.evaluate import compute_prediction_metrics, compute_portfolio_metrics

class SmartFolioEnsemble:
    """
    Weighted combination of XGBoost and PyTorch LSTM predictions.
    Weights are determined via validation performance (inverse RMSE) rather than arbitrary 50/50.
    """
    def __init__(self, weight_xgb: float = 0.6, weight_lstm: float = 0.4):
        self.weight_xgb = weight_xgb
        self.weight_lstm = weight_lstm
        self.xgb_model: Optional[SmartFolioXGBoost] = None
        self.lstm_model: Optional[SmartFolioLSTM] = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> "SmartFolioEnsemble":
        # Train both models
        self.xgb_model = SmartFolioXGBoost().fit(X_train, y_train)
        self.lstm_model = SmartFolioLSTM(epochs=20).fit(X_train, y_train)

        # Evaluate on validation split to set optimal ensemble weights
        preds_xgb = self.xgb_model.predict(X_val)
        preds_lstm = self.lstm_model.predict(X_val)

        rmse_xgb = compute_prediction_metrics(y_val.values, preds_xgb)["rmse"]
        rmse_lstm = compute_prediction_metrics(y_val.values, preds_lstm)["rmse"]

        # Inverse RMSE weighting
        inv_xgb = 1.0 / (rmse_xgb + 1e-6)
        inv_lstm = 1.0 / (rmse_lstm + 1e-6)
        total_inv = inv_xgb + inv_lstm

        self.weight_xgb = float(inv_xgb / total_inv)
        self.weight_lstm = float(inv_lstm / total_inv)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.xgb_model is None or self.lstm_model is None:
            raise ValueError("Ensemble models are not trained yet.")
        preds_xgb = self.xgb_model.predict(X)
        preds_lstm = self.lstm_model.predict(X)
        return (self.weight_xgb * preds_xgb) + (self.weight_lstm * preds_lstm)


class ChampionChallengerGate:
    """
    Statistically evaluates whether a newly retrained Challenger model should be promoted over the Champion.
    Critically, alignment with Gap 1 requires comparing portfolio-level risk-adjusted performance (Sharpe Ratio),
    not just raw RMSE.
    """
    def __init__(self, min_sharpe_improvement: float = 0.02):
        self.min_sharpe_improvement = min_sharpe_improvement

    def evaluate_promotion(
        self,
        champion_portfolio_returns: np.ndarray,
        challenger_portfolio_returns: np.ndarray,
        champion_rmse: float,
        challenger_rmse: float
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Evaluate if Challenger beats Champion on unseen out-of-sample data.
        Returns (should_promote, comparison_report).
        """
        champ_metrics = compute_portfolio_metrics(champion_portfolio_returns)
        chall_metrics = compute_portfolio_metrics(challenger_portfolio_returns)

        champ_sharpe = champ_metrics["sharpe_ratio"]
        chall_sharpe = chall_metrics["sharpe_ratio"]
        sharpe_diff = chall_sharpe - champ_sharpe
        rmse_diff = challenger_rmse - champion_rmse  # negative is better

        # Promotion rule: Challenger must provide superior out-of-sample Sharpe Ratio
        # without disastrous prediction degradation
        is_sharpe_superior = sharpe_diff >= self.min_sharpe_improvement
        is_rmse_acceptable = rmse_diff <= 0.05  # Challenger RMSE is not substantially worse

        should_promote = bool(is_sharpe_superior and is_rmse_acceptable)

        report = {
            "should_promote": should_promote,
            "decision": "PROMOTE_CHALLENGER" if should_promote else "REJECT_CHALLENGER_KEEP_CHAMPION",
            "champion_sharpe": champ_sharpe,
            "challenger_sharpe": chall_sharpe,
            "sharpe_difference": round(sharpe_diff, 4),
            "champion_rmse": round(champion_rmse, 6),
            "challenger_rmse": round(challenger_rmse, 6),
            "reason": (
                f"Challenger Sharpe ({chall_sharpe:.2f}) vs Champion ({champ_sharpe:.2f}) with ΔSharpe {sharpe_diff:.2f}."
                if should_promote else
                f"Challenger did not achieve required Sharpe improvement (ΔSharpe {sharpe_diff:.2f} < threshold {self.min_sharpe_improvement})."
            )
        }
        return should_promote, report
