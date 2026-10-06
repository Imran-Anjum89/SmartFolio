"""
Dual Feedback Engine and Retraining Manager for SmartFolio.
Logs both prediction errors and portfolio performance, and enforces the Champion-Challenger gate.
"""
import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from config.settings import RETRAIN_INTERVAL_DAYS, MIN_FEEDBACK_OBSERVATIONS, CACHE_DIR
from ml.models.ensemble import ChampionChallengerGate

logger = logging.getLogger(__name__)

FEEDBACK_DIR = os.path.join(CACHE_DIR, "feedback")
os.makedirs(FEEDBACK_DIR, exist_ok=True)

class FeedbackEngine:
    def __init__(
        self,
        retrain_interval_days: int = RETRAIN_INTERVAL_DAYS,
        min_feedback_obs: int = MIN_FEEDBACK_OBSERVATIONS
    ):
        self.retrain_interval_days = retrain_interval_days
        self.min_feedback_obs = min_feedback_obs
        self.gate = ChampionChallengerGate()
        self.prediction_feedback_file = os.path.join(FEEDBACK_DIR, "prediction_feedback.json")
        self.portfolio_feedback_file = os.path.join(FEEDBACK_DIR, "portfolio_feedback.json")
        self.promotions_file = os.path.join(FEEDBACK_DIR, "model_promotions.json")

    def _load_records(self, filepath: str) -> List[Dict[str, Any]]:
        if os.path.exists(filepath):
            try:
                with open(filepath, "r") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _save_records(self, filepath: str, records: List[Dict[str, Any]]):
        with open(filepath, "w") as f:
            json.dump(records, f, indent=2)

    def log_prediction_feedback(
        self,
        symbol: str,
        prediction_date: str,
        model_version: str,
        predicted_return: float,
        actual_return: float
    ) -> Dict[str, Any]:
        """
        Record prediction vs actual return outcome and compute prediction errors.
        """
        error = actual_return - predicted_return
        abs_error = abs(error)
        sq_error = error ** 2

        entry = {
            "id": f"pred_{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "prediction_date": prediction_date,
            "model_version": model_version,
            "predicted_return": round(float(predicted_return), 6),
            "actual_return": round(float(actual_return), 6),
            "error": round(float(error), 6),
            "absolute_error": round(float(abs_error), 6),
            "squared_error": round(float(sq_error), 8)
        }

        records = self._load_records(self.prediction_feedback_file)
        records.append(entry)
        self._save_records(self.prediction_feedback_file, records)
        return entry

    def log_portfolio_feedback(
        self,
        portfolio_id: str,
        allocations: List[Dict[str, Any]],
        realized_return: float,
        realized_sharpe: float,
        realized_drawdown: float
    ) -> Dict[str, Any]:
        """
        Record actual real-world portfolio performance out-of-sample.
        """
        entry = {
            "id": f"port_{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
            "timestamp": datetime.now().isoformat(),
            "portfolio_id": portfolio_id,
            "allocations": allocations,
            "realized_return": round(float(realized_return), 4),
            "realized_sharpe": round(float(realized_sharpe), 4),
            "realized_drawdown": round(float(realized_drawdown), 4)
        }

        records = self._load_records(self.portfolio_feedback_file)
        records.append(entry)
        self._save_records(self.portfolio_feedback_file, records)
        return entry

    def get_prediction_feedback(self, limit: int = 100) -> List[Dict[str, Any]]:
        records = self._load_records(self.prediction_feedback_file)
        return records[-limit:]

    def get_portfolio_feedback(self, limit: int = 100) -> List[Dict[str, Any]]:
        records = self._load_records(self.portfolio_feedback_file)
        return records[-limit:]

    def get_promotions_history(self) -> List[Dict[str, Any]]:
        return self._load_records(self.promotions_file)

    def record_champion_challenger_decision(
        self,
        champion_version: str,
        challenger_version: str,
        report: Dict[str, Any]
    ):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "champion_version": champion_version,
            "challenger_version": challenger_version,
            "decision": report["decision"],
            "promoted": report["should_promote"],
            "champion_sharpe": report["champion_sharpe"],
            "challenger_sharpe": report["challenger_sharpe"],
            "sharpe_diff": report["sharpe_difference"],
            "champion_rmse": report["champion_rmse"],
            "challenger_rmse": report["challenger_rmse"],
            "reason": report["reason"]
        }
        records = self._load_records(self.promotions_file)
        records.append(entry)
        self._save_records(self.promotions_file, records)
        return entry
