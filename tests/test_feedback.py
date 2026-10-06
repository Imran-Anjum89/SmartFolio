"""
Unit Tests for Dual Feedback Engine and Portfolio Sharpe Champion-Challenger Gate.
"""
import pytest
import numpy as np
from ml.models.ensemble import ChampionChallengerGate
from ml.feedback.feedback_loop import FeedbackEngine

def test_champion_challenger_promotion_gate():
    gate = ChampionChallengerGate(min_sharpe_improvement=0.02)
    
    # Case 1: Challenger has higher Sharpe -> Should PROMOTE
    champ_rets = np.array([0.0006, 0.0004, -0.0002, 0.0005, 0.0003] * 20)
    chall_rets = np.array([0.0015, 0.0012, 0.0008, 0.0014, 0.0010] * 20)
    
    should_promote, report = gate.evaluate_promotion(
        champion_portfolio_returns=champ_rets,
        challenger_portfolio_returns=chall_rets,
        champion_rmse=0.025,
        challenger_rmse=0.023
    )
    assert should_promote is True
    assert report["decision"] == "PROMOTE_CHALLENGER"
    assert report["sharpe_difference"] > 0

    # Case 2: Challenger has worse Sharpe -> Should REJECT
    chall_bad_rets = np.array([-0.0005, -0.0002, 0.0001, -0.0004, -0.0003] * 20)
    should_promote_bad, report_bad = gate.evaluate_promotion(
        champion_portfolio_returns=champ_rets,
        challenger_portfolio_returns=chall_bad_rets,
        champion_rmse=0.025,
        challenger_rmse=0.024  # Even if RMSE is marginally better, worse Sharpe must reject (Gap 1)!
    )
    assert should_promote_bad is False
    assert report_bad["decision"] == "REJECT_CHALLENGER_KEEP_CHAMPION"

def test_feedback_logging():
    engine = FeedbackEngine()
    entry = engine.log_prediction_feedback(
        symbol="TCS.NS",
        prediction_date="2023-09-10",
        model_version="v1.0",
        predicted_return=0.018,
        actual_return=0.007
    )
    assert entry["symbol"] == "TCS.NS"
    assert np.isclose(entry["error"], -0.011, atol=1e-5)
    assert np.isclose(entry["absolute_error"], 0.011, atol=1e-5)
