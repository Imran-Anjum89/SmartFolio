"""
Dual Feedback Engine and Retraining Endpoints for SmartFolio.
"""
from fastapi import APIRouter, HTTPException
from backend.app.schemas.schemas import PredictionFeedbackRequest, RetrainRequest
from backend.app.services.data_service import MarketDataService
from backend.app.services.ml_service import MLService
from backend.app.services.feedback_service import FeedbackService

router = APIRouter(prefix="/feedback", tags=["Feedback"])

data_service = MarketDataService()
ml_service = MLService()
feedback_service = FeedbackService(ml_service, data_service)

@router.post("/prediction")
def record_prediction_feedback(req: PredictionFeedbackRequest):
    """Record actual realized outcome for a previous return prediction."""
    try:
        res = feedback_service.record_prediction_outcome(
            symbol=req.symbol,
            predicted_return=req.predicted_return,
            actual_return=req.actual_return,
            prediction_date=req.prediction_date
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/summary")
def get_feedback_summary():
    """Retrieve feedback performance statistics and Champion-Challenger history."""
    try:
        return feedback_service.get_feedback_summary()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/retrain")
def trigger_retraining(req: RetrainRequest):
    """
    Trigger scheduled/on-demand adaptive model retraining.
    Trains a Challenger model and tests it against the Champion via the Portfolio Sharpe Gate.
    """
    try:
        report = feedback_service.trigger_adaptive_retraining(symbol=req.symbol)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")
